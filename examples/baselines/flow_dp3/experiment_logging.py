"""可选 W&B 日志：关闭时无 SDK 依赖；记录操作不改变训练随机状态。"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import random
import secrets

import numpy as np
import torch


def add_wandb_args(parser):
    parser.add_argument("--wandb-mode", choices=("disabled", "online", "offline"), default="disabled")
    parser.add_argument("--wandb-project", help="默认 flow_dp3；恢复时默认沿用保存的 project")
    parser.add_argument("--wandb-entity", help="W&B 个人或团队 workspace；默认使用登录账号")
    parser.add_argument("--wandb-name", help="网页显示名称；默认输出目录名")
    parser.add_argument("--wandb-run-id", help="可选 run ID；恢复训练时自动沿用已保存的 ID")


def require_wandb(args):
    if args.wandb_mode == "disabled":
        return None
    try:
        return importlib.import_module("wandb")
    except ModuleNotFoundError as exc:
        if exc.name != "wandb":
            raise
        raise RuntimeError("未安装 W&B。请在本项目 venv 中安装 requirements-logging.txt，或 --wandb-mode disabled") from exc


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class ExperimentLogger:
    def __init__(self, args, output, config, job_type, resume=False, previous=None, group=None):
        self.run = None
        self.metadata = previous
        self.module = None
        if args.wandb_mode == "disabled":
            return
        with preserve_rng():
            self.module = require_wandb(args)
            output = Path(output)
            marker = output / "wandb_run.json"
            if resume and not previous and marker.is_file():
                previous = json.loads(marker.read_text())
            project = args.wandb_project or (previous or {}).get("project") or "flow_dp3"
            entity = args.wandb_entity or (previous or {}).get("entity")
            run_id = args.wandb_run_id or (previous or {}).get("id") or secrets.token_hex(4)
            if resume and previous and (
                run_id != previous["id"] or project != previous["project"] or
                (previous.get("entity") and entity != previous["entity"])
            ):
                raise ValueError("恢复 W&B 日志时 run ID、project、entity 必须与已保存信息一致")
            output.mkdir(parents=True, exist_ok=True)
            try:
                self.run = self.module.init(
                    project=project, entity=entity, id=run_id,
                    name=args.wandb_name or output.name, group=group, job_type=job_type,
                    mode=args.wandb_mode, dir=str(output), config=config, save_code=False, allow_val_change=True,
                    resume="allow" if resume and args.wandb_mode == "online" else None,
                    settings=self.module.Settings(x_disable_stats=True, init_timeout=30),
                )
                self.run.define_metric("global_step")
                self.run.define_metric("train/*", step_metric="global_step")
                self.run.define_metric("val/*", step_metric="global_step")
                self.run.define_metric("episode")
                self.run.define_metric("eval/*", step_metric="episode")
                self.metadata = {"id": self.run.id, "project": self.run.project, "entity": self.run.entity,
                                 "mode": args.wandb_mode,
                                 "url": self.run.url if args.wandb_mode == "online" else None}
                marker.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n")
            except Exception:
                if self.run is not None:
                    self.run.finish(exit_code=1)
                raise
        if self.metadata["url"]:
            print(f"W&B 页面：{self.metadata['url']}", flush=True)
        else:
            print(f"W&B 离线日志：{output / 'wandb'}；同步后可在网页查看", flush=True)

    def log_training(self, record):
        if self.run is None:
            return
        keys = {"train_loss": "train/loss", "val_loss": "val/loss", "grad_norm": "train/grad_norm",
                "lr": "train/lr", "ema_decay": "train/ema_decay"}
        data = {keys[key]: value for key, value in record.items() if key in keys}
        data["global_step"] = record["step"]
        with preserve_rng():
            self.run.log(data)

    def log_episode(self, result, index, upload_video=False, fps=None):
        if self.run is None:
            return
        data = {f"eval/{key}": int(value) if isinstance(value, bool) else value
                for key, value in result.items() if isinstance(value, (int, float))}
        data["episode"] = index
        with preserve_rng():
            if upload_video:
                data["eval/video"] = self.module.Video(result["video_path"], fps=fps, format="mp4")
            self.run.log(data)

    def finish(self, summary=None, exit_code=0):
        if self.run is None:
            return
        with preserve_rng():
            if summary:
                self.run.summary.update(summary)
            self.run.finish(exit_code=exit_code)
        self.run = None
