"""独立配置、设备与 checkpoint 工具。"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
import yaml

from policy import FlowDP3, PolicyConfig


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "pickcube.yaml"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or set(config) != {"policy", "training"}:
        raise ValueError("配置必须包含 policy、training 两个字段")
    policy_config = PolicyConfig(**config["policy"])
    fields = {key for key in ("state_dim", "action_dim") if key in config["policy"]}
    config["policy"] = asdict(policy_config)
    for key in {"state_dim", "action_dim"} - fields:
        del config["policy"][key]
    required = {"seed", "batch_size", "steps", "lr", "betas", "weight_decay", "warmup_steps",
                "val_ratio", "val_every", "val_batches", "checkpoint_every", "grad_clip"}
    if set(config["training"]) != required:
        raise ValueError(f"training 配置键不匹配：{set(config['training']) ^ required}")
    return config


def select_device(name):
    if name not in ("cpu", "cuda:0"):
        raise ValueError("当前入口支持 cpu 或 cuda:0")
    if name == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("当前进程无法访问 PPU；请在 DSW 终端初始化 SDK 后运行，或显式 --device cpu 做烟雾测试")
    return torch.device(name)


def to_device(batch, device):
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    return batch.to(device)


def canonical(value):
    return json.dumps(value, sort_keys=True)


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".partial.pt")
    import shutil
    # Tensor storage dominates these checkpoints; budget for a new temporary file.
    storages = {}
    def collect(value):
        if torch.is_tensor(value):
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
    collect(payload)
    needed = int(sum(storages.values()) * 1.05) + 1024*1024
    if shutil.disk_usage(path.parent).free < needed:
        raise OSError(f"checkpoint 临时写入需约 {needed/1024**3:.2f} GiB 空闲空间，请扩容或更换输出存储")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_policy(path, device, use_ema=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("不支持的 FlowDP3 checkpoint 格式")
    from obs_adapter import config_from_contract
    config_from_contract(checkpoint["contract"])
    policy_config = PolicyConfig(**checkpoint["config"]["policy"])
    if any(getattr(policy_config, key) != checkpoint["contract"][key] for key in ("state_dim", "action_dim")):
        raise ValueError("checkpoint 模型维度与任务契约不匹配")
    policy = FlowDP3(policy_config)
    policy.load_state_dict(checkpoint["ema" if use_ema else "model"], strict=True)
    policy.to(device).eval()
    return policy, checkpoint
