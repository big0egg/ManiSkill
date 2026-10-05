"""离线 FlowDP3 训练：按 episode 留出验证、训练集统计、EMA 和可恢复 checkpoint。"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import default_collate

from dataset import DemoDataset
from policy import FlowDP3, PolicyConfig, update_ema
from runtime_utils import (DEFAULT_CONFIG, canonical, load_config, save_checkpoint,
                           select_device, sha256, to_device)


def sample_batch(dataset, size, generator):
    indices = torch.randint(len(dataset), (size,), generator=generator).tolist()
    return default_collate([dataset[index] for index in indices])


def train(args):
    config = load_config(args.config)
    settings = config["training"]
    steps = args.steps if args.steps is not None else settings["steps"]
    batch_size = args.batch_size if args.batch_size is not None else settings["batch_size"]
    if steps < 1 or batch_size < 1 or settings["lr"] <= 0 or settings["grad_clip"] <= 0 or any(
        settings[key] < 1 for key in ("val_every", "val_batches", "checkpoint_every")
    ):
        raise ValueError("步数、批大小、学习率、裁剪阈值及检查频率必须为正数")
    device = select_device(args.device)
    torch.manual_seed(settings["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings["seed"])
    policy_config = PolicyConfig(**config["policy"])
    common = dict(horizon=policy_config.horizon, n_obs_steps=policy_config.n_obs_steps,
                  n_action_steps=policy_config.n_action_steps, val_ratio=settings["val_ratio"], seed=settings["seed"])
    training = DemoDataset(args.data, split="train", **common)
    validation = DemoDataset(args.data, split="val", **common)
    if training.contract["pointcloud"]["length_scale"] != policy_config.length_scale:
        raise ValueError("数据与策略 length_scale 不一致，拒绝重复或错误缩放")
    fingerprint = sha256(args.data)
    model = FlowDP3(policy_config).to(device)
    states, actions = training.normalizer_data()
    model.state_normalizer.fit(states)
    model.action_normalizer.fit(actions)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], betas=tuple(settings["betas"]),
                                 weight_decay=settings["weight_decay"], eps=1e-8)
    sampler = torch.Generator().manual_seed(settings["seed"] + 10)
    step, best_validation = 0, math.inf
    # 学习率以配置的总 steps 为基准；--steps 可用于提前停下，不改变预定调度。
    def lr_multiplier(index):
        warmup = settings["warmup_steps"]
        if index < warmup:
            return index / max(1, warmup)
        progress = min(1, (index - warmup) / max(1, settings["steps"] - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    output = args.output.resolve()
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1 or checkpoint["data_sha256"] != fingerprint or canonical(
            checkpoint["config"]
        ) != canonical(config) or canonical(checkpoint["contract"]) != canonical(training.contract):
            raise ValueError("resume 的格式、配置、数据指纹或输入契约不匹配")
        if checkpoint["batch_size"] != batch_size:
            raise ValueError("恢复训练需要相同 batch_size")
        model.load_state_dict(checkpoint["model"], strict=True)
        ema.load_state_dict(checkpoint["ema"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        sampler.set_state(checkpoint["sampler_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            if checkpoint["cuda_rng"] is None:
                raise ValueError("不能精确地将 CPU 训练恢复到 PPU；请使用同一计算设备")
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        elif checkpoint["cuda_rng"] is not None:
            raise ValueError("不能精确地将 PPU 训练恢复到 CPU")
        step = checkpoint["step"]
        best_validation = checkpoint["best_validation"]
        if output != Path(args.resume).resolve().parent:
            raise ValueError("resume 必须写回原训练输出目录，防止混淆运行记录")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空：{output}；请选择新目录或显式 --resume")
    if step >= steps:
        raise ValueError(f"目标 steps={steps} 必须大于已训练 step={step}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(args.config.read_text())
    run = {"algorithm": "flow_dp3", "encoder": "ee_relation_pointnetpp", "device": str(device),
           "torch": str(torch.__version__), "data": str(args.data.resolve()), "data_sha256": fingerprint,
           "contract": training.contract, "train_episodes": training.episodes, "val_episodes": validation.episodes,
           "train_windows": len(training), "val_windows": len(validation),
           "parameter_count": sum(p.numel() for p in model.parameters()), "batch_size": batch_size}
    (output / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(run, ensure_ascii=False), flush=True)
    started = time.monotonic()
    try:
        with (output / "metrics.jsonl").open("a") as metrics:
            while step < steps:
                model.train()
                batch = to_device(sample_batch(training, batch_size, sampler), device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("训练损失非有限")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                decay = update_ema(ema, model, step)
                step += 1
                record = {"step": step, "train_loss": float(loss.detach().cpu()),
                          "grad_norm": float(grad_norm.cpu()), "lr": scheduler.get_last_lr()[0], "ema_decay": decay}
                is_best = False
                if step % settings["val_every"] == 0 or step == steps:
                    # 固定独立验证随机流，并恢复训练 RNG；不让验证影响后续训练采样。
                    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []), torch.no_grad():
                        torch.manual_seed(settings["seed"] + 100)
                        generator = torch.Generator().manual_seed(settings["seed"] + 101)
                        losses = [ema.compute_loss(to_device(sample_batch(validation, batch_size, generator), device))
                                  for _ in range(settings["val_batches"])]
                        record["val_loss"] = float(torch.stack(losses).mean().cpu())
                    if not math.isfinite(record["val_loss"]):
                        raise FloatingPointError("验证损失非有限")
                    is_best = record["val_loss"] < best_validation
                    best_validation = min(best_validation, record["val_loss"])
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                if step == 1 or step % 10 == 0 or step == steps:
                    print(json.dumps(record), flush=True)
                if is_best or step % settings["checkpoint_every"] == 0 or step == steps:
                    payload = {"format_version": 1, "config": config, "contract": training.contract,
                               "data_sha256": fingerprint, "model": model.state_dict(), "ema": ema.state_dict(),
                               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                               "step": step, "batch_size": batch_size, "best_validation": best_validation,
                               "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
                               "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
                    save_checkpoint(output / "last.pt", payload)
                    if is_best:
                        save_checkpoint(output / "best.pt", payload)
    finally:
        training.close()
        validation.close()
    print(f"完成 {step} updates；{time.monotonic() - started:.1f}s；checkpoint={output / 'last.pt'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--steps", type=int, help="目标总更新次数，包含已恢复步数")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", type=Path)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
