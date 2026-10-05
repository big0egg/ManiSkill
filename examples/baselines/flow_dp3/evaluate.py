"""加载独立 checkpoint，以观测历史和动作块在 CPU 仿真中闭环评估。"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

import torch

from obs_adapter import ObservationConfig, adapt_observation, make_env
from runtime_utils import canonical, load_policy, select_device, sha256


def evaluate(args):
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("episodes 与 max_steps 必须为正数")
    if args.output.exists():
        raise FileExistsError(f"报告已存在：{args.output}；请选择新路径")
    device = select_device(args.device)
    torch.manual_seed(args.policy_seed)
    policy, checkpoint = load_policy(args.checkpoint, device, use_ema=not args.raw_weights)
    contract = checkpoint["contract"]
    config = ObservationConfig(**contract["pointcloud"])
    if canonical(config.contract()) != canonical(contract):
        raise ValueError("checkpoint 观测/环境契约与当前适配器不同")
    if policy.config.length_scale != config.length_scale:
        raise ValueError("checkpoint 中数据与策略长度尺度不一致")
    env = make_env(max_episode_steps=args.max_steps)
    results = []
    started = time.monotonic()
    try:
        for seed in range(args.start_seed, args.start_seed + args.episodes):
            reset_start = time.monotonic()
            obs, _ = env.reset(seed=seed)
            reset_seconds = time.monotonic() - reset_start
            preprocess_start = time.monotonic()
            adapted, _ = adapt_observation(obs, env.unwrapped.agent, config)
            preprocess_seconds = time.monotonic() - preprocess_start
            history = deque([adapted] * policy.config.n_obs_steps, maxlen=policy.config.n_obs_steps)
            pending = deque()
            success_once, success_end, clipped, action_values, inferences = False, False, 0, 0, 0
            inference_seconds = 0.0
            simulation_seconds = 0.0
            executed_clipped, executed_values = 0, 0
            for step in range(args.max_steps):
                if not pending:
                    policy_obs = {key: torch.stack([frame[key] for frame in history], dim=1).to(device)
                                  for key in adapted}
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_start = time.monotonic()
                    prediction = policy.predict_action(policy_obs)["action"][0].cpu()
                    inference_seconds += time.monotonic() - inference_start
                    inferences += 1
                    clipped += int((prediction.abs() > 1).sum())
                    action_values += prediction.numel()
                    pending.extend(prediction)
                action = pending.popleft()
                executed_clipped += int((action.abs() > 1).sum())
                executed_values += action.numel()
                step_start = time.monotonic()
                obs, _, _, truncated, info = env.step(action.clamp(-1, 1).numpy())
                simulation_seconds += time.monotonic() - step_start
                success_end = bool(torch.as_tensor(info["success"]).item())
                success_once |= success_end
                # 不因瞬时 success/terminated 提前结束；统一执行到任务评估上限。
                if bool(torch.as_tensor(truncated).item()):
                    break
                preprocess_start = time.monotonic()
                adapted, _ = adapt_observation(obs, env.unwrapped.agent, config)
                preprocess_seconds += time.monotonic() - preprocess_start
                history.append(adapted)
            result = {"seed": seed, "steps": step + 1, "success_once": success_once,
                      "success_end": success_end, "inferences": inferences,
                      "mean_inference_seconds": inference_seconds / max(1, inferences),
                      "reset_seconds": reset_seconds, "preprocess_seconds": preprocess_seconds,
                      "simulation_and_render_seconds": simulation_seconds,
                      "predicted_action_clip_fraction": clipped / max(1, action_values),
                      "executed_action_clip_fraction": executed_clipped / max(1, executed_values)}
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        env.close()
    report = {"algorithm": "flow_dp3", "encoder": "ee_relation_pointnetpp",
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
              "checkpoint_step": checkpoint["step"], "weights": "raw" if args.raw_weights else "ema",
              "device": str(device), "contract": contract, "policy_seed": args.policy_seed,
              "max_steps": args.max_steps, "episodes": results,
              "success_once_rate": sum(r["success_once"] for r in results) / len(results),
              "success_end_rate": sum(r["success_end"] for r in results) / len(results),
              "elapsed_seconds": time.monotonic() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"成功率 once={report['success_once_rate']:.3f}, end={report['success_end_rate']:.3f}；报告={args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1000, help="与训练示范 seed 分离")
    parser.add_argument("--policy-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--raw-weights", action="store_true", help="默认使用 EMA")
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
