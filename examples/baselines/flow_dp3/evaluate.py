"""加载独立 checkpoint，以观测历史和动作块在 CPU 仿真中闭环评估。"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

import torch

from experiment_logging import ExperimentLogger, add_wandb_args, require_wandb, preserve_rng
from obs_adapter import config_from_contract, adapt_observation, make_env
from runtime_utils import load_policy, select_device, sha256


def evaluate(args):
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("episodes 与 max_steps 必须为正数")
    if args.output.exists():
        raise FileExistsError(f"报告已存在：{args.output}；请选择新路径")
    if args.video_dir and not args.save_video:
        raise ValueError("--video-dir 需要同时指定 --save-video")
    if args.video_fps is not None and args.video_fps < 1:
        raise ValueError("video-fps 必须为正整数")
    if args.wandb_upload_videos and (not args.save_video or args.wandb_mode == "disabled"):
        raise ValueError("上传 W&B 视频需要 --save-video 和 online/offline 的 --wandb-mode")
    video_dir = (args.video_dir or args.output.parent / (args.output.stem + "-videos")).resolve()
    if args.save_video:
        for seed in range(args.start_seed, args.start_seed + args.episodes):
            if (video_dir / f"seed_{seed}.mp4").exists():
                raise FileExistsError(f"视频已存在：{video_dir / f'seed_{seed}.mp4'}；请选择新目录")
    with preserve_rng():
        require_wandb(args)
    device = select_device(args.device)
    torch.manual_seed(args.policy_seed)
    policy, checkpoint = load_policy(args.checkpoint, device, use_ema=not args.raw_weights)
    contract = checkpoint["contract"]
    config = config_from_contract(contract)
    if policy.config.length_scale != config.length_scale:
        raise ValueError("checkpoint 中数据与策略长度尺度不一致")
    env = make_env(max_episode_steps=args.max_steps, render_mode="rgb_array" if args.save_video else None,
                   contract=contract)
    video_fps = args.video_fps or env.unwrapped.control_freq
    if args.save_video:
        from mani_skill.utils.wrappers.record import RecordEpisode
        env = RecordEpisode(env, output_dir=str(video_dir), save_trajectory=False, save_video=True,
                            save_on_reset=False, info_on_video=False, video_fps=video_fps)
    results = []
    started = time.monotonic()
    logger = None
    completed = False
    try:
        logger = ExperimentLogger(args, args.output.parent / (args.output.stem + "-logging"),
                                  {"policy": checkpoint["config"]["policy"], "contract": contract,
                                   "checkpoint": str(args.checkpoint.resolve()), "checkpoint_step": checkpoint["step"],
                                   "source_training_run": checkpoint.get("wandb"), "device": str(device),
                                   "max_steps": args.max_steps, "start_seed": args.start_seed,
                                   "policy_seed": args.policy_seed, "save_video": args.save_video},
                                  "eval", group=(checkpoint.get("wandb") or {}).get("id"))
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
            if args.save_video:
                encode_start = time.monotonic()
                env.flush_video(name=f"seed_{seed}", verbose=True)
                video_path = video_dir / f"seed_{seed}.mp4"
                if not video_path.is_file() or not video_path.stat().st_size:
                    raise RuntimeError(f"视频录制未生成有效文件：{video_path}")
                result["video_path"] = str(video_path)
                result["video_encoding_seconds"] = time.monotonic() - encode_start
            results.append(result)
            print(json.dumps(result), flush=True)
            logger.log_episode(result, len(results), upload_video=args.wandb_upload_videos, fps=video_fps)
        completed = True
    finally:
        try:
            env.close()
        finally:
            if logger is not None:
                logger.finish({"eval/success_once_rate": sum(r["success_once"] for r in results) / max(1, len(results)),
                               "eval/success_end_rate": sum(r["success_end"] for r in results) / max(1, len(results)),
                               "eval/episodes": len(results)}, exit_code=0 if completed else 1)
    report = {"algorithm": "flow_dp3", "encoder": "ee_relation_pointnetpp",
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
              "checkpoint_step": checkpoint["step"], "weights": "raw" if args.raw_weights else "ema",
              "device": str(device), "contract": contract, "policy_seed": args.policy_seed,
              "max_steps": args.max_steps, "episodes": results,
              "success_once_rate": sum(r["success_once"] for r in results) / len(results),
              "success_end_rate": sum(r["success_end"] for r in results) / len(results),
              "elapsed_seconds": time.monotonic() - started,
              "save_video": args.save_video, "video_fps": video_fps if args.save_video else None,
              "wandb": logger.metadata if logger is not None else None}
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
    parser.add_argument("--save-video", action="store_true", help="保存每局 MP4，默认不录制")
    parser.add_argument("--video-dir", type=Path, help="默认 <报告文件名>-videos 目录")
    parser.add_argument("--video-fps", type=int, help="默认使用任务控制频率")
    add_wandb_args(parser)
    parser.add_argument("--wandb-upload-videos", action="store_true", help="将已录制的视频同时记录到 W&B")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
