"""加载独立 checkpoint，以观测历史和动作块在 CPU 仿真中闭环评估。"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from experiment_logging import ExperimentLogger, add_wandb_args, require_wandb, preserve_rng
from obs_adapter import config_from_contract, adapt_observation, make_env, validate_env, action_bounds, clip_action
from pickcube_metrics import pickcube_metrics
from task_registry import TASKS, get_task
from control_modes import CONTROL_CHOICES, check_control_mode
from runtime_utils import load_policy, select_device, sha256


def evaluate(args):
    if args.episodes < 1 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError("episodes 与 max_steps 必须为正数")
    if args.output.exists():
        raise FileExistsError(f"报告已存在：{args.output}；请选择新路径")
    trace_path = args.output.with_name(args.output.stem + "-trace.json")
    if getattr(args, "save_trace", False) and trace_path.exists():
        raise FileExistsError(trace_path)
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
    if getattr(args, "n_action_steps", None) is not None:
        policy.config = replace(policy.config, n_action_steps=args.n_action_steps)
    contract = checkpoint["contract"]
    check_control_mode(getattr(args, "control_mode", None), contract)
    config = config_from_contract(contract)
    if args.env_id is not None and args.env_id != contract["env_id"]:
        raise ValueError("指定任务与 checkpoint 不匹配")
    args.max_steps = args.max_steps or get_task(contract["env_id"]).max_steps
    print(f'实际任务={contract["env_id"]} | robot={contract["robot_uids"]} | '
          f'control_mode={contract["control_mode"]} | action_dim={contract["action_dim"]}', flush=True)
    if policy.config.length_scale != config.length_scale:
        raise ValueError("checkpoint 中数据与策略长度尺度不一致")
    env = make_env(max_episode_steps=args.max_steps, render_mode="rgb_array" if args.save_video else None,
                   contract=contract)
    validate_env(env, contract)
    video_fps = args.video_fps or env.unwrapped.control_freq
    if args.save_video:
        from mani_skill.utils.wrappers.record import RecordEpisode
        env = RecordEpisode(env, output_dir=str(video_dir), save_trajectory=False, save_video=True,
                            save_on_reset=False, info_on_video=False, video_fps=video_fps)
    results = []
    traces = []
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
            if getattr(args, "reset_policy_seed_per_episode", False):
                torch.manual_seed(args.policy_seed)
            reset_start = time.monotonic()
            obs, _ = env.reset(seed=seed)
            reset_seconds = time.monotonic() - reset_start
            preprocess_start = time.monotonic()
            adapted, _ = adapt_observation(obs, env.unwrapped.agent, config, contract=contract)
            preprocess_seconds = time.monotonic() - preprocess_start
            history = deque([adapted] * policy.config.n_obs_steps, maxlen=policy.config.n_obs_steps)
            pending = deque()
            success_once, success_end, clipped, action_values, inferences = False, False, 0, 0, 0
            inference_seconds = 0.0
            simulation_seconds = 0.0
            executed_clipped, executed_values = 0, 0
            low, high = (torch.tensor(x) for x in action_bounds(contract))
            episode_trace = []
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
                    clipped += int(((prediction < low) | (prediction > high)).sum())
                    action_values += prediction.numel()
                    pending.extend(prediction)
                action = pending.popleft()
                executed_clipped += int(((action < low) | (action > high)).sum())
                executed_values += action.numel()
                step_start = time.monotonic()
                executed = clip_action(action, contract)
                obs, _, _, truncated, info = env.step(executed.numpy())
                simulation_seconds += time.monotonic() - step_start
                success_end = bool(torch.as_tensor(info["success"]).item())
                success_once |= success_end
                if contract["env_id"] == "PickCube-v1":
                    episode_trace.append({"step": step + 1, **pickcube_metrics(env, info),
                                          "action": executed.tolist()})
                # 不因瞬时 success/terminated 提前结束；统一执行到任务评估上限。
                if bool(torch.as_tensor(truncated).item()):
                    break
                preprocess_start = time.monotonic()
                adapted, _ = adapt_observation(obs, env.unwrapped.agent, config, contract=contract)
                preprocess_seconds += time.monotonic() - preprocess_start
                history.append(adapted)
            result = {"seed": seed, "steps": step + 1, "success_once": success_once,
                      "success_end": success_end, "inferences": inferences,
                      "mean_inference_seconds": inference_seconds / max(1, inferences),
                      "reset_seconds": reset_seconds, "preprocess_seconds": preprocess_seconds,
                      "simulation_and_render_seconds": simulation_seconds,
                      "predicted_action_clip_fraction": clipped / max(1, action_values),
                      "executed_action_clip_fraction": executed_clipped / max(1, executed_values)}
            if episode_trace:
                result.update({"final_goal_error_m": episode_trace[-1]["goal_error_m"],
                               "min_goal_error_m": min(x["goal_error_m"] for x in episode_trace),
                               "final_arm_qvel_maxabs": episode_trace[-1]["arm_qvel_maxabs"],
                               "grasped_once": any(x["is_grasped"] for x in episode_trace),
                               "grasped_end": episode_trace[-1]["is_grasped"],
                               "success_steps": sum(x["success"] for x in episode_trace),
                               "last40_success_fraction": sum(x["success"] for x in episode_trace[-40:]) / min(40, len(episode_trace))})
                traces.append({"seed": seed, "steps": episode_trace})
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
              "n_action_steps": policy.config.n_action_steps,
              "reset_policy_seed_per_episode": getattr(args, "reset_policy_seed_per_episode", False),
              "success_once_rate": sum(r["success_once"] for r in results) / len(results),
              "success_end_rate": sum(r["success_end"] for r in results) / len(results),
              "elapsed_seconds": time.monotonic() - started,
              "save_video": args.save_video, "video_fps": video_fps if args.save_video else None,
              "wandb": logger.metadata if logger is not None else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if getattr(args, "save_trace", False):
        trace_path.write_text(json.dumps({"checkpoint_sha256": report["checkpoint_sha256"],
                                         "n_action_steps": policy.config.n_action_steps,
                                         "episodes": traces}, ensure_ascii=False, indent=2) + "\n")
    print(f"成功率 once={report['success_once_rate']:.3f}, end={report['success_end_rate']:.3f}；报告={args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--control-mode", choices=CONTROL_CHOICES,
                        help="默认跟随checkpoint；指定ee/joint时校验一致性，不能切换模型动作维度")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1000, help="与训练示范 seed 分离")
    parser.add_argument("--policy-seed", type=int, default=42)
    parser.add_argument("--n-action-steps", type=int, help="只调整推理动作块长度，不改变模型权重；例如2与8对照")
    parser.add_argument("--reset-policy-seed-per-episode", action="store_true", help="每局从同一个随机流开始，便于配对比较")
    parser.add_argument("--save-trace", action="store_true", help="保存PickCube逐步物体距离/速度/抓取诊断；不输入模型")
    parser.add_argument("--env-id", choices=list(TASKS), help="可选任务校验；实际环境从 checkpoint 读取，不允许跨任务强制覆盖")
    parser.add_argument("--max-steps", type=int, help="默认按任务：Pick/Push200、Stack400、Peg500、Draw300；Draw不得超过300")
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
