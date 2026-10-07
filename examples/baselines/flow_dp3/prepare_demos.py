"""生成或回放 Panda 示范，转换真实闭环控制并保存紧凑的距离点云数据。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import time

import gymnasium as gym
import h5py
import numpy as np
import torch

from obs_adapter import ObservationConfig, adapt_observation, make_env, validate_env, action_bounds
from pickcube_metrics import pickcube_metrics
from task_registry import TASKS, get_task
from scene_bounds import get_scene_crop_bounds
from control_modes import CONTROL_CHOICES, DUAL_LAYOUT


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_raw(path, count, start_seed, max_attempts, max_steps, env_id="PickCube-v1"):
    from importlib import import_module
    task = get_task(env_id)
    solve = import_module("mani_skill.examples.motionplanning.panda.solutions." + task.solver).solve
    from mani_skill.utils.wrappers.record import RecordEpisode
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"原始数据已存在：{path}；请使用新的路径")
    record = RecordEpisode(make_env("pd_joint_pos", visual=False, max_episode_steps=max_steps, env_id=env_id),
                        output_dir=str(path.parent), trajectory_name=path.stem,
                        save_video=False, save_on_reset=False, record_reward=False,
                        source_type="motionplanning", source_desc=f"ManiSkill bundled {env_id} solver, CPU")
    env = ExpertStepLimit(record, max_steps)
    saved = 0
    try:
        for seed in range(start_seed, start_seed + max_attempts):
            try:
                result = solve(env, seed=seed, debug=False, vis=False)
            except EpisodeLimitError:
                result = -1
            success = not isinstance(result, int) and bool(result[-1]["success"].item())
            record.flush_trajectory(save=success)
            saved += int(success)
            print(f"生成 seed={seed}: success={success}, 已保存 {saved}/{count}", flush=True)
            if saved >= count:
                break
    finally:
        env.close()
    if saved < count:
        raise RuntimeError(f"{max_attempts} 次尝试仅生成 {saved}/{count} 条成功示范；已保留原始数据")
    return path


class EpisodeLimitError(RuntimeError):
    pass


class ExpertStepLimit(gym.Wrapper):
    """Experts may ignore truncation; never step beyond task buffers."""
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps, self.steps = max_steps, 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.steps >= self.max_steps:
            raise EpisodeLimitError("专家超过任务步数上限")
        result = self.env.step(action)
        self.steps += 1
        return result


class ObservationCollector(gym.Wrapper):
    def __init__(self, env, config, contract=None, max_steps=None):
        super().__init__(env)
        self.config = config
        self.contract = contract or config.contract()
        self.max_steps = max_steps or get_task(self.contract["env_id"]).max_steps

    def start(self):
        self.observations = []
        self.actions = []
        self.diagnostics = []
        self.success_once = False
        self.task_metrics = []
        self._capture(self.unwrapped.get_obs())

    def _capture(self, obs, info=None):
        adapted, detail = adapt_observation(obs, self.unwrapped.agent, self.config, contract=self.contract)
        self.observations.append({key: value[0].numpy().copy() for key, value in adapted.items()})
        self.diagnostics.append(detail)
        if self.contract["env_id"] == "PickCube-v1":
            self.task_metrics.append(pickcube_metrics(self, info))

    def step(self, action):
        action = torch.as_tensor(action).detach().cpu().numpy().reshape(-1)
        if len(self.actions) >= self.max_steps:
            raise EpisodeLimitError("转换后动作超过任务步数上限")
        low, high = action_bounds(self.contract)
        if (action.shape != (self.contract["action_dim"],) or not np.isfinite(action).all() or
                np.any(action < np.asarray(low) - 1e-4) or np.any(action > np.asarray(high) + 1e-4)):
            raise ValueError(f'{self.contract["control_mode"]} 动作维度或范围错误：{action}')
        result = self.env.step(action)
        self.actions.append(action.astype(np.float32).copy())
        self._capture(result[0], result[-1])
        self.success_once |= bool(torch.as_tensor(result[-1]["success"]).item())
        return result


def pose_hold_action(collector, target_pose, gripper):
    """Keep one fixed base-frame target; compute a fresh correction at every step."""
    from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix, matrix_to_euler_angles
    controller = collector.unwrapped.agent.controller.controllers["arm"]
    current = controller.ee_pose_at_base
    delta_pos = target_pose.p - current.p
    pos = 2 * (delta_pos - delta_pos.new_tensor(controller.config.pos_lower)) / (
        controller.config.pos_upper - controller.config.pos_lower) - 1
    rotation = quaternion_to_matrix(target_pose.q) @ quaternion_to_matrix(current.q).transpose(-1, -2)
    # Local PDEEPoseController scales normalized rotation by rot_lower (negative).
    rot = matrix_to_euler_angles(rotation, "XYZ") / controller.config.rot_lower
    rot = rot / rot.norm(dim=-1, keepdim=True).clamp(min=1)
    return torch.cat((pos.clamp(-1, 1), rot, pos.new_tensor([[gripper]])), dim=-1)[0]


def replay_demo(target, original, traj, episode, contract, hold_steps):
    """Replay one controller from the source initial state, including a real hold."""
    from mani_skill.trajectory import utils as trajectory_utils
    from mani_skill.trajectory.utils.actions import conversion
    mode = episode["control_mode"]
    if mode not in ("pd_joint_pos", contract["control_mode"]):
        raise ValueError(f"暂不支持从 {mode} 转换为 {contract['control_mode']}；"
                         "双分支录制需要原始 pd_joint_pos 轨迹，不能补造动作标签")
    reset = dict(episode["reset_kwargs"])
    if isinstance(reset.get("seed"), list):
        reset["seed"] = reset["seed"][0]
    target.reset(**reset)
    original.reset(**reset)
    first_state = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])[0]
    target.unwrapped.set_state_dict(first_state)
    original.unwrapped.set_state_dict(first_state)
    target.start()
    try:
        if mode == contract["control_mode"]:
            for action in traj["actions"][:]:
                info = target.step(action)[-1]
        else:
            info = conversion.from_pd_joint_pos(contract["control_mode"],
                                                traj["actions"][:], original, target)
        motion_steps = len(target.actions)
        pre_hold_success = bool(torch.as_tensor(info["success"]).item())
        if hold_steps and pre_hold_success:
            last_action = target.actions[-1].copy()
            controller = target.unwrapped.agent.controller.controllers["arm"]
            from mani_skill.utils.structs.pose import Pose
            fixed_pose = (Pose.create(controller._target_pose.raw_pose.clone())
                          if contract["control_mode"] == "pd_ee_delta_pose" else None)
            for _ in range(hold_steps):
                action = (last_action if fixed_pose is None else
                          pose_hold_action(target, fixed_pose, float(last_action[-1])))
                info = target.step(action)[-1]
    except EpisodeLimitError as exc:
        return None, {"reason": str(exc)}
    success_end = bool(torch.as_tensor(info["success"]).item())
    hold_metrics = target.task_metrics[motion_steps + 1:] if hold_steps else []
    hold_stable = (len(hold_metrics) == hold_steps and all(m["success"] for m in hold_metrics)) if hold_steps else True
    if not success_end or not hold_stable or not target.actions:
        return None, {"success_end": success_end, "steps": len(target.actions), "hold_stable": hold_stable}
    detail = {"source_episode": episode["episode_id"], "seed": reset.get("seed"),
              "control_mode": contract["control_mode"],
              "steps": len(target.actions), "success_once": target.success_once,
              "success_end": success_end, "motion_steps": motion_steps, "hold_steps": hold_steps,
              "pre_hold_success": pre_hold_success, "hold_stable": hold_stable,
              "hold_goal_error_max_m": max((m["goal_error_m"] for m in hold_metrics), default=None),
              "min_cropped_points": min(d["cropped_points"] for d in target.diagnostics),
              "max_distance_norm_error": max(d["distance_norm_error"] for d in target.diagnostics)}
    return detail, None


def save_demo(group, target, detail):
    for key in ("pointcloud_distance", "state"):
        group.create_dataset(key, data=np.stack([obs[key] for obs in target.observations]), compression="lzf")
    group.create_dataset("action", data=np.stack(target.actions))
    if target.task_metrics:
        metrics_group = group.create_group("task_metrics")
        for key in target.task_metrics[0]:
            metrics_group.create_dataset(key, data=[row[key] for row in target.task_metrics])
    group.attrs["metadata"] = json.dumps(detail)


def recording_modes(env_id, requested):
    task = get_task(env_id)
    if requested in (None, "both") and env_id == "PickCube-v1":
        return {"ee": ("pd_ee_delta_pose", 5), "joint": ("pd_joint_pos", 6)}
    if requested == "both":
        raise ValueError("双控制分支目前仅支持 PickCube；其他任务使用默认控制器或 --control-mode ee")
    mode = {"ee": task.control_mode, "joint": "pd_joint_pos"}.get(requested, requested or task.control_mode)
    if mode != task.control_mode and (env_id != "PickCube-v1" or mode != "pd_joint_pos"):
        raise ValueError(f"{env_id} 不支持选择 {mode}")
    version = (6 if mode == "pd_joint_pos" else 5) if env_id == "PickCube-v1" else 3
    return {"joint" if mode == "pd_joint_pos" else "ee": (mode, version)}


def prepare(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"输出已存在：{output}；请选择新文件名")
    temporary = output.with_suffix(".partial.h5")
    if temporary.exists():
        raise FileExistsError(f"上次未完成输出：{temporary}；检查后使用新的输出路径")
    save_video = getattr(args, "save_video", True)
    if getattr(args, "video_dir", None) and not save_video:
        raise ValueError("--video-dir 需要开启录像")
    if getattr(args, "video_fps", None) is not None and args.video_fps < 1:
        raise ValueError("video-fps 必须为正整数")
    video_dir = (getattr(args, "video_dir", None) or output.parent / (output.stem + "-videos-full")).resolve()
    if save_video and video_dir.exists() and any(video_dir.iterdir()):
        raise FileExistsError(f"视频目录非空：{video_dir}；请选择新的输出路径")
    env_id = args.env_id or "PickCube-v1"
    if args.generate:
        args.max_steps = args.max_steps or get_task(env_id).max_steps
        source = generate_raw(output.parent / (output.stem + ".raw.h5"), args.generate,
                              args.start_seed, args.max_attempts, args.max_steps, env_id)
    else:
        source = args.source.resolve()
    metadata = json.loads(source.with_suffix(".json").read_text())
    env_id = metadata["env_info"]["env_id"]
    if args.env_id is not None and args.env_id != env_id:
        raise ValueError(f"指定任务 {args.env_id} 与原始示范任务 {env_id} 不匹配")
    task = get_task(env_id)
    modes = recording_modes(env_id, getattr(args, "control_mode", None))
    dual = len(modes) == 2
    hold_steps = getattr(args, "hold_steps", None)
    hold_steps = (40 if env_id == "PickCube-v1" else 0) if hold_steps is None else hold_steps
    if hold_steps and env_id != "PickCube-v1":
        raise ValueError("保持示范只用于 PickCube，其他任务请使用 --hold-steps 0")
    robot = metadata["env_info"]["env_kwargs"].get("robot_uids", task.robot)
    if robot != task.robot:
        raise ValueError(f"{env_id} 需要 {task.robot}，原数据 robot_uids={robot}")
    args.max_steps = args.max_steps or task.max_steps
    low, high = get_scene_crop_bounds(env_id)
    config = ObservationConfig(num_points=args.num_points, length_scale=args.length_scale,
                               crop_min=tuple(args.crop_min or low), crop_max=tuple(args.crop_max or high))
    contracts = {key: config.contract(env_id=env_id, contract_version=version)
                 for key, (_, version) in modes.items()}
    attempts, rejected = 0, []
    saved = {key: [] for key in modes}
    targets, records = {}, {}
    started = time.monotonic()
    with ExitStack() as stack:
        original = make_env("pd_joint_pos", visual=False, max_episode_steps=args.max_steps, env_id=env_id)
        stack.callback(original.close)
        for key, contract in contracts.items():
            env = make_env(max_episode_steps=args.max_steps, contract=contract,
                           render_mode="rgb_array" if save_video else None)
            if save_video:
                from mani_skill.utils.wrappers.record import RecordEpisode
                try:
                    env = RecordEpisode(env, output_dir=str(video_dir / key), save_trajectory=False,
                                        save_video=True, save_on_reset=False, info_on_video=False,
                                        video_fps=getattr(args, "video_fps", None) or env.unwrapped.control_freq)
                except Exception:
                    env.close()
                    raise
                records[key] = env
            target = ObservationCollector(env, config, contract, args.max_steps)
            stack.callback(target.close)
            validate_env(target, contract)
            targets[key] = target
        with h5py.File(source, "r") as raw, h5py.File(temporary, "w") as prepared:
            if dual:
                prepared.attrs["layout"] = DUAL_LAYOUT
                roots = {key: prepared.create_group(key) for key in modes}
            else:
                roots = {next(iter(modes)): prepared}
            for key, root in roots.items():
                root.attrs["contract"] = json.dumps(contracts[key], sort_keys=True)
            for episode in metadata["episodes"]:
                count = len(next(iter(saved.values())))
                if args.count and count >= args.count:
                    break
                attempts += 1
                identifier = episode["episode_id"]
                traj = raw[f"traj_{identifier}"]
                if not len(traj["actions"]):
                    rejected.append({"source_episode": identifier, "reason": "empty_actions"})
                    continue
                details, failures = {}, {}
                for key, target in targets.items():
                    detail, failure = replay_demo(target, original, traj, episode, contracts[key], hold_steps)
                    if failure is not None:
                        failures[key] = failure
                    else:
                        details[key] = detail
                if failures:
                    rejected.append({"source_episode": identifier, "branches": failures})
                    for record in records.values():
                        record.flush_video(save=False)
                    print(f"回放 {identifier}: 拒绝，{failures}", flush=True)
                    continue
                for key, target in targets.items():
                    detail = details[key]
                    if save_video:
                        name = f"episode_{count:05d}_seed_{detail['seed']}"
                        record = records[key]
                        frames = len(record.render_images)
                        if frames != len(target.actions) + 1:
                            raise RuntimeError(f"{key} 视频帧数与 T+1 观测不一致：{frames}")
                        record.flush_video(name=name)
                        detail.update(video_path=os.path.relpath(video_dir / key / (name + ".mp4"), output.parent),
                                      video_fps=record.video_fps, video_frames=frames)
                    save_demo(roots[key].create_group(f"episode_{count:05d}"), target, detail)
                    saved[key].append(detail)
                print(f"回放 {identifier}: 成功，{ {key: d['steps'] for key, d in details.items()} } steps，已保存 {count + 1}", flush=True)
            count = len(next(iter(saved.values())))
            common = {"source": str(source), "source_sha256": file_hash(source),
                      "source_json_sha256": file_hash(source.with_suffix(".json")),
                      "source_type": metadata.get("source_type", "unknown"), "hold_steps": hold_steps,
                      "attempted": attempts, "saved": count, "rejected": rejected,
                      "save_video": save_video, "elapsed_seconds": time.monotonic() - started}
            branches = {}
            for key, root in roots.items():
                branch = {**common, "contract": contracts[key], "episodes": saved[key]}
                root.attrs["manifest"] = json.dumps(branch)
                branches[key] = branch
            manifest = ({**common, "layout": DUAL_LAYOUT, "default_control": "ee", "branches": branches,
                         "paired_source_episodes": [d["source_episode"] for d in saved["ee"]]}
                        if dual else next(iter(branches.values())))
            prepared.attrs["manifest"] = json.dumps(manifest)
        if not count:
            raise RuntimeError("没有回放成功的示范；未生成可训练输出")
        temporary.rename(output)
        output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(f"输出 {output}；成功 {count}/{attempts}，分支={list(modes)}", flush=True)
        if args.count and count < args.count:
            raise RuntimeError(f"仅得到 {count}/{args.count} 条成功示范；已保存，需补充原始数据")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="ManiSkill 原始 .h5；同目录必须有同名 .json")
    source.add_argument("--generate", type=int, help="离线使用仓库内置运动规划器生成成功原始示范")
    parser.add_argument("--env-id", choices=list(TASKS), help="生成时默认 PickCube-v1；转换时从原始 JSON 读取，显式指定则严格核对")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5, help="保存多少条成功回放；0表示所有原始示范")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--max-steps", type=int, help="任务步数上限：Pick/Push200、Stack400、Peg500、Draw300；Draw不得超过300")
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--control-mode", choices=("both",) + CONTROL_CHOICES,
                        help="PickCube默认both，保存同源ee/joint双分支；也可单独录制ee或joint，其他任务默认原控制器")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True,
                        help="默认保存与最终训练轨迹对应的场景MP4；--no-save-video关闭")
    parser.add_argument("--video-dir", type=Path, help="默认 <数据集名称>-videos-full/{ee,joint}")
    parser.add_argument("--video-fps", type=int, help="默认环境控制频率；仅影响视频播放速度")
    parser.add_argument("--hold-steps", type=int, help="PickCube到点后真实保持固定目标，默认40步；其他任务默认0")
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--crop-min", type=float, nargs=3, default=None,
                        help="基座系 xyz 下界（米），默认按任务预设操作区")
    parser.add_argument("--crop-max", type=float, nargs=3, default=None,
                        help="基座系 xyz 上界（米），默认按任务预设操作区")
    args = parser.parse_args()
    if args.hold_steps is not None and args.hold_steps < 0:
        parser.error("hold-steps >= 0")
    if args.count < 0 or (args.max_steps is not None and args.max_steps < 16) or args.max_attempts < 1 or (args.generate is not None and args.generate < 1):
        parser.error("count >= 0，max_steps >= 16，generate/max_attempts >= 1")
    if args.generate and args.count > args.generate:
        parser.error("count 不能超过 generate；增加 generate 可给回放失败预留余量")
    prepare(args)


if __name__ == "__main__":
    main()
