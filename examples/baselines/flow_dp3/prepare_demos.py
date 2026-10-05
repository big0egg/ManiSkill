"""生成或回放 Panda 示范，转换真实闭环控制并保存紧凑的距离点云数据。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import gymnasium as gym
import h5py
import numpy as np
import torch

from obs_adapter import ObservationConfig, adapt_observation, make_env


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_raw(path, count, start_seed, max_attempts, max_steps):
    from mani_skill.examples.motionplanning.panda.solutions.pick_cube import solve
    from mani_skill.utils.wrappers.record import RecordEpisode
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"原始数据已存在：{path}；请使用新的路径")
    env = RecordEpisode(make_env("pd_joint_pos", visual=False, max_episode_steps=max_steps),
                        output_dir=str(path.parent), trajectory_name=path.stem,
                        save_video=False, save_on_reset=False, record_reward=False,
                        source_type="motionplanning", source_desc="ManiSkill bundled PickCube solver, CPU")
    saved = 0
    try:
        for seed in range(start_seed, start_seed + max_attempts):
            result = solve(env, seed=seed, debug=False, vis=False)
            success = not isinstance(result, int) and bool(result[-1]["success"].item())
            env.flush_trajectory(save=success)
            saved += int(success)
            print(f"生成 seed={seed}: success={success}, 已保存 {saved}/{count}", flush=True)
            if saved >= count:
                break
    finally:
        env.close()
    if saved < count:
        raise RuntimeError(f"{max_attempts} 次尝试仅生成 {saved}/{count} 条成功示范；已保留原始数据")
    return path


class ObservationCollector(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config

    def start(self):
        self.observations = []
        self.actions = []
        self.diagnostics = []
        self.success_once = False
        self._capture(self.unwrapped.get_obs())

    def _capture(self, obs):
        adapted, detail = adapt_observation(obs, self.unwrapped.agent, self.config)
        self.observations.append({key: value[0].numpy().copy() for key, value in adapted.items()})
        self.diagnostics.append(detail)

    def step(self, action):
        action = torch.as_tensor(action).detach().cpu().numpy().reshape(-1)
        if action.shape != (4,) or not np.isfinite(action).all() or np.abs(action).max() > 1.0001:
            raise ValueError(f"pd_ee_delta_pos 动作应为 [-1,1] 内的4维值：{action}")
        result = self.env.step(action)
        self.actions.append(action.astype(np.float32).copy())
        self._capture(result[0])
        self.success_once |= bool(torch.as_tensor(result[-1]["success"]).item())
        return result


def prepare(args):
    from mani_skill.trajectory import utils as trajectory_utils
    from mani_skill.trajectory.utils.actions import conversion

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"输出已存在：{output}；请选择新文件名")
    if args.generate:
        source = generate_raw(output.parent / (output.stem + ".raw.h5"), args.generate,
                              args.start_seed, args.max_attempts, args.max_steps)
    else:
        source = args.source.resolve()
    metadata = json.loads(source.with_suffix(".json").read_text())
    if metadata["env_info"]["env_id"] != "PickCube-v1":
        raise ValueError("第一阶段仅支持 PickCube-v1 / Panda")
    robot = metadata["env_info"]["env_kwargs"].get("robot_uids", "panda")
    if robot != "panda":
        raise ValueError(f"仅支持 Panda，原数据 robot_uids={robot}")
    config = ObservationConfig(num_points=args.num_points, length_scale=args.length_scale,
                               crop_min=tuple(args.crop_min), crop_max=tuple(args.crop_max))
    target = ObservationCollector(make_env(max_episode_steps=args.max_steps), config)
    original = make_env("pd_joint_pos", visual=False, max_episode_steps=args.max_steps)
    attempts, saved, rejected = 0, [], []
    temporary = output.with_suffix(".partial.h5")
    if temporary.exists():
        raise FileExistsError(f"上次未完成输出：{temporary}；检查后使用新的输出路径")
    started = time.monotonic()
    try:
        with h5py.File(source, "r") as raw, h5py.File(temporary, "w") as prepared:
            prepared.attrs["contract"] = json.dumps(config.contract(), sort_keys=True)
            for episode in metadata["episodes"]:
                if args.count and len(saved) >= args.count:
                    break
                attempts += 1
                identifier = episode["episode_id"]
                mode = episode["control_mode"]
                if mode not in ("pd_joint_pos", "pd_ee_delta_pos"):
                    raise ValueError(f"暂不支持从 {mode} 转换动作")
                traj = raw[f"traj_{identifier}"]
                if not len(traj["actions"]):
                    rejected.append({"source_episode": identifier, "reason": "empty_actions"})
                    continue
                reset = dict(episode["reset_kwargs"])
                if isinstance(reset.get("seed"), list):
                    reset["seed"] = reset["seed"][0]
                target.reset(**reset)
                original.reset(**reset)
                # 仅恢复初始状态。随后真实执行转换动作，不能逐帧强设状态冒充闭环。
                first_state = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])[0]
                target.unwrapped.set_state_dict(first_state)
                original.unwrapped.set_state_dict(first_state)
                target.start()
                if mode == "pd_joint_pos":
                    info = conversion.from_pd_joint_pos("pd_ee_delta_pos",
                                                        traj["actions"][:], original, target)
                else:
                    for action in traj["actions"][:]:
                        info = target.step(action)[-1]
                success_end = bool(torch.as_tensor(info["success"]).item())
                if not success_end or not target.actions or len(target.actions) > args.max_steps:
                    rejected.append({"source_episode": identifier, "success_end": success_end,
                                     "steps": len(target.actions)})
                    print(f"回放 {identifier}: 拒绝，success_end={success_end}, steps={len(target.actions)}", flush=True)
                    continue
                group = prepared.create_group(f"episode_{len(saved):05d}")
                for key in ("pointcloud_distance", "state"):
                    group.create_dataset(key, data=np.stack([obs[key] for obs in target.observations]),
                                         compression="lzf")
                group.create_dataset("action", data=np.stack(target.actions))
                detail = {"source_episode": identifier, "seed": reset.get("seed"),
                          "steps": len(target.actions), "success_once": target.success_once,
                          "success_end": success_end,
                          "min_cropped_points": min(d["cropped_points"] for d in target.diagnostics),
                          "max_distance_norm_error": max(d["distance_norm_error"] for d in target.diagnostics)}
                group.attrs["metadata"] = json.dumps(detail)
                saved.append(detail)
                print(f"回放 {identifier}: 成功，{len(target.actions)} steps，已保存 {len(saved)}", flush=True)
            manifest = {"contract": config.contract(), "source": str(source),
                        "source_sha256": file_hash(source),
                        "source_json_sha256": file_hash(source.with_suffix(".json")),
                        "source_type": metadata.get("source_type", "unknown"),
                        "attempted": attempts, "saved": len(saved), "episodes": saved,
                        "rejected": rejected, "elapsed_seconds": time.monotonic() - started}
            prepared.attrs["manifest"] = json.dumps(manifest)
        if not saved:
            raise RuntimeError("没有回放成功的示范；未生成可训练输出")
        temporary.rename(output)
        output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(f"输出 {output}；成功 {len(saved)}/{attempts}", flush=True)
        if args.count and len(saved) < args.count:
            raise RuntimeError(f"仅得到 {len(saved)}/{args.count} 条成功示范；已保存，需补充原始数据")
    finally:
        target.close()
        original.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="ManiSkill 原始 .h5；同目录必须有同名 .json")
    source.add_argument("--generate", type=int, help="离线使用仓库内置运动规划器生成成功原始示范")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5, help="保存多少条成功回放；0表示所有原始示范")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--crop-min", type=float, nargs=3, default=list(ObservationConfig.crop_min),
                        help="基座系 xyz 下界（米），默认覆盖完整 PickCube 场景")
    parser.add_argument("--crop-max", type=float, nargs=3, default=list(ObservationConfig.crop_max),
                        help="基座系 xyz 上界（米），默认覆盖完整 PickCube 场景")
    args = parser.parse_args()
    if args.count < 0 or args.max_steps < 16 or args.max_attempts < 1 or (args.generate is not None and args.generate < 1):
        parser.error("count >= 0，max_steps >= 16，generate/max_attempts >= 1")
    if args.generate and args.count > args.generate:
        parser.error("count 不能超过 generate；增加 generate 可给回放失败预留余量")
    prepare(args)


if __name__ == "__main__":
    main()
