"""PickCube 单环境的统一物理点云与本体状态契约，供采集、探测和评估使用。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from ee_relation_encoder import farthest_point_indices, gather_points


STATE_FIELDS = {"qpos": [0, 9], "qvel": [9, 18], "tcp_base_pose_wxyz": [18, 25],
                "goal_base_pos": [25, 28]}


@dataclass(frozen=True)
class ObservationConfig:
    num_points: int = 512
    length_scale: float = 1.0
    crop_min: tuple = (0.05, -0.8, -0.1)
    crop_max: tuple = (1.2, 0.8, 1.0)
    pre_sample_points: int = 4096
    sampling_seed: int = 42

    def __post_init__(self):
        if self.num_points < 128 or self.pre_sample_points < self.num_points:
            raise ValueError("num_points 至少128，pre_sample_points 不小于 num_points")
        if not math.isfinite(self.length_scale) or self.length_scale <= 0:
            raise ValueError("length_scale 必须为有限正数")
        if len(self.crop_min) != 3 or len(self.crop_max) != 3 or any(
            not math.isfinite(a) or not math.isfinite(b) or a >= b
            for a, b in zip(self.crop_min, self.crop_max)
        ):
            raise ValueError("裁剪边界必须是三维有限值，min < max")

    def contract(self):
        return {"version": 1, "pointcloud": asdict(self), "state_dim": 28,
                "state_fields": STATE_FIELDS, "frame": "robot_base",
                "distance_channels": "[relative_xyz_m, norm_m] / length_scale",
                "env_id": "PickCube-v1", "robot_uids": "panda",
                "control_mode": "pd_ee_delta_pos", "action_dim": 4,
                "sim_backend": "physx_cpu", "render_backend": "cpu",
                "obs_mode": "pointcloud", "sensor_configs": {"shader_pack": "default"},
                "reconfiguration_freq": 1}


def pointcloud_features(obs, agent, config):
    xyzw = torch.as_tensor(obs["pointcloud"]["xyzw"]).detach().to("cpu", torch.float32)
    if xyzw.ndim != 3 or xyzw.shape[0] != 1 or xyzw.shape[-1] != 4:
        raise ValueError(f"预期单环境 [1,N,4] xyzw，实际 {tuple(xyzw.shape)}")
    valid = (xyzw[0, :, 3] > 0.5) & torch.isfinite(xyzw[0]).all(dim=-1)
    world = xyzw[0, valid, :3]
    world_to_base = agent.robot.pose.inv().to_transformation_matrix().detach().cpu()[0]
    points = world @ world_to_base[:3, :3].T + world_to_base[:3, 3]
    keep = ((points >= torch.tensor(config.crop_min)) &
            (points <= torch.tensor(config.crop_max))).all(dim=-1)
    points = points[keep]
    if len(points) < config.num_points:
        raise ValueError(f"裁剪后仅 {len(points)} 点，少于 {config.num_points}；检查相机与裁剪范围")
    pre_sample_points = getattr(config, "pre_sample_points", 4096)
    if len(points) > pre_sample_points:
        generator = torch.Generator().manual_seed(getattr(config, "sampling_seed", 42))
        points = points[torch.randperm(len(points), generator=generator)[:pre_sample_points]]
    sampled = gather_points(points[None], farthest_point_indices(points[None], config.num_points))
    tcp_world = agent.tcp_pose.p.detach().cpu()[0]
    tcp_base = world_to_base[:3, :3] @ tcp_world + world_to_base[:3, 3]
    relative = sampled - tcp_base[None, None]
    distances = relative.norm(dim=-1, keepdim=True)
    features = torch.cat((relative, distances), dim=-1) / config.length_scale
    if not torch.isfinite(features).all():
        raise RuntimeError("距离点云包含 NaN/Inf")
    return features, {"valid_points": int(valid.sum()), "cropped_points": int(keep.sum()),
                      "tcp_base_m": tcp_base.tolist(), "feature_shape": list(features.shape),
                      "distance_range_m": [float(distances.min()), float(distances.max())],
                      "distance_norm_error": float((features[..., 3] - features[..., :3].norm(dim=-1)).abs().max())}


def adapt_observation(obs, agent, config):
    distance, diagnostics = pointcloud_features(obs, agent, config)
    # 仅使用关节、末端与任务目标，不使用物体真值、is_grasped 或 success。
    qpos = torch.as_tensor(obs["agent"]["qpos"]).detach().cpu().reshape(1, -1)
    qvel = torch.as_tensor(obs["agent"]["qvel"]).detach().cpu().reshape(1, -1)
    tcp_base = (agent.robot.pose.inv() * agent.tcp_pose).raw_pose.detach().cpu().clone()
    tcp_base[:, 3:] *= torch.where(tcp_base[:, 3:4] < 0, -1.0, 1.0)
    world_to_base = agent.robot.pose.inv().to_transformation_matrix().detach().cpu()[0]
    goal_world = torch.as_tensor(obs["extra"]["goal_pos"]).detach().cpu().reshape(1, 3)
    goal_base = goal_world @ world_to_base[:3, :3].T + world_to_base[:3, 3]
    state = torch.cat((qpos, qvel, tcp_base, goal_base), dim=-1).float()
    if state.shape != (1, 28) or not torch.isfinite(state).all():
        raise ValueError(f"Panda 本体/目标状态异常：{tuple(state.shape)}")
    return {"pointcloud_distance": distance, "state": state}, diagnostics


def make_env(control_mode="pd_ee_delta_pos", visual=True, max_episode_steps=200):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    return gym.make("PickCube-v1", robot_uids="panda", num_envs=1,
                    obs_mode="pointcloud" if visual else "none",
                    control_mode=control_mode, sim_backend="physx_cpu",
                    render_backend="cpu" if visual else "none", render_mode=None,
                    sensor_configs={"shader_pack": "default"},
                    reconfiguration_freq=1,
                    max_episode_steps=max_episode_steps)
