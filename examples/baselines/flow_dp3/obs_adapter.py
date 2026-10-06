"""任务物理点云与本体接口；兼容历史 PickCube v1/v2。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import json
import math

import torch

from ee_relation_encoder import farthest_point_indices, gather_points
from scene_bounds import PICKCUBE_SCENE_CROP_MIN, PICKCUBE_SCENE_CROP_MAX, get_scene_crop_bounds
from mani_skill.utils.task_pointcloud import pointcloud_sensor_configs
from task_registry import get_task, task_sensors


STATE_FIELDS = {"qpos": [0, 9], "qvel": [9, 18], "tcp_base_pose_wxyz": [18, 25],
                "goal_base_pos": [25, 28]}


@dataclass(frozen=True)
class ObservationConfig:
    num_points: int = 512
    length_scale: float = 1.0
    crop_min: tuple = PICKCUBE_SCENE_CROP_MIN
    crop_max: tuple = PICKCUBE_SCENE_CROP_MAX
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

    def contract(self, *, sensor_configs=None, env_id="PickCube-v1"):
        task = get_task(env_id)
        return {"version": 2 if env_id == "PickCube-v1" else 3,
                "pointcloud": asdict(self), "state_dim": task.state_dim,
                "state_fields": task.state_fields, "frame": "robot_base",
                "distance_channels": "[relative_xyz_m, norm_m] / length_scale",
                "env_id": env_id, "robot_uids": task.robot,
                "control_mode": task.control_mode, "action_dim": task.action_dim,
                "sim_backend": "physx_cpu", "render_backend": "cpu",
                "obs_mode": "pointcloud",
                "sensor_configs": deepcopy(sensor_configs) if sensor_configs is not None else task_sensors(env_id),
                "reconfiguration_freq": 1}


def config_from_contract(contract):
    """Validate task identity, dimensions, camera geometry and historical schemas."""
    config = ObservationConfig(**contract["pointcloud"])
    version = contract.get("version")
    if version == 1:
        expected = config.contract(sensor_configs={"shader_pack": "default"})
        expected["version"] = 1
    elif version in (2, 3):
        env_id = contract.get("env_id", "PickCube-v1")
        task = get_task(env_id)
        sensors = contract.get("sensor_configs", {})
        names = {"shader_pack", "base_camera"}
        if task.robot == "panda_wristcam":
            names.add("hand_camera")
        if set(sensors) != names or sensors.get("shader_pack") != "default":
            raise ValueError("观测契约必须记录 default shader 和任务的全部点云相机")
        for name in names - {"shader_pack"}:
            camera = sensors[name]
            keys = {"pose", "width", "height", "fov", "near", "far"}
            if name == "hand_camera":
                keys.add("entity_uid")
                if camera.get("entity_uid") != "camera_link":
                    raise ValueError("腕部相机必须挂载 camera_link")
            if set(camera) != keys:
                raise ValueError("观测契约缺少完整相机位置、分辨率或视场参数")
            pose = camera["pose"]
            if len(pose) != 7 or not all(math.isfinite(x) for x in pose) or abs(sum(x*x for x in pose[3:]) - 1) > 1e-5:
                raise ValueError("相机 pose 必须为有限 xyz + 单位四元数 wxyz")
            if any(type(camera[k]) is not int or camera[k] <= 0 for k in ("width", "height")):
                raise ValueError("相机宽高必须为正整数")
            if not all(math.isfinite(camera[k]) for k in ("fov", "near", "far")) or not (
                0 < camera["fov"] < math.pi and 0 < camera["near"] < camera["far"]
            ):
                raise ValueError("相机视场和裁剪面参数无效")
        expected = config.contract(sensor_configs=sensors, env_id=env_id)
    else:
        raise ValueError(f"不支持观测契约版本 {version!r}")
    if json.dumps(expected, sort_keys=True) != json.dumps(contract, sort_keys=True):
        raise ValueError("数据/checkpoint 观测或环境契约与当前任务适配接口不同")
    return config


def pointcloud_features(obs, agent, config=None, *, env_id="PickCube-v1"):
    """共享点云采样；未传 config 时按任务选 crop，默认输出512点。

    显式 config（例如数据或 checkpoint 中的观测配置）优先，保持历史契约。
    状态适配按契约选择任务，不额外注入 RGB 或隐藏物体状态。
    """
    if config is None:
        crop_min, crop_max = get_scene_crop_bounds(env_id)
        config = ObservationConfig(crop_min=crop_min, crop_max=crop_max)
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
    tcp_pose = agent.tcp_pose if hasattr(agent, "tcp_pose") else agent.tcp.pose
    tcp_world = tcp_pose.p.detach().cpu()[0]
    tcp_base = world_to_base[:3, :3] @ tcp_world + world_to_base[:3, 3]
    relative = sampled - tcp_base[None, None]
    distances = relative.norm(dim=-1, keepdim=True)
    features = torch.cat((relative, distances), dim=-1) / config.length_scale
    if not torch.isfinite(features).all():
        raise RuntimeError("距离点云包含 NaN/Inf")
    return features, {"valid_points": int(valid.sum()), "cropped_points": int(keep.sum()),
                      "sampled_points": int(sampled.shape[1]),
                      "tcp_base_m": tcp_base.tolist(), "feature_shape": list(features.shape),
                      "distance_range_m": [float(distances.min()), float(distances.max())],
                      "distance_norm_error": float((features[..., 3] - features[..., :3].norm(dim=-1)).abs().max())}


def adapt_observation(obs, agent, config, *, contract=None):
    env_id = contract["env_id"] if contract is not None else "PickCube-v1"
    task = get_task(env_id)
    distance, diagnostics = pointcloud_features(obs, agent, config)
    qpos = torch.as_tensor(obs["agent"]["qpos"]).detach().cpu().reshape(1, -1)
    qvel = torch.as_tensor(obs["agent"]["qvel"]).detach().cpu().reshape(1, -1)
    tcp_pose = agent.tcp_pose if hasattr(agent, "tcp_pose") else agent.tcp.pose
    tcp_base = (agent.robot.pose.inv() * tcp_pose).raw_pose.detach().cpu().clone()
    tcp_base[:, 3:] *= torch.where(tcp_base[:, 3:4] < 0, -1.0, 1.0)
    parts = [qpos, qvel, tcp_base]
    # New visual tasks use proprioception only: no hidden object poses or goal labels.
    if "goal_base_pos" in task.state_fields:
        world_to_base = agent.robot.pose.inv().to_transformation_matrix().detach().cpu()[0]
        goal_world = torch.as_tensor(obs["extra"]["goal_pos"]).detach().cpu().reshape(1, 3)
        parts.append(goal_world @ world_to_base[:3, :3].T + world_to_base[:3, 3])
    state = torch.cat(parts, dim=-1).float()
    if qpos.shape != (1, task.joints) or qvel.shape != qpos.shape or state.shape != (1, task.state_dim) or not torch.isfinite(state).all():
        raise ValueError(f"{env_id} 本体状态异常：{tuple(state.shape)}")
    return {"pointcloud_distance": distance, "state": state}, diagnostics


def make_env(control_mode=None, visual=True, max_episode_steps=None, render_mode=None,
             *, contract=None, env_id=None):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    if contract is not None and env_id is not None and env_id != contract["env_id"]:
        raise ValueError("指定任务与数据/checkpoint 不匹配")
    env_id = contract["env_id"] if contract is not None else (env_id or "PickCube-v1")
    task = get_task(env_id)
    control_mode = control_mode or task.control_mode
    max_episode_steps = max_episode_steps or task.max_steps
    if env_id == "DrawTriangle-v1" and max_episode_steps > 300:
        raise ValueError("DrawTriangle 画迹缓冲最多300步，max_episode_steps 不得超过300")
    if contract is None:
        sensors = task_sensors(env_id)
    else:
        config_from_contract(contract)
        if control_mode != contract["control_mode"]:
            raise ValueError("环境控制模式与保存的观测契约不同")
        sensors = (pointcloud_sensor_configs(legacy_pickcube=True) if contract["version"] == 1
                   else deepcopy(contract["sensor_configs"]))
    env = gym.make(env_id, robot_uids=task.robot, num_envs=1,
                   obs_mode="pointcloud" if visual else "none",
                   control_mode=control_mode, sim_backend="physx_cpu",
                   render_backend="cpu" if visual else "none", render_mode=render_mode,
                   sensor_configs=sensors, reconfiguration_freq=1,
                   max_episode_steps=max_episode_steps)
    return env


def validate_env(env, contract):
    if env.unwrapped.spec.id != contract["env_id"] or env.unwrapped.agent.uid != contract["robot_uids"]:
        raise ValueError("实际任务/机器人与契约不匹配")
    if env.action_space.shape != (contract["action_dim"],):
        raise ValueError(f"实际动作空间 {env.action_space.shape} 与契约不匹配")
