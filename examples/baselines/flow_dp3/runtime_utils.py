"""独立配置、设备与 checkpoint 工具。"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
import yaml

from policy import FlowDP3, PolicyConfig
from pointcloud_inputs import INPUT_VERSION, STORAGE_CONTRACT, ABLATIONS, ablation_group


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "pickcube.yaml"
CHECKPOINT_VERSION = 2
ABLATION_CONFIG_DIR = Path(__file__).parent / "config" / "pointcloud_ablation"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path, ablation=None):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or set(config) != {"policy", "training"}:
        raise ValueError("配置必须包含 policy、training 两个字段")
    if ablation is not None:
        if ablation not in ABLATIONS:
            raise ValueError(f"Unknown ablation group: {ablation}")
        preset = yaml.safe_load((ABLATION_CONFIG_DIR / f"{ablation.lower()}.yaml").read_text())["policy"]
        for key, value in preset.items():
            if key in config["policy"] and config["policy"][key] != value:
                raise ValueError(f"--ablation {ablation} 与配置 {key} 冲突")
        config["policy"].update(preset)
    policy_config = PolicyConfig(**config["policy"])
    fields = {key for key in ("state_dim", "action_dim") if key in config["policy"]}
    config["policy"] = asdict(policy_config)
    for key in {"state_dim", "action_dim"} - fields:
        del config["policy"][key]
    required = {"seed", "batch_size", "steps", "lr", "betas", "weight_decay", "warmup_steps",
                "val_ratio", "val_every", "val_batches", "checkpoint_every", "grad_clip"}
    if set(config["training"]) - {"required_contract_version"} != required:
        raise ValueError(f"training 配置键不匹配：{set(config['training']) ^ required}")
    version = config["training"].get("required_contract_version")
    if version is not None and (type(version) is not int or version < 1):
        raise ValueError("required_contract_version 必须为正整数")
    return config


def pointcloud_contract(config, observation_contract):
    """Serialize geometry and attribute semantics; compare before loading weights."""
    from obs_adapter import config_from_contract
    config_from_contract(observation_contract)
    if config.length_scale != observation_contract["pointcloud"]["length_scale"]:
        raise ValueError("数据与策略 length_scale 不一致")
    if config.pointcloud_num_centers[0] > observation_contract["pointcloud"]["num_points"]:
        raise ValueError("SA1 centers exceed the number of real observed points")
    start = observation_contract["state_fields"]["tcp_base_pose_wxyz"][0]
    return {"version": INPUT_VERSION, "storage": STORAGE_CONTRACT,
            "representation": config.pointcloud_representation,
            "encoder_type": config.pointcloud_encoder_type, "output_dim": 64,
            "frame": observation_contract["frame"], "axes": "robot_base_fixed",
            "tcp_reference": "current_observed_tcp", "tcp_position_slice": [start, start + 3],
            "length_scale": config.length_scale,
            "sampling": observation_contract["pointcloud"],
            "geometry_backend": "torch", "geometry_version": "canonical_fps_ball_query_v1",
            "num_centers": list(config.pointcloud_num_centers),
            "num_neighbors": list(config.pointcloud_num_neighbors),
            "radius_m": [config.radius1_m, config.radius2_m],
            "local_relation_mode": config.local_relation_mode, "readout_mode": config.readout_mode}


def validate_checkpoint(checkpoint):
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("旧 checkpoint 已停用；需要重新训练四组实验格式 v2")
    fields = ("pointcloud_representation", "pointcloud_encoder_type", "pointcloud_input_version",
              "pointcloud_num_centers", "pointcloud_num_neighbors", "local_relation_mode", "readout_mode")
    policy_settings = checkpoint.get("config", {}).get("policy")
    if not isinstance(policy_settings, dict) or any(field not in policy_settings for field in fields):
        raise ValueError("checkpoint 缺少显式点云配置，拒绝猜测输入语义")
    config = PolicyConfig(**policy_settings)
    expected = pointcloud_contract(config, checkpoint["contract"])
    if canonical(checkpoint.get("pointcloud_input")) != canonical(expected):
        raise ValueError("checkpoint 点云表示、编码器或几何设置与输入契约不匹配")
    if any(getattr(config, key) != checkpoint["contract"][key] for key in ("state_dim", "action_dim")):
        raise ValueError("checkpoint 模型维度与任务契约不匹配")
    if checkpoint.get("ablation") != ablation_group(config):
        raise ValueError("checkpoint 组别与点云配置不匹配")
    return config


def select_device(name):
    if name not in ("cpu", "cuda:0"):
        raise ValueError("当前入口支持 cpu 或 cuda:0")
    if name == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("当前进程无法访问显卡；请检查 ROCm 版 PyTorch、驱动和设备权限，或显式 --device cpu 做烟雾测试")
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
    policy_config = validate_checkpoint(checkpoint)
    policy = FlowDP3(policy_config)
    policy.load_state_dict(checkpoint["ema" if use_ema else "model"], strict=True)
    policy.to(device).eval()
    return policy, checkpoint
