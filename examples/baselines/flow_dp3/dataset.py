"""按 episode 划分；历史契约保留旧窗口，EE保持契约覆盖末尾真实状态。"""
from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from obs_adapter import config_from_contract
from control_modes import select_dataset_group
from pointcloud_inputs import STORAGE_CONTRACT


class DemoDataset(Dataset):
    def __init__(self, path, horizon=16, n_obs_steps=2, n_action_steps=8,
                 split="train", val_ratio=0.2, seed=42, control_mode=None):
        self.path = str(path)
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.file = None
        self.index = []
        if split not in ("train", "val") or not 0 < val_ratio < 1:
            raise ValueError("需要 train/val 划分及 0 < val_ratio < 1")
        with h5py.File(self.path, "r") as stream:
            root, self.contract, self.manifest = select_dataset_group(stream, control_mode)
            if json.loads(root.attrs.get("pointcloud_storage", "null")) != STORAGE_CONTRACT:
                raise ValueError("需要新录制的独立 XYZ 存储版本；旧数据已停用")
            self.group_name = root.name
            config_from_contract(self.contract)
            episodes = sorted(root.keys())
            if len(episodes) < 2:
                raise ValueError("至少需要两条成功示范，才能按 episode 留出验证集")
            n_val = min(max(1, round(len(episodes) * val_ratio)), len(episodes) - 1)
            val_indices = set(np.random.default_rng(seed).choice(len(episodes), n_val, replace=False).tolist())
            self.episodes = [name for i, name in enumerate(episodes) if (i in val_indices) == (split == "val")]
            for name in self.episodes:
                group = root[name]
                length = len(group["action"])
                if "pointcloud_xyz" not in group:
                    raise ValueError("旧数据缺少独立 pointcloud_xyz，请重新录制 EE 数据")
                if group["state"].shape != (length + 1, self.contract["state_dim"]) or group["pointcloud_xyz"].shape != (
                    length + 1, self.contract["pointcloud"]["num_points"], 3
                ) or group["action"].shape != (length, self.contract["action_dim"]):
                    raise ValueError(f"{name} 的 T+1 观测 / T 动作形状异常")
                if not json.loads(group.attrs["metadata"])["success_end"]:
                    raise ValueError("训练数据中包含失败回放")
                if "sequence_sampling" in self.contract:
                    # 当前时刻从0到T；terminal观测是真实保存的，不再丢掉尾部状态。
                    starts = range(-(n_obs_steps - 1), length - n_obs_steps + 2)
                else:
                    # 历史数据/checkpoint 保持原 SequenceSampler 行为。
                    starts = range(-(n_obs_steps - 1), length - horizon + n_action_steps)
                self.index.extend((name, start, length) for start in starts)
        if not self.index:
            raise ValueError(f"{split} 没有足够长的动作序列")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        name, start, length = self.index[item]
        group = self.file[self.group_name][name]
        # 当前执行时刻是 start+n_obs_steps-1，对应 predict_action 的起始索引。
        def read(key, count):
            raw_indices = np.arange(start, start + count)
            terminal = length if key != "action" and "sequence_sampling" in self.contract else length - 1
            indices = np.clip(raw_indices, 0, terminal)
            low, high = int(indices.min()), int(indices.max())
            values = group[key][low:high + 1][indices - low].astype(np.float32)
            if key == "action" and self.contract.get("sequence_sampling", {}).get("tail_action") == "zero_arm_keep_gripper":
                values[raw_indices >= length, :-1] = 0
            return torch.from_numpy(values)
        state = read("state", self.n_obs_steps)
        tcp_start = self.contract["state_fields"]["tcp_base_pose_wxyz"][0]
        sample = {"obs": {"pointcloud_xyz": read("pointcloud_xyz", self.n_obs_steps),
                          "ee_position": state[:, tcp_start:tcp_start + 3], "state": state},
                  "action": read("action", self.horizon)}
        if not all(torch.isfinite(x).all() for x in [*sample["obs"].values(), sample["action"]]):
            raise ValueError(f"{name} 含非有限数据")
        return sample

    def normalizer_data(self):
        # 只使用训练 episode，排除 terminal 观测以及窗口重复填充值。
        with h5py.File(self.path, "r") as stream:
            root = stream[self.group_name]
            states = np.concatenate([root[name]["state"][:-1] for name in self.episodes])
            actions = np.concatenate([root[name]["action"][:] for name in self.episodes])
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("归一化统计含非有限数据")
        return states, actions

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["file"] = None
        return state
