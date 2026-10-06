"""按整条 episode 划分、逐窗口读取；边缘重复填充与参考 SequenceSampler 一致。"""
from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from obs_adapter import config_from_contract


class DemoDataset(Dataset):
    def __init__(self, path, horizon=16, n_obs_steps=2, n_action_steps=8,
                 split="train", val_ratio=0.2, seed=42):
        self.path = str(path)
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.file = None
        self.index = []
        if split not in ("train", "val") or not 0 < val_ratio < 1:
            raise ValueError("需要 train/val 划分及 0 < val_ratio < 1")
        with h5py.File(self.path, "r") as stream:
            self.contract = json.loads(stream.attrs["contract"])
            self.manifest = json.loads(stream.attrs["manifest"])
            config_from_contract(self.contract)
            episodes = sorted(stream.keys())
            if len(episodes) < 2:
                raise ValueError("至少需要两条成功示范，才能按 episode 留出验证集")
            n_val = min(max(1, round(len(episodes) * val_ratio)), len(episodes) - 1)
            val_indices = set(np.random.default_rng(seed).choice(len(episodes), n_val, replace=False).tolist())
            self.episodes = [name for i, name in enumerate(episodes) if (i in val_indices) == (split == "val")]
            for name in self.episodes:
                group = stream[name]
                length = len(group["action"])
                if group["state"].shape != (length + 1, self.contract["state_dim"]) or group["pointcloud_distance"].shape != (
                    length + 1, self.contract["pointcloud"]["num_points"], 4
                ) or group["action"].shape != (length, self.contract["action_dim"]):
                    raise ValueError(f"{name} 的 T+1 观测 / T 动作形状异常")
                if not json.loads(group.attrs["metadata"])["success_end"]:
                    raise ValueError("训练数据中包含失败回放")
                # 参考 pad_before=n_obs_steps-1，pad_after=n_action_steps-1。
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
        group = self.file[name]
        # 当前执行时刻是 start+n_obs_steps-1，对应 predict_action 的起始索引。
        def read(key, count):
            indices = np.clip(np.arange(start, start + count), 0, length - 1)
            low, high = int(indices.min()), int(indices.max())
            return torch.from_numpy(group[key][low:high + 1][indices - low].astype(np.float32))
        sample = {"obs": {"pointcloud_distance": read("pointcloud_distance", self.n_obs_steps),
                          "state": read("state", self.n_obs_steps)}, "action": read("action", self.horizon)}
        if not all(torch.isfinite(x).all() for x in [*sample["obs"].values(), sample["action"]]):
            raise ValueError(f"{name} 含非有限数据")
        distance = sample["obs"]["pointcloud_distance"]
        if (distance[..., 3] - distance[..., :3].norm(dim=-1)).abs().max() > 1e-5:
            raise ValueError(f"{name} 的距离通道与 xyz 不匹配")
        return sample

    def normalizer_data(self):
        # 只使用训练 episode，排除 terminal 观测以及窗口重复填充值。
        with h5py.File(self.path, "r") as stream:
            states = np.concatenate([stream[name]["state"][:-1] for name in self.episodes])
            actions = np.concatenate([stream[name]["action"][:] for name in self.episodes])
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
