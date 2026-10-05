"""验证真实风险：动作时序、episode 隔离、训练统计与几何坐标转换。"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import DemoDataset
from obs_adapter import ObservationConfig, pointcloud_features
from policy import LimitsNormalizer, PolicyConfig


class DataContractTests(unittest.TestCase):
    def make_data(self, path):
        with h5py.File(path, "w") as stream:
            stream.attrs["contract"] = json.dumps(ObservationConfig(num_points=128).contract())
            stream.attrs["manifest"] = "{}"
            for i in range(3):
                group = stream.create_group(f"episode_{i:05d}")
                group.attrs["metadata"] = '{"success_end": true}'
                group["state"] = np.repeat((100 * i + np.arange(7))[:, None], 28, axis=1).astype(np.float32)
                group["action"] = np.repeat(np.arange(6)[:, None], 4, axis=1).astype(np.float32)
                group["pointcloud_distance"] = np.zeros((7, 128, 4), dtype=np.float32)

    def test_episode_split_and_action_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.h5"
            self.make_data(path)
            train = DemoDataset(path, horizon=4, n_obs_steps=2, n_action_steps=2, split="train")
            val = DemoDataset(path, horizon=4, n_obs_steps=2, n_action_steps=2, split="val")
            self.assertFalse(set(train.episodes) & set(val.episodes))
            # 第一帧两次重复：predict_action 丢弃的索引0后，第一个执行标签仍是 a_0。
            sample = train[0]
            self.assertEqual(sample["action"][:, 0].tolist(), [0, 0, 1, 2])
            self.assertEqual(sample["obs"]["state"][:, 0].tolist()[0],
                             sample["obs"]["state"][:, 0].tolist()[1])
            last = train[4]
            self.assertEqual(last["action"][:, 0].tolist(), [3, 4, 5, 5])
            states, actions = train.normalizer_data()
            self.assertEqual(len(states), 12)
            excluded = int(val.episodes[0].split("_")[1]) * 100
            self.assertFalse(np.any((states[:, 0] >= excluded) & (states[:, 0] < excluded + 7)))
            normalizer = LimitsNormalizer(28)
            normalizer.fit(states)
            reconstructed = normalizer.unnormalize(normalizer.normalize(torch.from_numpy(states)))
            torch.testing.assert_close(reconstructed, torch.from_numpy(states), atol=2e-5, rtol=1e-5)
            train.close()
            val.close()

    def test_base_transform_and_single_scaling(self):
        # 构造旋转且平移的基座，已知基座坐标；无需依赖仿真设备。
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        translation = torch.tensor([3., -2., 1.])
        inverse = torch.eye(4)
        inverse[:3, :3] = rotation.T
        inverse[:3, 3] = -rotation.T @ translation
        pose = SimpleNamespace(inv=lambda: SimpleNamespace(to_transformation_matrix=lambda: inverse[None]))
        tcp_base = torch.tensor([0.6, 0.0, 0.2])
        torch.manual_seed(9)
        base = torch.rand(128, 3) * 0.1 + torch.tensor([0.5, -0.05, 0.1])
        world = base @ rotation.T + translation
        agent = SimpleNamespace(robot=SimpleNamespace(pose=pose),
                                tcp_pose=SimpleNamespace(p=(rotation @ tcp_base + translation)[None]))
        obs = {"pointcloud": {"xyzw": torch.cat((world, torch.ones(128, 1)), dim=-1)[None]}}
        features, detail = pointcloud_features(obs, agent, ObservationConfig(num_points=128, length_scale=2))
        actual = features[0, :, :3] * 2 + tcp_base
        # FPS 可以重排，但每个原始点必须保留一次。
        nearest = torch.cdist(actual, base).amin(dim=1)
        self.assertLess(float(nearest.max()), 1e-3)
        self.assertLess(detail["distance_norm_error"], 1e-6)
        torch.testing.assert_close(torch.tensor(detail["tcp_base_m"]), tcp_base, atol=1e-6, rtol=1e-6)
        invalid = obs["pointcloud"]["xyzw"].clone()
        invalid[0, 0, 3] = 0
        with self.assertRaises(ValueError):
            pointcloud_features({"pointcloud": {"xyzw": invalid}}, agent, ObservationConfig(num_points=128))

    def test_constant_normalizer_and_bad_horizon(self):
        normalizer = LimitsNormalizer(2)
        data = torch.tensor([[7., -1.], [7., 1.]])
        normalizer.fit(data)
        torch.testing.assert_close(normalizer.normalize(data), torch.tensor([[0., -1.], [0., 1.]]))
        with self.assertRaises(ValueError):
            PolicyConfig(horizon=15)

    def test_default_crop_preserves_full_scene_points(self):
        # 实际场景包含远处地面、桌腿与基座后方，不能只保留桌面操作区域。
        generator = torch.Generator().manual_seed(17)
        base = torch.rand(128, 3, generator=generator)
        base = base * torch.tensor([100., 100., 1.9]) + torch.tensor([-49.385, -50., -0.92])
        inverse = torch.eye(4)
        pose = SimpleNamespace(inv=lambda: SimpleNamespace(to_transformation_matrix=lambda: inverse[None]))
        tcp = torch.tensor([0.6, 0., 0.2])
        agent = SimpleNamespace(robot=SimpleNamespace(pose=pose),
                                tcp_pose=SimpleNamespace(p=tcp[None]))
        obs = {"pointcloud": {"xyzw": torch.cat((base, torch.ones(128, 1)), dim=-1)[None]}}
        features, detail = pointcloud_features(obs, agent, ObservationConfig(num_points=128))
        self.assertEqual(detail["cropped_points"], len(base))
        actual = features[0, :, :3] + tcp
        # 无预采样且 FPS 保留所有索引时，逐点重建应完全覆盖输入场景。
        for axis in (2, 1, 0):
            actual = actual[torch.argsort(actual[:, axis], stable=True)]
            base = base[torch.argsort(base[:, axis], stable=True)]
        torch.testing.assert_close(actual, base, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
