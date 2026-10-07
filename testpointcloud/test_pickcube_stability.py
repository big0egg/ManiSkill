"""Meaningful regressions for terminal conditioning, delta padding and radian actions."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/baselines/flow_dp3"))
sys.path.insert(0, str(ROOT / "visual"))
from dataset_io import load_episode
from dataset import DemoDataset
from obs_adapter import ObservationConfig, clip_action, config_from_contract
from prepare_demos import pose_hold_action
from train import train
from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry.rotation_conversions import euler_angles_to_matrix, matrix_to_quaternion


class StabilityTests(unittest.TestCase):
    def data(self, directory, version, length=6):
        contract = ObservationConfig(num_points=128).contract(contract_version=version)
        path = Path(directory) / f"v{version}.h5"
        with h5py.File(path, "w") as f:
            f.attrs["contract"] = json.dumps(contract)
            f.attrs["manifest"] = "{}"
            for i in range(3):
                g = f.create_group(f"episode_{i:05d}")
                g.attrs["metadata"] = '{"success_end":true}'
                g["state"] = np.repeat(np.arange(length + 1, dtype=np.float32)[:, None], 28, 1)
                g["pointcloud_distance"] = np.zeros((length + 1, 128, 4), np.float32)
                actions = np.full((length, contract["action_dim"]), .03, np.float32)
                actions[:, -1] = -1
                if version == 6:
                    actions[:, :7] = [0, .2, 0, -1.5, 0, 2.2, .8]
                g["action"] = actions
        return path

    def test_terminal_observation_is_condition_and_delta_padding_stops(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "testpointcloud") as directory:
            data = DemoDataset(self.data(directory, 5), horizon=4, n_obs_steps=2, n_action_steps=2)
            try:
                indices = [i for i, entry in enumerate(data.index) if entry[0] == data.episodes[0]]
                self.assertEqual(len(indices), 7)  # t=0 through the real terminal t=6.
                last = data[indices[-1]]
                self.assertEqual(last["obs"]["state"][:, 0].tolist(), [5, 6])
                torch.testing.assert_close(last["action"][1:, :6], torch.zeros(3, 6))
                torch.testing.assert_close(last["action"][:, -1], -torch.ones(4))
                self.assertAlmostEqual(last["action"][0, 0].item(), .03)
            finally:
                data.close()

    def test_absolute_joint_padding_preserves_the_fixed_target(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "testpointcloud") as directory:
            data = DemoDataset(self.data(directory, 6), horizon=4, n_obs_steps=2, n_action_steps=2)
            try:
                index = next(i for i, entry in enumerate(data.index) if entry[1] == 5)
                sample = data[index]
                torch.testing.assert_close(sample["action"], sample["action"][0:1].expand(4, 8))
                self.assertGreater(sample["action"][1, 5], 1)  # radians are preserved.
            finally:
                data.close()

    def test_legacy_windows_and_padding_remain_unchanged(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "testpointcloud") as directory:
            data = DemoDataset(self.data(directory, 4), horizon=4, n_obs_steps=2, n_action_steps=2)
            try:
                indices = [i for i, entry in enumerate(data.index) if entry[0] == data.episodes[0]]
                self.assertEqual(len(indices), 5)
                last = data[indices[-1]]
                self.assertEqual(last["obs"]["state"][:, 0].tolist(), [3, 4])
                self.assertAlmostEqual(last["action"][-1, 0].item(), .03)
            finally:
                data.close()

    def test_joint_bounds_do_not_clip_valid_angles_to_one_radian(self):
        contract = ObservationConfig().contract(contract_version=6)
        config_from_contract(contract)
        action = torch.tensor([0, .2, 0, -1.5, 0, 2.2, .8, -1.])
        torch.testing.assert_close(clip_action(action, contract), action)
        action[0] = 10
        self.assertAlmostEqual(clip_action(action, contract)[0].item(), contract["action_space"]["high"][0])
        contract["action_space"]["high"][0] = 1
        with self.assertRaises(ValueError):
            config_from_contract(contract)
        with tempfile.TemporaryDirectory(dir=ROOT / "testpointcloud") as directory:
            path = self.data(directory, 6)
            episode = load_episode(path, '0')
            self.assertGreater(episode.actions[:, 5].min(), 1)
            with h5py.File(path, 'r+') as stream:
                stream['episode_00000/action'][0, 0] = 10
            with self.assertRaisesRegex(ValueError, "控制器范围"):
                load_episode(path, '0')

    def test_new_training_rejects_old_seven_dimensional_data_without_holding_contract(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "testpointcloud") as directory:
            args = SimpleNamespace(config=ROOT / "examples/baselines/flow_dp3/configs/pickcube_smoke.yaml",
                                   steps=1, batch_size=1, wandb_log_every=10, wandb_mode="disabled",
                                   device="cpu", data=self.data(directory, 4, length=20), env_id="PickCube-v1")
            with patch("train.FlowDP3", side_effect=AssertionError("must reject before model construction")), \
                    self.assertRaisesRegex(ValueError, "要求数据契约 v5"):
                train(args)

    def test_fixed_pose_correction_matches_local_controller_rotation_convention(self):
        current = Pose.create_from_pq(p=[0, 0, 0])
        target_rotation = euler_angles_to_matrix(torch.tensor([[.04, -.03, .02]]), "XYZ")
        target = Pose.create_from_pq(p=[.01, .02, -.01], q=matrix_to_quaternion(target_rotation))
        config = SimpleNamespace(pos_lower=-.1, pos_upper=.1, rot_lower=-.1,
                                 use_delta=True, frame="root_translation:root_aligned_body_rotation")
        controller = SimpleNamespace(config=config, ee_pose_at_base=current)
        env = SimpleNamespace(unwrapped=SimpleNamespace(agent=SimpleNamespace(
            controller=SimpleNamespace(controllers={"arm": controller}))))
        action = pose_hold_action(env, target, -1)
        scaled = torch.cat((action[:3] * .1, action[3:6] * -.1))[None]
        fake = SimpleNamespace(config=config)
        result = PDEEPoseController.compute_target_pose(fake, current, scaled)
        torch.testing.assert_close(result.to_transformation_matrix(), target.to_transformation_matrix(), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
