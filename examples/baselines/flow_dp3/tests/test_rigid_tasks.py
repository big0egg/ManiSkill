"""Regressions for upright geometry, observable goals and terminal actions."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from transforms3d.euler import euler2quat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import DemoDataset
from obs_adapter import ObservationConfig, adapt_observation, config_from_contract
from scene_bounds import get_scene_crop_bounds
from task_metrics import demonstration_success
from mani_skill.envs.tasks.tabletop.lift_peg_upright import LiftPegUprightEnv
from mani_skill.utils.structs import Pose


class RigidTaskTests(unittest.TestCase):
    def contract(self, env_id):
        low, high = get_scene_crop_bounds(env_id)
        return ObservationConfig(crop_min=low, crop_max=high).contract(env_id=env_id)

    def test_uprightness_uses_long_axis_not_yaw(self):
        angles = [(0, np.pi/2, yaw) for yaw in (0, .8, 2.5)]
        angles += [(0, np.pi/2 - .079, .6), (0, np.pi/2 - .081, .6), (0, 0, np.pi/2)]
        poses = Pose.create_from_pq(p=torch.tensor([[0., 0., .12]] * len(angles)),
                                   q=torch.tensor(np.array([euler2quat(*a) for a in angles]), dtype=torch.float32))
        env = SimpleNamespace(peg=SimpleNamespace(pose=poses), peg_half_length=.12)
        result = LiftPegUprightEnv.evaluate(env)
        self.assertEqual(result["success"].tolist(), [True, True, True, True, False, False])
        poses.p = torch.tensor([[0., 0., .126]] * len(angles))
        self.assertFalse(LiftPegUprightEnv.evaluate(env)["success"].any())

    def test_pull_goal_transforms_from_world_to_robot_base(self):
        base_pose = Pose.create_from_pq(p=[1, -2, 0], q=euler2quat(0, 0, np.pi/2))
        tcp = base_pose * Pose.create_from_pq(p=[.6, 0, .2])
        agent = SimpleNamespace(robot=SimpleNamespace(pose=base_pose), tcp=SimpleNamespace(pose=tcp))
        contract = self.contract("PullCube-v1")
        goal_world = (base_pose * Pose.create_from_pq(p=[.315, -.1, .001])).p
        obs = {"agent": {"qpos": torch.zeros(1, 9), "qvel": torch.zeros(1, 9)},
               "extra": {"goal_pos": goal_world, "obj_pose": "must never read this", "success": True}}
        with patch("obs_adapter.pointcloud_features", return_value=(torch.zeros(1, 512, 4), {})):
            adapted, _ = adapt_observation(obs, agent, config_from_contract(contract), contract=contract)
        torch.testing.assert_close(adapted["state"][0, 25:], torch.tensor([.315, -.1, .001]), atol=1e-6, rtol=1e-6)

    def test_new_tasks_include_terminal_and_zero_delta_tail(self):
        for env_id in ("LiftPegUpright-v1", "PlaceSphere-v1", "PullCube-v1"):
            with self.subTest(task=env_id), tempfile.TemporaryDirectory() as directory:
                contract = self.contract(env_id)
                path = Path(directory)/"data.h5"
                with h5py.File(path, "w") as f:
                    f.attrs["contract"] = json.dumps(contract)
                    f.attrs["manifest"] = "{}"
                    for i in range(2):
                        g = f.create_group(f"episode_{i:05d}")
                        g.attrs["metadata"] = '{"success_end": true}'
                        g["state"] = np.repeat(np.arange(4)[:, None], contract["state_dim"], axis=1).astype(np.float32)
                        g["pointcloud_distance"] = np.zeros((4, 512, 4), np.float32)
                        g["action"] = np.ones((3, contract["action_dim"]), np.float32) * .5
                data = DemoDataset(path, horizon=16)
                self.assertEqual(len(data), 4)
                terminal = data[-1]
                self.assertEqual(terminal["obs"]["state"][:, 0].tolist(), [2, 3])
                self.assertTrue((terminal["action"][1:, :-1] == 0).all())
                self.assertTrue((terminal["action"][:, -1] == .5).all())
                data.close()
                changed = copy.deepcopy(contract)
                changed["sequence_sampling"]["tail_action"] = "repeat_absolute_target"
                with self.assertRaises(ValueError):
                    config_from_contract(changed)

    def test_supported_peg_does_not_count_as_valid_demonstration(self):
        row = {"success": True, "is_grasped": True, "is_obj_static": True}
        self.assertFalse(demonstration_success("LiftPegUpright-v1", row))
        row["is_grasped"] = False
        row["is_obj_static"] = False
        self.assertFalse(demonstration_success("LiftPegUpright-v1", row))
        row["is_obj_static"] = True
        self.assertTrue(demonstration_success("LiftPegUpright-v1", row))


if __name__ == "__main__":
    unittest.main()
