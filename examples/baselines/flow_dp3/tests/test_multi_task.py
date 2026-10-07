"""Task isolation, dynamic network dimensions, and failed checkpoint writes."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import DemoDataset
from obs_adapter import ObservationConfig, adapt_observation, config_from_contract, make_env
from policy import FlowDP3, PolicyConfig
from runtime_utils import save_checkpoint, load_policy
from scene_bounds import get_scene_crop_bounds
from task_registry import TASKS
from mani_skill.utils.structs import Pose
from prepare_demos import ExpertStepLimit, EpisodeLimitError


class MultiTaskTests(unittest.TestCase):
    def test_expert_ignoring_truncation_cannot_overrun_draw_buffer(self):
        import gymnasium as gym
        class IgnoringExpertEnv(gym.Env):
            def reset(self, **kwargs):
                self.calls = 0
                return {}, {}
            def step(self, action):
                self.calls += 1
                return {}, 0, False, True, {}
        base = IgnoringExpertEnv()
        env = ExpertStepLimit(base, 2)
        env.reset()
        env.step(None)
        env.step(None)
        with self.assertRaises(EpisodeLimitError):
            env.step(None)
        self.assertEqual(base.calls, 2)
        env.reset()
        env.step(None)
        self.assertEqual(base.calls, 1)

    def contract(self, env_id):
        low, high = get_scene_crop_bounds(env_id)
        return ObservationConfig(crop_min=low, crop_max=high).contract(env_id=env_id)

    def test_task_contracts_and_mismatch_rejection(self):
        for env_id, task in TASKS.items():
            with self.subTest(task=env_id):
                contract = self.contract(env_id)
                config_from_contract(contract)
                changed = copy.deepcopy(contract)
                changed["action_dim"] += 1
                with self.assertRaisesRegex(ValueError, "适配接口"):
                    config_from_contract(changed)
                with self.assertRaisesRegex(ValueError, "不匹配"):
                    make_env(contract=contract, env_id="Unknown-v1")
                self.assertEqual(contract["state_dim"], 28 if env_id == "PickCube-v1" else (21 if task.robot == "panda_stick" else 25))
                self.assertEqual("hand_camera" in contract["sensor_configs"], task.robot == "panda_wristcam")

    def test_new_visual_tasks_do_not_read_hidden_object_or_goal_state(self):
        xyz = torch.rand(512, 3, generator=torch.Generator().manual_seed(3))*.08 + torch.tensor([.52, -.08, .1])
        for env_id, task in TASKS.items():
            if env_id == "PickCube-v1":
                continue
            with self.subTest(task=env_id):
                contract = self.contract(env_id)
                tcp = Pose.create_from_pq(p=[.6, 0, .2])
                agent = SimpleNamespace(robot=SimpleNamespace(pose=Pose.create_from_pq(p=[0, 0, 0])),
                                        tcp=SimpleNamespace(pose=tcp))
                obs = {"pointcloud": {"xyzw": torch.cat((xyz, torch.ones(512, 1)), -1)[None]},
                       "agent": {"qpos": torch.zeros(1, task.joints), "qvel": torch.zeros(1, task.joints)},
                       "extra": {}}
                result, _ = adapt_observation(obs, agent, config_from_contract(contract), contract=contract)
                self.assertEqual(result["state"].shape, (1, task.state_dim))
                self.assertEqual(result["pointcloud_distance"].shape, (1, 512, 4))
                np.testing.assert_allclose(result["state"][0, -7:-4], [.6, 0, .2])

    def test_dynamic_dataset_loss_prediction_and_checkpoint_roundtrip(self):
        for env_id, task in TASKS.items():
            with self.subTest(task=env_id), tempfile.TemporaryDirectory() as directory:
                contract = self.contract(env_id)
                path = Path(directory)/"data.h5"
                with h5py.File(path, "w") as f:
                    f.attrs["contract"] = json.dumps(contract)
                    f.attrs["manifest"] = "{}"
                    for i in range(3):
                        g = f.create_group(f"episode_{i:05d}")
                        g.attrs["metadata"] = '{"success_end": true}'
                        g["state"] = np.zeros((21, task.state_dim), np.float32)
                        g["action"] = np.zeros((20, task.action_dim), np.float32)
                        g["pointcloud_distance"] = np.zeros((21, 512, 4), np.float32)
                data = DemoDataset(path)
                sample = data[0]
                batch = {"obs": {key: value[None] for key, value in sample["obs"].items()},
                         "action": sample["action"][None]}
                config = PolicyConfig(state_dim=task.state_dim, action_dim=task.action_dim,
                                      down_dims=(64, 128, 256), num_inference_steps=1)
                model = FlowDP3(config)
                loss = model.compute_loss(batch)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNotNone(model.state_mlp[0].weight.grad)
                model.eval()
                torch.manual_seed(12)
                expected = model.predict_action(batch["obs"])["action"]
                checkpoint = Path(directory)/"model.pt"
                from dataclasses import asdict
                save_checkpoint(checkpoint, {"format_version": 1, "contract": contract,
                                            "config": {"policy": asdict(config)},
                                            "model": model.state_dict(), "ema": model.state_dict()})
                loaded, _ = load_policy(checkpoint, torch.device("cpu"))
                torch.manual_seed(12)
                torch.testing.assert_close(loaded.predict_action(batch["obs"])["action"], expected)
                self.assertEqual(expected.shape, (1, 8, task.action_dim))
                data.close()

    def test_atomic_checkpoint_failure_preserves_previous_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"last.pt"
            save_checkpoint(path, {"x": torch.tensor([1])})
            original = path.read_bytes()
            def fail(payload, temporary):
                Path(temporary).write_bytes(b"partial")
                raise OSError("simulated storage failure")
            with patch("runtime_utils.torch.save", side_effect=fail), self.assertRaises(OSError):
                save_checkpoint(path, {"x": torch.tensor([2])})
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(path.with_suffix(".partial.pt").exists())
            with patch("shutil.disk_usage", return_value=SimpleNamespace(free=0)), self.assertRaisesRegex(OSError, "空闲空间"):
                save_checkpoint(path, {"x": torch.tensor([2])})


if __name__ == "__main__":
    unittest.main()
