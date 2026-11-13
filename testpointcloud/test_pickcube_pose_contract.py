"""PickCube pose actions: historical interfaces and checkpoint execution remain isolated."""
from copy import deepcopy
from dataclasses import asdict
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
from obs_adapter import ObservationConfig, config_from_contract, make_env
from policy import FlowDP3, PolicyConfig
from runtime_utils import load_config, load_policy, save_checkpoint
from task_registry import TASKS
from train import train
from pointcloud_inputs import STORAGE_CONTRACT


class PickCubePoseContractTests(unittest.TestCase):
    def contract(self, version):
        sensors = {"shader_pack": "default"} if version == 1 else None
        return ObservationConfig().contract(sensor_configs=sensors, contract_version=version)

    def test_saved_version_selects_controller_and_camera(self):
        for version, mode, dim in ((1, "pd_ee_delta_pos", 4),
                                   (2, "pd_ee_delta_pos", 4),
                                   (4, "pd_ee_delta_pose", 7),
                                   (5, "pd_ee_delta_pose", 7)):
            with self.subTest(version=version):
                contract = self.contract(version)
                config_from_contract(contract)
                with patch("gymnasium.make") as create:
                    make_env(contract=contract)
                self.assertEqual(create.call_args.kwargs["control_mode"], mode)
                self.assertEqual(contract["action_dim"], dim)
                self.assertEqual(create.call_args.kwargs["sensor_configs"]["base_camera"]["width"],
                                 128 if version == 1 else 256)
                wrong_mode = "pd_ee_delta_pose" if dim == 4 else "pd_ee_delta_pos"
                with self.assertRaisesRegex(ValueError, "控制模式"):
                    make_env(wrong_mode, contract=contract)

    def test_relabeling_old_data_cannot_manufacture_rotation_actions(self):
        old = self.contract(2)
        for changes in ({"version": 4}, {"action_dim": 7},
                        {"control_mode": "pd_ee_delta_pose"}):
            changed = deepcopy(old)
            changed.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config_from_contract(changed)
        new = self.contract(4)
        new["action_dim"] = 4
        with self.assertRaises(ValueError):
            config_from_contract(new)
        with self.assertRaises(ValueError):
            ObservationConfig().contract(contract_version=3)
        with self.assertRaises(ValueError):
            ObservationConfig().contract(env_id="StackCube-v1", contract_version=4)

    def test_pickcube_configs_require_seven_actions_and_other_tasks_are_unchanged(self):
        for name in ("pickcube.yaml", "pickcube_smoke.yaml"):
            config = load_config(ROOT / "examples/baselines/flow_dp3/configs" / name)
            self.assertEqual(config["policy"]["action_dim"], 7)
            self.assertEqual((config["policy"]["radius1_m"], config["policy"]["radius2_m"]), (.05, .12))
        for env_id, mode, dim in (("PushCube-v1", "pd_ee_delta_pos", 4),
                                  ("StackCube-v1", "pd_ee_delta_pose", 7),
                                  ("PegInsertionSide-v1", "pd_ee_delta_pose", 7),
                                  ("DrawTriangle-v1", "pd_ee_delta_pos", 3)):
            self.assertEqual((TASKS[env_id].control_mode, TASKS[env_id].action_dim), (mode, dim))

    def test_retired_checkpoint_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            torch.save({"format_version": 1}, path)
            with self.assertRaisesRegex(ValueError, "旧 checkpoint"):
                load_policy(path, torch.device("cpu"))

    def test_new_training_rejects_old_four_action_data_before_building_model(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "testpointcloud") as directory:
            path = Path(directory) / "legacy.h5"
            with h5py.File(path, "w") as stream:
                stream.attrs["contract"] = json.dumps(self.contract(2))
                stream.attrs["pointcloud_storage"] = json.dumps(STORAGE_CONTRACT)
                stream.attrs["manifest"] = "{}"
                for index in range(3):
                    group = stream.create_group(f"episode_{index:05d}")
                    group.attrs["metadata"] = '{"success_end": true}'
                    group["state"] = np.zeros((21, 28), np.float32)
                    group["action"] = np.zeros((20, 4), np.float32)
                    group["pointcloud_distance"] = np.zeros((21, 512, 4), np.float32)
                    group["pointcloud_xyz"] = np.zeros((21, 512, 3), np.float32)
            args = SimpleNamespace(config=ROOT / "examples/baselines/flow_dp3/configs/pickcube_smoke.yaml",
                                   steps=1, batch_size=1, wandb_log_every=10, wandb_mode="disabled",
                                   device="cpu", data=path, env_id="PickCube-v1")
            with patch("train.FlowDP3", side_effect=AssertionError("must reject before model construction")), \
                    self.assertRaisesRegex(ValueError, "action_dim.*数据契约"):
                train(args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
