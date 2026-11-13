"""Policy control is EE-only; expert joint trajectories remain convertible."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control_modes import CONTROL_CHOICES, check_control_mode, select_dataset_group
from obs_adapter import ObservationConfig, config_from_contract
from task_registry import TASKS


class EEControlTests(unittest.TestCase):
    def test_all_tasks_keep_their_ee_action_dimensions(self):
        for name, task in TASKS.items():
            contract = ObservationConfig().contract(env_id=name)
            check_control_mode(None, contract)
            check_control_mode("ee", contract)
            self.assertIn(task.action_dim, (3, 4, 7))
        self.assertNotIn("joint", CONTROL_CHOICES)
        self.assertNotIn("pd_joint_pos", CONTROL_CHOICES)

    def test_joint_contract_rejected_even_without_cli_override(self):
        contract = ObservationConfig().contract()
        for requested in ("joint", "both", "pd_joint_pos"):
            with self.assertRaises(ValueError):
                check_control_mode(requested, contract)
        contract.update(version=6, control_mode="pd_joint_pos", action_dim=8)
        with self.assertRaises(ValueError):
            check_control_mode(None, contract)
        with self.assertRaises(ValueError):
            config_from_contract(contract)

    def test_retired_layout_rejected_instead_of_selecting_a_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retired.h5"
            with h5py.File(path, "w") as stream:
                stream.attrs["layout"] = "dual_control_v1"
                group = stream.create_group("ee")
                group.attrs["contract"] = json.dumps(ObservationConfig().contract())
                with self.assertRaisesRegex(ValueError, "单分支 EE"):
                    select_dataset_group(stream)
