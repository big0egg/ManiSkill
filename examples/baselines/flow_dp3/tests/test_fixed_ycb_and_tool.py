"""Fixed geometry replay, real grasp transforms and tool-use quality gates."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import sapien
from transforms3d.euler import euler2quat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obs_adapter import ObservationConfig, config_from_contract, make_env, validate_env
from scene_bounds import get_scene_crop_bounds
from task_registry import validate_source_env_kwargs
from task_metrics import demonstration_success, demonstration_trace_success
from mani_skill.envs.tasks.tabletop.pick_single_ycb import PickSingleYCBEnv
from mani_skill.examples.motionplanning.panda.solutions.pick_single_ycb import object_goal_tcp_pose


class FixedYCBAndToolTests(unittest.TestCase):
    def contract(self):
        low, high = get_scene_crop_bounds("PickSingleYCB-v1")
        return ObservationConfig(crop_min=low, crop_max=high).contract(env_id="PickSingleYCB-v1")

    def test_model_configuration_survives_contract_and_rejects_other_geometry(self):
        contract = self.contract()
        self.assertEqual(contract["env_kwargs"], {"model_ids": ["004_sugar_box"]})
        self.assertEqual(contract["state_dim"], 28)
        config_from_contract(contract)
        for kwargs in ({}, {"model_ids": ["006_mustard_bottle"]}, {"model_ids": []}):
            changed = copy.deepcopy(contract)
            changed["env_kwargs"] = kwargs
            with self.assertRaises(ValueError):
                config_from_contract(changed)
            with self.assertRaises(ValueError):
                validate_source_env_kwargs("PickSingleYCB-v1", kwargs)
        validate_source_env_kwargs("PickSingleYCB-v1", {"model_ids": ["004_sugar_box"]})

    def test_only_requested_model_is_available_across_scene_reconfiguration(self):
        metadata = {"004_sugar_box": {}, "006_mustard_bottle": {}, "029_plate": {}}
        with patch("mani_skill.envs.tasks.tabletop.pick_single_ycb.load_json", return_value=metadata), \
                patch("mani_skill.envs.tasks.tabletop.pick_single_ycb.BaseEnv.__init__", return_value=None):
            fixed = PickSingleYCBEnv(model_ids=["004_sugar_box"])
            self.assertEqual(fixed.all_model_ids.tolist(), ["004_sugar_box"])
            for seed in range(20):
                sampled = np.random.default_rng(seed).choice(fixed.all_model_ids, 10)
                self.assertTrue((sampled == "004_sugar_box").all())
            self.assertEqual(set(PickSingleYCBEnv().all_model_ids), {"004_sugar_box", "006_mustard_bottle"})
            for ids in ([], "004_sugar_box", ["missing"], ["029_plate"], ["004_sugar_box"] * 2):
                with self.assertRaises(ValueError):
                    PickSingleYCBEnv(model_ids=ids)

    def test_environment_gets_fixed_models_and_live_geometry_is_checked(self):
        contract = self.contract()
        fake = SimpleNamespace(spec=SimpleNamespace(id="PickSingleYCB-v1"),
                               agent=SimpleNamespace(uid="panda"),
                               all_model_ids=("004_sugar_box",), model_ids=("004_sugar_box",))
        wrapped = SimpleNamespace(unwrapped=fake, action_space=SimpleNamespace(
            shape=(7,), low=np.full(7, -1), high=np.ones(7)))
        with patch("pathlib.Path.is_file", return_value=True), patch("gymnasium.make", return_value=wrapped) as make:
            env = make_env(contract=contract)
            self.assertEqual(make.call_args.kwargs["model_ids"], ["004_sugar_box"])
            validate_env(env, contract)
            fake.model_ids = ("006_mustard_bottle",)
            with self.assertRaises(ValueError):
                validate_env(env, contract)
        with patch("pathlib.Path.is_file", return_value=False), self.assertRaisesRegex(FileNotFoundError, "download_asset ycb"):
            make_env(contract=contract)

    def test_goal_moves_object_origin_instead_of_tcp_origin(self):
        tcp = sapien.Pose([.2, -.1, .3], euler2quat(.2, -.3, .6))
        relative = sapien.Pose([.01, -.02, .07], euler2quat(.1, 0, .4))
        obj = tcp * relative
        goal = np.array([-.15, .13, .4])
        target = object_goal_tcp_pose(tcp, obj, goal)
        actual_obj = target * relative
        np.testing.assert_allclose(actual_obj.p, goal, atol=1e-6)
        self.assertAlmostEqual(abs(float(actual_obj.q @ obj.q)), 1., places=5)
        self.assertGreater(np.linalg.norm(target.p - goal), .05)

    def test_success_without_tool_use_is_not_accepted_as_tool_demonstration(self):
        no_tool = [{"is_tool_grasped": False, "tool_cube_contact": True}]
        unrelated = [{"is_tool_grasped": True, "tool_cube_contact": False},
                     {"is_tool_grasped": False, "tool_cube_contact": True}]
        valid = [{"is_tool_grasped": True, "tool_cube_contact": True}]
        self.assertFalse(demonstration_trace_success("PullCubeTool-v1", no_tool))
        self.assertFalse(demonstration_trace_success("PullCubeTool-v1", unrelated))
        self.assertTrue(demonstration_trace_success("PullCubeTool-v1", valid))
        self.assertFalse(demonstration_success("PickSingleYCB-v1", {"success": True, "is_grasped": False}))


if __name__ == "__main__":
    unittest.main()
