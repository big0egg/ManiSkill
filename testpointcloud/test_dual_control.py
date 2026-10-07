"""Regressions for branch isolation, paired rejection and controller compatibility."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/baselines/flow_dp3"))
sys.path.insert(0, str(ROOT / "visual"))
from control_modes import DUAL_LAYOUT, check_control_mode
from dataset import DemoDataset
from dataset_io import episode_names, load_episode
from obs_adapter import ObservationConfig
from prepare_demos import prepare, recording_modes


class FakeEnv(gym.Env):
    control_freq = 20
    num_envs = 1
    def __init__(self):
        self.render_count = 0
        self.step_count = 0

    def reset(self, **kwargs):
        return {}, {"reconfigure": False}

    def step(self, action):
        self.step_count += 1
        return {}, 0., False, False, {}

    def render(self):
        self.render_count += 1
        return np.full((8, 8, 3), self.render_count, np.uint8)

    def close(self):
        pass


class DualControlTests(unittest.TestCase):
    def make_data(self, directory):
        path = Path(directory) / "dual.h5"
        with h5py.File(path, "w") as stream:
            stream.attrs["layout"] = DUAL_LAYOUT
            stream.attrs["manifest"] = "{}"
            for mode, version, length in (("ee", 5, 6), ("joint", 6, 8)):
                contract = ObservationConfig(num_points=128).contract(contract_version=version)
                root = stream.create_group(mode)
                root.attrs["contract"] = json.dumps(contract)
                root.attrs["manifest"] = json.dumps({"branch": mode})
                for i in range(5):
                    group = root.create_group(f"episode_{i:05d}")
                    group.attrs["metadata"] = json.dumps({"success_end": True, "source_episode": i})
                    group["state"] = np.full((length + 1, 28), i + version * 100, np.float32)
                    group["pointcloud_distance"] = np.zeros((length + 1, 128, 4), np.float32)
                    action = np.zeros((length, contract["action_dim"]), np.float32)
                    if mode == "joint":
                        action[:, :7] = [0, .2, 0, -1.5, 0, 2.2, .8]
                    action[:, -1] = -1
                    group["action"] = action
        return path

    def test_same_split_different_lengths_and_branch_only_normalization(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "testpointcloud") as directory:
            path = self.make_data(directory)
            datasets = [DemoDataset(path, split=split, control_mode=mode)
                        for mode in ("ee", "joint") for split in ("train", "val")]
            try:
                ee, ee_val, joint, joint_val = datasets
                self.assertEqual(ee.episodes, joint.episodes)
                self.assertEqual(ee_val.episodes, joint_val.episodes)
                self.assertFalse(set(ee.episodes) & set(ee_val.episodes))
                for data, dimension, length, offset in ((ee, 7, 6, 500), (joint, 8, 8, 600)):
                    sample = data[0]
                    self.assertEqual(sample["action"].shape, (16, dimension))
                    states, actions = data.normalizer_data()
                    self.assertEqual(states.shape, (4 * length, 28))
                    self.assertEqual(actions.shape, (4 * length, dimension))
                    self.assertTrue(np.all((states[:, 0] >= offset) & (states[:, 0] < offset + 5)))
                    if dimension == 8:
                        self.assertTrue(np.all(actions[:, 5] > 1))
            finally:
                for data in datasets:
                    data.close()

    def test_correct_terminal_padding_for_each_branch(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "testpointcloud") as directory:
            path = self.make_data(directory)
            for mode in ("ee", "joint"):
                data = DemoDataset(path, control_mode=mode, horizon=4)
                try:
                    last = data[len(data) - 1]
                    if mode == "ee":
                        torch.testing.assert_close(last["action"][1:, :6], torch.zeros(3, 6))
                    else:
                        self.assertTrue(torch.all(last["action"][:, 5] == 2.2))
                    self.assertTrue(torch.all(last["action"][:, -1] == -1))
                finally:
                    data.close()

    def test_visual_loader_and_default_branch(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "testpointcloud") as directory:
            path = self.make_data(directory)
            self.assertEqual(len(episode_names(path, "joint")), 5)
            self.assertEqual(load_episode(path).actions.shape, (6, 7))
            self.assertEqual(load_episode(path, "0", "joint").actions.shape, (8, 8))
            data = DemoDataset(path)
            try:
                self.assertEqual(data.contract["version"], 5)
            finally:
                data.close()

    def test_explicit_controller_mismatch_rejected_including_legacy_data(self):
        ee = ObservationConfig().contract(contract_version=5)
        joint = ObservationConfig().contract(contract_version=6)
        check_control_mode("ee", ee)
        check_control_mode("joint", joint)
        with self.assertRaisesRegex(ValueError, "不匹配"):
            check_control_mode("joint", ee)
        with self.assertRaisesRegex(ValueError, "不匹配"):
            check_control_mode("ee", joint)
        with tempfile.TemporaryDirectory(dir=ROOT / "testpointcloud") as directory:
            path = self.make_data(directory)
            with h5py.File(path, "r+") as stream:
                del stream.attrs["layout"]
                stream.attrs["contract"] = json.dumps(ee)
            with self.assertRaisesRegex(ValueError, "不匹配"):
                DemoDataset(path, control_mode="joint")

    def test_recording_defaults_and_other_tasks_stay_single_controller(self):
        self.assertEqual(set(recording_modes("PickCube-v1", None)), {"ee", "joint"})
        self.assertEqual(recording_modes("PickCube-v1", "joint")["joint"], ("pd_joint_pos", 6))
        self.assertEqual(recording_modes("PushCube-v1", None)["ee"], ("pd_ee_delta_pos", 3))
        with self.assertRaises(ValueError):
            recording_modes("StackCube-v1", "both")

    def test_one_failed_branch_rejects_both_and_accepted_sources_match(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "testpointcloud") as directory:
            directory = Path(directory)
            source, output = directory / "raw.h5", directory / "paired.h5"
            episodes = [{"episode_id": i, "control_mode": "pd_joint_pos", "reset_kwargs": {"seed": i}}
                        for i in range(2)]
            source.with_suffix(".json").write_text(json.dumps({
                "env_info": {"env_id": "PickCube-v1", "env_kwargs": {}}, "episodes": episodes}))
            with h5py.File(source, "w") as stream:
                for i in range(2):
                    stream.create_group(f"traj_{i}")["actions"] = np.zeros((2, 8))

            def replay(target, original, traj, episode, contract, hold_steps):
                if episode["episode_id"] == 0 and contract["version"] == 6:
                    return None, {"success_end": False}
                target.actions = [np.zeros(contract["action_dim"], np.float32)] * 2
                target.observations = [{"state": np.zeros(28, np.float32),
                                        "pointcloud_distance": np.zeros((128, 4), np.float32)}] * 3
                target.task_metrics = []
                return {"source_episode": episode["episode_id"], "success_end": True, "steps": 2}, None

            args = SimpleNamespace(output=output, generate=None, source=source, env_id=None,
                                   control_mode=None, hold_steps=0, max_steps=None, num_points=128,
                                   length_scale=1., crop_min=None, crop_max=None, count=1, save_video=False)
            with patch("prepare_demos.make_env", side_effect=lambda *a, **kw: FakeEnv()), \
                    patch("prepare_demos.validate_env"), patch("prepare_demos.replay_demo", side_effect=replay):
                prepare(args)
            with h5py.File(output, "r") as stream:
                manifest = json.loads(stream.attrs["manifest"])
                self.assertEqual(manifest["paired_source_episodes"], [1])
                self.assertEqual(manifest["saved"], 1)
                self.assertEqual(manifest["rejected"][0]["source_episode"], 0)
                for mode in ("ee", "joint"):
                    self.assertEqual(list(stream[mode]), ["episode_00000"])
                    self.assertEqual(json.loads(stream[f"{mode}/episode_00000"].attrs["metadata"])["source_episode"], 1)
            self.assertFalse(output.with_suffix(".partial.h5").exists())

    def test_video_sampling_uses_saved_count_and_really_skips_rendering(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "testpointcloud") as directory:
            directory = Path(directory)
            source, output = directory / "raw.h5", directory / "sampled.h5"
            episodes = [{"episode_id": i, "control_mode": "pd_joint_pos", "reset_kwargs": {"seed": i}}
                        for i in range(13)]
            source.with_suffix(".json").write_text(json.dumps({
                "env_info": {"env_id": "PickCube-v1", "env_kwargs": {}}, "episodes": episodes}))
            with h5py.File(source, "w") as stream:
                for i in range(13):
                    stream.create_group(f"traj_{i}")["actions"] = np.zeros((2, 8))
            environments = []
            def make(*a, **kw):
                env = FakeEnv()
                environments.append(env)
                return env
            def replay(target, original, traj, episode, contract, hold_steps):
                target.env.reset()
                target.actions = [np.zeros(contract["action_dim"], np.float32)] * 2
                for action in target.actions:
                    target.env.step(action)
                target.observations = [{"state": np.zeros(28, np.float32),
                                        "pointcloud_distance": np.zeros((128, 4), np.float32)}] * 3
                target.task_metrics = []
                if episode["episode_id"] in (0, 3) and contract["version"] == 6:
                    return None, {"success_end": False}
                return {"source_episode": episode["episode_id"], "seed": episode["episode_id"],
                        "success_end": True, "steps": 2}, None
            videos = []
            def encode(images, output_dir, video_name, **kwargs):
                videos.append((video_name, len(images)))
                (Path(output_dir) / (video_name + ".mp4")).write_bytes(b"test encoder")
            args = SimpleNamespace(output=output, generate=None, source=source, env_id=None,
                                   control_mode=None, hold_steps=0, max_steps=None, num_points=128,
                                   length_scale=1., crop_min=None, crop_max=None, count=11, save_video=True)
            with patch("prepare_demos.make_env", side_effect=make), \
                    patch("prepare_demos.validate_env"), patch("prepare_demos.replay_demo", side_effect=replay), \
                    patch("mani_skill.utils.wrappers.record.images_to_video", side_effect=encode):
                prepare(args)
            self.assertEqual(videos, [("episode_00000_seed_1", 3)] * 2 +
                                    [("episode_00010_seed_12", 3)] * 2)
            for env in environments[1:]:
                self.assertEqual(env.step_count, 26)
                self.assertEqual(env.render_count, 9)  # failed selected + two accepted selected; others zero
            with h5py.File(output, "r") as stream:
                manifest = json.loads(stream.attrs["manifest"])
                self.assertEqual(manifest["video_every"], 10)
                self.assertEqual(manifest["saved"], 11)
                for mode in ("ee", "joint"):
                    self.assertEqual(len(stream[mode]), 11)
                    recorded = []
                    for name in sorted(stream[mode]):
                        meta = json.loads(stream[f"{mode}/{name}"].attrs["metadata"])
                        if meta["video_recorded"]:
                            recorded.append(name)
                            self.assertEqual(meta["video_frames"], 3)
                        else:
                            self.assertNotIn("video_path", meta)
                    self.assertEqual(recorded, ["episode_00000", "episode_00010"])

    def test_video_trigger_can_switch_without_leaking_frames(self):
        from mani_skill.utils.wrappers.record import RecordEpisode
        with tempfile.TemporaryDirectory(dir=ROOT / "testpointcloud") as directory:
            gate = {"enabled": False}
            env = FakeEnv()
            record = RecordEpisode(env, directory, save_trajectory=False, save_video=True,
                                   save_on_reset=False, save_video_trigger=lambda step: gate["enabled"])
            try:
                record.reset()
                record.step(np.zeros(8))
                self.assertEqual(env.render_count, 0)
                self.assertEqual(record.render_images, [])
                gate["enabled"] = True
                record.reset()
                for _ in range(2):
                    record.step(np.zeros(8))
                self.assertEqual(len(record.render_images), 3)
                record.flush_video(save=False)
                gate["enabled"] = False
                record.reset()
                record.step(np.zeros(8))
                self.assertEqual(env.render_count, 3)
                self.assertEqual(record.render_images, [])
            finally:
                record.close()

    def test_invalid_video_interval_rejected_before_generation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "testpointcloud") as directory:
            for interval in (0, -1, 1.5):
                with self.assertRaisesRegex(ValueError, "video-every"):
                    prepare(SimpleNamespace(output=Path(directory) / "data.h5", save_video=True,
                                            video_every=interval))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
