"""W&B 接口约束与 RNG 隔离测试；Mock 仅模拟 SDK，真实 SDK 单独进行离线检查。"""
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_logging import ExperimentLogger


def args(mode="offline"):
    return SimpleNamespace(wandb_mode=mode, wandb_project=None, wandb_entity=None,
                           wandb_name=None, wandb_run_id=None)


class FakeRun:
    def __init__(self, kwargs):
        self.id = kwargs["id"]
        self.project = kwargs["project"]
        self.entity = kwargs["entity"] or "test-user"
        self.url = "https://wandb.ai/test-user/flow_dp3/runs/" + self.id
        self.summary = {}
        self.logs = []
        self.metrics = []
        self.exit_code = None

    def define_metric(self, name, **kwargs):
        self.metrics.append((name, kwargs))

    def log(self, data):
        random.random()
        np.random.rand()
        torch.rand(2)
        self.logs.append(data)

    def finish(self, exit_code=0):
        self.exit_code = exit_code


class FakeSDK:
    Settings = SimpleNamespace

    def __init__(self):
        self.calls = []
        self.runs = []

    def init(self, **kwargs):
        random.random()
        np.random.rand()
        torch.rand(2)
        self.calls.append(kwargs)
        self.runs.append(FakeRun(kwargs))
        return self.runs[-1]

    def Video(self, path, **kwargs):
        return {"path": path, **kwargs}


class LoggingTests(unittest.TestCase):
    def test_disabled_needs_no_sdk(self):
        with patch("experiment_logging.importlib.import_module", side_effect=AssertionError("不能导入 SDK")):
            logger = ExperimentLogger(args("disabled"), "/unused", {}, "train")
            logger.log_training({"step": 1})
            logger.finish()
        self.assertIsNone(logger.metadata)

    def test_rng_resume_identity_and_video_mapping(self):
        sdk = FakeSDK()
        with tempfile.TemporaryDirectory() as directory, patch("experiment_logging.require_wandb", return_value=sdk):
            random.seed(17)
            np.random.seed(18)
            torch.manual_seed(19)
            py_rng, np_rng, th_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
            logger = ExperimentLogger(args(), directory, {}, "train")
            logger.log_training({"step": 12, "train_loss": .2, "val_loss": .3})
            logger.finish()
            self.assertEqual(random.getstate(), py_rng)
            np.testing.assert_array_equal(np.random.get_state()[1], np_rng[1])
            torch.testing.assert_close(torch.get_rng_state(), th_rng, atol=0, rtol=0)
            metadata = json.loads((Path(directory) / "wandb_run.json").read_text())
            restored = ExperimentLogger(args("online"), directory, {}, "train", resume=True)
            self.assertEqual(restored.metadata["id"], metadata["id"])
            self.assertEqual(sdk.calls[-1]["resume"], "allow")
            self.assertEqual(sdk.calls[-1]["entity"], "test-user")
            self.assertEqual(sdk.runs[0].logs[0], {"global_step": 12, "train/loss": .2, "val/loss": .3})
            restored.log_episode({"seed": 1000, "success_end": False, "video_path": "/tmp/seed_1000.mp4"},
                                 1, upload_video=True, fps=20)
            self.assertEqual(sdk.runs[-1].logs[-1]["eval/video"]["format"], "mp4")
            self.assertEqual(sdk.runs[-1].logs[-1]["episode"], 1)
            restored.finish()

    @unittest.skipUnless(importlib.util.find_spec("wandb"), "需要安装真实 W&B SDK 才能执行离线检查")
    def test_real_sdk_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = ExperimentLogger(args(), directory, {"test": "offline"}, "test")
            logger.log_training({"step": 1, "train_loss": .5, "val_loss": .6})
            logger.finish()
            self.assertTrue(list((Path(directory) / "wandb").rglob("*.wandb")))


if __name__ == "__main__":
    unittest.main()
