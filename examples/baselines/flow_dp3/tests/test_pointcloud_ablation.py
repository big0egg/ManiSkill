"""A/B/C/D semantics, original-D parity, actual learning and strict persistence."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointcloud_inputs import ABLATIONS, build_pointcloud_inputs, STORAGE_CONTRACT
from policy import FlowDP3, PolicyConfig
from dataset import DemoDataset
from obs_adapter import ObservationConfig
from runtime_utils import load_config, load_policy, pointcloud_contract, validate_checkpoint


def observations(batch=1, frames=2):
    generator = torch.Generator().manual_seed(17)
    xyz = torch.rand(batch, frames, 128, 3, generator=generator) * .2 + torch.tensor([.45, -.1, .02])
    state = torch.randn(batch, frames, 28, generator=generator) * .01
    state[..., 18:21] = torch.tensor([.6, 0., .2])
    return {"pointcloud_xyz": xyz, "state": state, "ee_position": state[..., 18:21]}


def small_config(group):
    rep, enc = ABLATIONS[group]
    return PolicyConfig(pointcloud_representation=rep, pointcloud_encoder_type=enc,
                        pointcloud_num_centers=(16, 4), pointcloud_num_neighbors=(8, 4),
                        horizon=4, n_action_steps=2, down_dims=(32, 64),
                        diffusion_step_embed_dim=16, num_inference_steps=1)


def payload(model, contract, group):
    return {"format_version": 2, "config": {"policy": asdict(model.config)},
            "contract": contract, "pointcloud_input": pointcloud_contract(model.config, contract),
            "ablation": group, "model": model.state_dict(), "ema": model.state_dict()}


class InputTests(unittest.TestCase):
    def test_xyz_ignores_tcp_state_and_poisoned_distance_cache(self):
        obs = observations(batch=2, frames=3)
        expected = obs["pointcloud_xyz"] / 2
        obs["ee_position"] = torch.full((2, 3, 3), float("nan"))
        obs["state"] = None
        obs["pointcloud_distance"] = object()
        geometry, features = build_pointcloud_inputs(obs, "xyz", 2)
        self.assertEqual(geometry.shape, (6, 128, 3))
        torch.testing.assert_close(features.reshape(2, 3, 128, 3), expected)
        torch.testing.assert_close(geometry, features)

    def test_current_tcp_fixed_scale_and_common_translation(self):
        obs = observations(batch=2, frames=3)
        g, f = build_pointcloud_inputs(obs, "ee_vector_distance", 2)
        v = obs["pointcloud_xyz"] - obs["ee_position"].unsqueeze(-2)
        torch.testing.assert_close(g.reshape_as(v), v / 2)
        torch.testing.assert_close(f[..., 3].reshape(2, 3, 128), v.norm(dim=-1) / 2)
        shift = torch.tensor([1., -2., 3.])
        shifted = {"pointcloud_xyz": obs["pointcloud_xyz"] + shift,
                   "ee_position": obs["ee_position"] + shift}
        g2, f2 = build_pointcloud_inputs(shifted, "ee_vector_distance", 2)
        torch.testing.assert_close(g2, g, atol=2e-7, rtol=1e-5)
        torch.testing.assert_close(f2, f, atol=2e-7, rtol=1e-5)

    def test_axes_time_alignment_and_padding_are_checked(self):
        xyz = torch.tensor([[[[2., 20., 200.]], [[4., 40., 400.]]]])
        g, _ = build_pointcloud_inputs({"pointcloud_xyz": xyz}, "xyz", 2)
        torch.testing.assert_close(g[:, 0], torch.tensor([[1., 10., 100.], [2., 20., 200.]]))
        with self.assertRaises(ValueError):
            build_pointcloud_inputs({"pointcloud_xyz": xyz, "ee_position": torch.zeros(1, 1, 3)}, "ee_vector_distance")
        with self.assertRaisesRegex(ValueError, "Padded"):
            build_pointcloud_inputs({"pointcloud_xyz": xyz, "valid_mask": torch.ones(1, 2, 1)}, "xyz")
        with self.assertRaisesRegex(ValueError, "独立"):
            build_pointcloud_inputs({"pointcloud_distance": torch.zeros(1, 2, 128, 4)}, "xyz")


class EncoderTests(unittest.TestCase):
    def test_padded_neighbor_slots_do_not_contribute_or_receive_gradient(self):
        from ee_relation_encoder import ball_query_indices, masked_max
        geometry = torch.tensor([[[0., 0., 0.], [10., 0., 0.]]])
        indices, mask = ball_query_indices(geometry, geometry[:, :1], .1, 4)
        self.assertEqual(mask.tolist(), [[[True, False, False, False]]])
        self.assertEqual(indices.shape, (1, 1, 4))
        features = torch.tensor([[[[1., 2.], [999., 999.], [999., 999.], [999., 999.]]]], requires_grad=True)
        pooled = masked_max(features, mask)
        torch.testing.assert_close(pooled, torch.tensor([[[1., 2.]]]))
        pooled.sum().backward()
        self.assertEqual(float(features.grad[:, :, 1:].abs().sum()), 0.)

    def test_original_d_forward_and_uniform_encode_are_identical(self):
        torch.manual_seed(2)
        model = FlowDP3(small_config("D")).eval()
        g, f = build_pointcloud_inputs(observations(), "ee_vector_distance")
        old = model.distance_encoder(f)
        new = model.distance_encoder.encode(g, f)
        torch.testing.assert_close(old, new, atol=0, rtol=0)

    def test_c_has_d_layers_and_only_uses_xyz(self):
        c, d = (FlowDP3(small_config(x)).eval().distance_encoder for x in ("C", "D"))
        self.assertEqual({k: tuple(v.shape) for k, v in c.state_dict().items()},
                         {k: tuple(v.shape) for k, v in d.state_dict().items()})
        obs = observations()
        g, f = build_pointcloud_inputs(obs, "xyz")
        expected = c.encode(g, f)
        lifted = torch.cat((f, f.norm(dim=-1, keepdim=True)), dim=-1)
        torch.testing.assert_close(expected, c(lifted), atol=0, rtol=0)
        obs["ee_position"] += 5
        obs["pointcloud_distance"] = torch.full((1, 2, 128, 4), float("nan"))
        g2, f2 = build_pointcloud_inputs(obs, "xyz")
        torch.testing.assert_close(c.encode(g2, f2), expected, atol=0, rtol=0)

    def test_pointnetpp_retains_distance_attribute_gradient(self):
        encoder = FlowDP3(small_config("B")).distance_encoder
        g, f = build_pointcloud_inputs(observations(), "ee_vector_distance")
        f = f.detach().requires_grad_()
        encoder.encode(g, f).square().mean().backward()
        self.assertGreater(float(f.grad[..., 3].abs().sum()), 0)
        self.assertFalse(hasattr(encoder, "score_a"))
        self.assertFalse(hasattr(encoder, "region_projection"))

    def test_all_groups_learn_and_predict_finite_actions(self):
        for group in ABLATIONS:
            with self.subTest(group=group):
                torch.manual_seed(5)
                model = FlowDP3(small_config(group))
                obs = observations()
                before = model.distance_encoder.sa1[0].weight.detach().clone()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                loss = model.compute_loss({"obs": obs, "action": torch.randn(1, 4, 7) * .1})
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(x).all() for x in grads))
                optimizer.step()
                self.assertFalse(torch.equal(before, model.distance_encoder.sa1[0].weight))
                model.eval()
                action = model.predict_action(obs)["action"]
                self.assertEqual(action.shape, (1, 2, 7))
                self.assertTrue(torch.isfinite(action).all())


class PersistenceTests(unittest.TestCase):
    def test_presets_are_independent_and_conflicts_rejected(self):
        path = Path(__file__).resolve().parents[1] / "configs/pickcube_smoke.yaml"
        for group, pair in ABLATIONS.items():
            policy = load_config(path, group)["policy"]
            self.assertEqual((policy["pointcloud_representation"], policy["pointcloud_encoder_type"]), pair)
        with tempfile.TemporaryDirectory() as directory:
            import yaml
            config = load_config(path, "A")
            file = Path(directory) / "a.yaml"
            file.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "冲突"):
                load_config(file, "D")

    def test_all_groups_restore_predictions_and_reject_semantic_mismatch(self):
        contract = ObservationConfig(num_points=128).contract()
        for group in ABLATIONS:
            with self.subTest(group=group), tempfile.TemporaryDirectory() as directory:
                model = FlowDP3(small_config(group)).eval()
                obs = observations()
                checkpoint = payload(model, contract, group)
                path = Path(directory) / "model.pt"
                torch.save(checkpoint, path)
                loaded, _ = load_policy(path, torch.device("cpu"))
                torch.manual_seed(8)
                expected = model.predict_action(obs)["action"]
                torch.manual_seed(8)
                torch.testing.assert_close(loaded.predict_action(obs)["action"], expected, atol=0, rtol=0)
                for field, value in (("representation", "invalid"), ("length_scale", 2),
                                     ("num_centers", [8, 2]), ("radius_m", [.1, .2]),
                                     ("readout_mode", "global_max")):
                    wrong = copy.deepcopy(checkpoint)
                    wrong["pointcloud_input"][field] = value
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        validate_checkpoint(wrong)
                wrong = copy.deepcopy(checkpoint)
                del wrong["config"]["policy"]["pointcloud_representation"]
                with self.assertRaisesRegex(ValueError, "显式"):
                    validate_checkpoint(wrong)
        with self.assertRaisesRegex(ValueError, "旧 checkpoint"):
            validate_checkpoint({"format_version": 1})

    def test_offline_and_live_inputs_match_and_cache_is_not_read(self):
        obs = observations()
        contract = ObservationConfig(num_points=128).contract()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.h5"
            with h5py.File(path, "w") as stream:
                stream.attrs["contract"] = json.dumps(contract)
                stream.attrs["manifest"] = "{}"
                stream.attrs["pointcloud_storage"] = json.dumps(STORAGE_CONTRACT)
                for index in range(3):
                    group = stream.create_group(f"episode_{index:05d}")
                    group.attrs["metadata"] = '{"success_end": true}'
                    group["pointcloud_xyz"] = obs["pointcloud_xyz"][0].numpy()
                    group["state"] = obs["state"][0].numpy()
                    group["action"] = np.zeros((1, 7), np.float32)
                    group["pointcloud_distance"] = np.full((2, 128, 4), np.nan, np.float32)
            dataset = DemoDataset(path, horizon=4, n_obs_steps=2, n_action_steps=2)
            try:
                item = next(i for i, (_, start, _) in enumerate(dataset.index) if start == 0)
                offline = {key: value[None] for key, value in dataset[item]["obs"].items()}
                for rep in ("xyz", "ee_vector_distance"):
                    expected = build_pointcloud_inputs(obs, rep, 2)
                    actual = build_pointcloud_inputs(offline, rep, 2)
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
            finally:
                dataset.close()
