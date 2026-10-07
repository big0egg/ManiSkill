"""Audit real PickCube pose-action recordings against their raw joint demonstrations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/baselines/flow_dp3"))
from dataset import DemoDataset
from obs_adapter import config_from_contract
from policy import FlowDP3, PolicyConfig
from runtime_utils import sha256
from mani_skill.agents.robots.panda.panda import Panda


def tcp_transforms(qpos):
    root = ET.parse(Panda.urdf_path).getroot()
    parents = {joint.find("child").attrib["link"]: joint for joint in root.findall("joint")}
    link, chain = "panda_hand_tcp", []
    while link in parents:
        joint = parents[link]
        chain.append(joint)
        link = joint.find("parent").attrib["link"]
    result = np.repeat(np.eye(4)[None], len(qpos), axis=0)
    for joint in reversed(chain):
        origin = joint.find("origin")
        attrs = origin.attrib if origin is not None else {}
        transform = np.eye(4)
        transform[:3, 3] = np.fromstring(attrs.get("xyz", "0 0 0"), sep=" ")
        transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(attrs.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        result = result @ transform
        if joint.attrib["type"] == "fixed":
            continue
        index = int(joint.attrib["name"][len("panda_joint"):]) - 1
        axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
        motion = np.repeat(np.eye(4)[None], len(qpos), axis=0)
        motion[:, :3, :3] = Rotation.from_rotvec(qpos[:, index, None] * axis).as_matrix()
        result = result @ motion
    return result


def quaternion_matrices(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[..., [1, 2, 3, 0]]).as_matrix()


def angle_deg(first, second):
    return np.rad2deg(Rotation.from_matrix(np.asarray(first).T @ second).magnitude())


def closing_error_deg(tcp_rotation, cube_rotation):
    closing = tcp_rotation[:2, 1]
    closing = closing / np.linalg.norm(closing)
    cube_axes = cube_rotation[:2, :2]
    cube_axes = cube_axes / np.linalg.norm(cube_axes, axis=0)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(closing @ cube_axes).max(), 0, 1))))


def summary(values):
    return dict(zip(("min", "median", "max"), np.quantile(values, (0, .5, 1)).tolist()))


def audit(dataset, output):
    output = output.resolve()
    if not output.is_relative_to(ROOT / "testpointcloud"):
        raise ValueError("核验输出必须位于 testpointcloud")
    if output.exists():
        raise FileExistsError(output)
    rows = []
    with h5py.File(dataset, "r") as prepared:
        contract = json.loads(prepared.attrs["contract"])
        manifest = json.loads(prepared.attrs["manifest"])
        config_from_contract(contract)
        assert (contract["env_id"], contract["version"], contract["control_mode"],
                contract["state_dim"], contract["action_dim"]) == (
                    "PickCube-v1", 4, "pd_ee_delta_pose", 28, 7)
        assert contract["pointcloud"]["num_points"] == 512
        source = Path(manifest["source"])
        source_hash = sha256(source)
        assert source_hash == manifest["source_sha256"]
        assert sha256(source.with_suffix(".json")) == manifest["source_json_sha256"]
        with h5py.File(source, "r") as raw:
            for name in sorted(prepared):
                group = prepared[name]
                metadata = json.loads(group.attrs["metadata"])
                raw_group = raw[f'traj_{metadata["source_episode"]}']
                state, action, points = (group[key][:] for key in ("state", "action", "pointcloud_distance"))
                steps = len(action)
                assert state.shape == (steps + 1, 28)
                assert action.shape == (steps, 7)
                assert points.shape == (steps + 1, 512, 4)
                assert all(np.isfinite(value).all() for value in (state, action, points))
                assert np.abs(action).max() <= 1.0001
                assert np.linalg.norm(action[:, 3:6], axis=1).max() <= 1.0001
                norm_error = float(np.abs(points[..., 3] - np.linalg.norm(points[..., :3], axis=-1)).max())
                assert norm_error < 1e-5
                assert metadata["success_end"] and metadata["min_cropped_points"] >= 512
                raw_state = raw_group["env_states/articulations/panda"][:]
                transforms = tcp_transforms(raw_state[:, 13:22])
                rotations = quaternion_matrices(state[:, 21:25])
                initial_position_error = float(np.linalg.norm(transforms[0, :3, 3] - state[0, 18:21]))
                initial_rotation_error = float(angle_deg(transforms[0, :3, :3], rotations[0]))
                assert initial_position_error < 1e-5 and initial_rotation_error < .01
                np.testing.assert_allclose(state[0, :9], raw_state[0, 13:22], atol=1e-6, rtol=0)
                raw_close = int(np.flatnonzero(raw_group["actions"][:, -1] < -.5)[0])
                close = int(np.flatnonzero(action[:, -1] < -.5)[0])
                cube_rotation = quaternion_matrices(raw_group["env_states/actors/cube"][0, 3:7])
                rows.append({"episode": name, "seed": metadata["seed"],
                             "source_episode": metadata["source_episode"], "steps": steps,
                             "raw_rotation_before_close_deg": float(angle_deg(transforms[0, :3, :3], transforms[raw_close, :3, :3])),
                             "converted_rotation_before_close_deg": float(angle_deg(rotations[0], rotations[close])),
                             "raw_closing_axis_error_deg": closing_error_deg(transforms[raw_close, :3, :3], cube_rotation),
                             "converted_closing_axis_error_deg": closing_error_deg(rotations[close], cube_rotation),
                             "rotation_action_fraction": float(np.mean(np.linalg.norm(action[:, 3:6], axis=1) > 1e-4)),
                             "max_distance_norm_error": norm_error,
                             "initial_fk_position_error_m": initial_position_error,
                             "initial_fk_rotation_error_deg": initial_rotation_error})
    # This checks network compatibility on real observations; it is not policy training/evaluation.
    torch.set_num_threads(1)
    torch.manual_seed(42)
    data = DemoDataset(dataset)
    try:
        sample = data[0]
        batch = {"obs": {key: value[None] for key, value in sample["obs"].items()},
                 "action": sample["action"][None]}
        config = PolicyConfig(down_dims=(64, 128, 256), num_inference_steps=1)
        model = FlowDP3(config)
        states, actions = data.normalizer_data()
        model.state_normalizer.fit(states)
        model.action_normalizer.fit(actions)
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        model.eval()
        prediction = model.predict_action(batch["obs"])["action"]
        assert prediction.shape == (1, 8, 7) and torch.isfinite(prediction).all()
    finally:
        data.close()
    result = {"time_utc": datetime.now(timezone.utc).isoformat(),
              "dataset": str(dataset.resolve()), "dataset_sha256": sha256(dataset),
              "source": str(source), "source_sha256": source_hash,
              "contract": contract, "attempted": manifest["attempted"], "saved": len(rows),
              "rejected": manifest["rejected"], "steps": sum(row["steps"] for row in rows),
              "model_loss_backward_prediction": "passed", "rows": rows,
              "metrics": {key: summary([row[key] for row in rows]) for key in rows[0]
                          if key not in ("episode", "seed", "source_episode", "steps")}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("attempted", "saved", "steps", "metrics")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.dataset, args.output)
