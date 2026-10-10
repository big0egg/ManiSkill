"""Replay prepared rigid-task actions and audit real observations and FPS coverage.

Segmentation is used for this diagnostic only; it never selects policy points.
Run from the project root with the existing software Vulkan environment.
"""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/baselines/flow_dp3"))
from dataset import DemoDataset
from ee_relation_encoder import farthest_point_indices
from obs_adapter import adapt_observation, config_from_contract, make_env
from task_metrics import demonstration_success, task_metrics
from mani_skill.trajectory import utils as trajectory_utils

TASK_OBJECTS = {"LiftPegUpright-v1": ("peg",), "PlaceSphere-v1": ("obj", "bin"),
                "PullCube-v1": ("obj", "goal_region")}


def coverage(obs, env, config):
    pc = obs["pointcloud"]
    xyzw = torch.as_tensor(pc["xyzw"]).cpu().float()[0]
    seg = torch.as_tensor(pc["segmentation"]).cpu().numpy()[0].reshape(-1)
    valid = (xyzw[:, 3] > .5) & torch.isfinite(xyzw).all(-1)
    transform = env.agent.robot.pose.inv().to_transformation_matrix().cpu()[0]
    points = xyzw[valid, :3] @ transform[:3, :3].T + transform[:3, 3]
    keep = ((points >= torch.tensor(config.crop_min)) & (points <= torch.tensor(config.crop_max))).all(-1)
    ids = torch.nonzero(valid).flatten()[keep]
    points = points[keep]
    if len(points) > config.pre_sample_points:
        indices = torch.randperm(len(points), generator=torch.Generator().manual_seed(config.sampling_seed))[:config.pre_sample_points]
        points, ids = points[indices], ids[indices]
    fps = farthest_point_indices(points[None], config.num_points)[0]
    sampled_seg = seg[ids[fps].numpy()]
    return {name: {"raw": int(np.isin(seg[valid.numpy()], getattr(env, name).per_scene_id.cpu().numpy()).sum()),
                   "fps": int(np.isin(sampled_seg, getattr(env, name).per_scene_id.cpu().numpy()).sum())}
            for name in TASK_OBJECTS[env.spec.id]}


def audit(path, output):
    with h5py.File(path, "r") as f:
        contract = json.loads(f.attrs["contract"])
        config = config_from_contract(contract)
        manifest = json.loads(f.attrs["manifest"])
        report = {"dataset": str(path.resolve()), "contract": contract,
                  "episodes": len(f), "actions": 0, "observations": 0}
        for g in f.values():
            t = len(g["action"])
            assert g["state"].shape == (t+1, contract["state_dim"])
            assert g["pointcloud_distance"].shape == (t+1, 512, 4)
            meta = json.loads(g.attrs["metadata"])
            assert meta["success_end"] and meta["hold_stable"] and meta["hold_steps"] == 20
            assert all(np.isfinite(g[key][:]).all() for key in ("state", "action", "pointcloud_distance"))
            report["actions"] += t
            report["observations"] += t+1
        train, val = DemoDataset(path), DemoDataset(path, split="val")
        try:
            assert not set(train.episodes) & set(val.episodes)
            report.update(train_windows=len(train), val_windows=len(val))
        finally:
            train.close()
            val.close()
        g = f["episode_00000"]
        meta = json.loads(g.attrs["metadata"])
        source = Path(manifest["source"])
        source_meta = json.loads(source.with_suffix(".json").read_text())
        episode = next(e for e in source_meta["episodes"] if e["episode_id"] == meta["source_episode"])
        env = make_env(contract=contract, render_mode="rgb_array")
        try:
            reset = dict(episode["reset_kwargs"])
            if isinstance(reset.get("seed"), list):
                reset["seed"] = reset["seed"][0]
            env.reset(**reset)
            with h5py.File(source, "r") as raw:
                first = trajectory_utils.dict_to_list_of_dicts(raw[f"traj_{episode['episode_id']}"]["env_states"])[0]
                env.unwrapped.set_state_dict(first)
            obs = env.unwrapped.get_obs()
            errors = {"state": 0., "pointcloud_distance": 0.}
            counts = []
            actions = g["action"][:]
            for frame in range(len(actions)+1):
                adapted, _ = adapt_observation(obs, env.unwrapped.agent, config, contract=contract)
                for key in errors:
                    errors[key] = max(errors[key], float(np.abs(adapted[key][0].numpy() - g[key][frame]).max()))
                if frame % 10 == 0 or frame == len(actions) or contract["env_id"] == "PullCube-v1":
                    counts.append({"frame": frame, **coverage(obs, env.unwrapped, config)})
                if frame in (0, len(actions)):
                    rgb = torch.as_tensor(env.render()).cpu().numpy()
                    if rgb.ndim == 4:
                        rgb = rgb[0]
                    if np.issubdtype(rgb.dtype, np.floating):
                        rgb = np.clip(rgb * 255, 0, 255)
                    Image.fromarray(rgb.astype(np.uint8)).save(output / f"frame-{frame:04d}.png")
                if frame < len(actions):
                    obs = env.step(actions[frame])[0]
            assert all(x < 1e-5 for x in errors.values()), errors
            final = task_metrics(env)
            assert demonstration_success(contract["env_id"], final), final
            report.update(replayed_frames=len(actions)+1, max_replay_errors=errors,
                          final_task_metrics=final, coverage=counts, passed=True)
        finally:
            env.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="directory containing <env-id>/demos.h5")
    parser.add_argument("--pull-dataset", type=Path, help="optional PullCube dataset override")
    parser.add_argument("--output", type=Path, help="default <root>/observation-audit.json")
    parser.add_argument("--audit-directory", default="audit", help="new image subdirectory inside each task directory")
    args = parser.parse_args()
    destination = args.output or args.root / "observation-audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    reports = {}
    for task in TASK_OBJECTS:
        output = args.root / task / args.audit_directory
        output.mkdir(parents=True, exist_ok=False)
        dataset = args.pull_dataset if task == "PullCube-v1" and args.pull_dataset else args.root/task/"demos.h5"
        reports[task] = audit(dataset, output)
        print(task, "passed", reports[task]["max_replay_errors"], flush=True)
    destination.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
