"""PickCube camera/crop experiment; repository policy and task defaults stay untouched.

Screen and validate re-render identical stored simulator states. Replay executes the
recorded joint actions normally; snapshot results are not policy success claims.
Segmentation and actor poses are used for diagnostics, never for point selection.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / "examples/baselines/flow_dp3")]
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json")
os.environ.setdefault("MS_ASSET_DIR", str(ROOT / ".runtime/maniskill"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".cache/matplotlib"))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import gymnasium as gym
import h5py
import numpy as np
import torch

from ee_relation_encoder import farthest_point_indices
from obs_adapter import ObservationConfig, pointcloud_features

CROP_MIN = (0.44, -0.23, -0.03)
CROP_MAX = (0.79, 0.25, 0.52)
WRIST_LINKS = ("panda_link7", "panda_link8", "panda_hand", "panda_leftfinger",
               "panda_rightfinger", "panda_leftfinger_pad", "panda_rightfinger_pad")


@dataclass(frozen=True)
class Camera:
    name: str
    eye: tuple
    target: tuple
    resolution: int = 256
    fov_degrees: float = 60


def cameras():
    result = [Camera(f"legacy{r}_90", (.3, 0., .6), (-.1, 0., .1), r, 90)
              for r in (128, 256, 384)]
    result += [Camera(f"front256_{f}", (.3, 0., .55), (0., 0., .15), 256, f)
               for f in (60, 75)]
    for side, sign in (("left", -1), ("right", 1)):
        result += [Camera(f"{side}256_{f}", (.3, sign * .3, .55), (0., 0., .15), 256, f)
                   for f in (60, 75)]
        result += [Camera(f"near_{side}256_{f}", (.25, sign * .2, .45), (0., 0., .15), 256, f)
                   for f in (45, 60)]
        result += [Camera(f"low_{side}256_60", (.3, sign * .3, .35), (0., 0., .12))]
        result += [Camera(f"{side}384_60", (.3, sign * .3, .55), (0., 0., .15), 384)]
    result += [Camera(f"overhead256_{f}", (.02, 0., .65), (0., 0., .10), 256, f)
               for f in (45, 60)]
    for side, sign in (("left", -1), ("right", 1)):
        result += [Camera(f"adjusted_{side}256_{f}", (.25, sign * .2, .45), (0., 0., .18), 256, f)
                   for f in (50, 55)]
        result += [Camera(f"balanced_{side}256_{f}", (.28, sign * .24, .50), (0., 0., .17), 256, f)
                   for f in (50, 55)]
        result += [Camera(f"balanced_{side}384_55", (.28, sign * .24, .50), (0., 0., .17), 384, 55)]
    return {c.name: c for c in result}


def array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def ids(actor):
    return array(actor.per_scene_id).reshape(-1).astype(int).tolist()


def make_env(camera):
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils import sapien_utils
    pose = sapien_utils.look_at(eye=camera.eye, target=camera.target)
    settings = {"pose": np.r_[array(pose.p).reshape(-1), array(pose.q).reshape(-1)].tolist(), "width": camera.resolution,
                "height": camera.resolution, "fov": math.radians(camera.fov_degrees)}
    return gym.make("PickCube-v1", robot_uids="panda", num_envs=1,
                    obs_mode="pointcloud", control_mode="pd_joint_pos",
                    sim_backend="physx_cpu", render_backend="cpu",
                    sensor_configs={"shader_pack": "default", "base_camera": settings},
                    reconfiguration_freq=1, max_episode_steps=200)


def labels_for(env):
    base = env.unwrapped
    links = {link.name: ids(link) for link in base.agent.robot.links}
    result = {"cube": ids(base.cube), "robot": sum(links.values(), []),
              "wrist": sum((links[name] for name in WRIST_LINKS), []),
              "table": ids(base.table_scene.table), "ground": ids(base.table_scene.ground)}
    return result, links


def classify(segmentation, labels):
    result = {name: int(np.isin(segmentation, values).sum()) for name, values in labels.items()}
    known = sum((labels[name] for name in ("cube", "robot", "table", "ground")), [])
    result["unknown"] = int((~np.isin(segmentation, known)).sum())
    assert sum(result[name] for name in ("cube", "robot", "table", "ground", "unknown")) == len(segmentation)
    return result


def capture(env, obs, camera, output, frame, length, source_id, seed, labels, mode):
    base = env.unwrapped
    pc = obs["pointcloud"]
    xyzw = torch.as_tensor(pc["xyzw"]).detach().cpu().float()[0]
    seg = array(pc["segmentation"])[0].reshape(-1)
    rgb = array(pc["rgb"])[0].reshape(-1, 3).astype(np.uint8)
    valid = (xyzw[:, 3] > .5) & torch.isfinite(xyzw).all(-1)
    transform = base.agent.robot.pose.inv().to_transformation_matrix().detach().cpu()[0]
    valid_ids = torch.nonzero(valid).flatten()
    xyz = xyzw[valid, :3] @ transform[:3, :3].T + transform[:3, 3]
    keep = ((xyz >= torch.tensor(CROP_MIN)) & (xyz <= torch.tensor(CROP_MAX))).all(-1)
    crop_ids = valid_ids[keep]
    cropped = xyz[keep]
    presample = torch.arange(len(cropped))
    if len(presample) > 4096:
        presample = torch.randperm(len(cropped), generator=torch.Generator().manual_seed(42))[:4096]
    pre_ids = crop_ids[presample]
    pre_xyz = cropped[presample]
    tcp_world = base.agent.tcp_pose.p.detach().cpu()[0]
    tcp_base = transform[:3, :3] @ tcp_world + transform[:3, 3]
    cube_world = array(base.cube.pose.p)[0]
    phase = "lifted" if cube_world[2] > .055 else (
        "near_contact" if np.linalg.norm(array(tcp_world) - cube_world) < .08 else "approach")
    if mode == "action_replay" and bool(array(base.evaluate()["is_grasped"])[0]) and phase != "lifted":
        phase = "grasped"
    row = {"camera": camera.name, "source_episode": source_id, "seed": seed,
           "frame": frame, "trajectory_actions": length, "phase": phase,
           "mode": mode, "crop_total": len(cropped), "presample_total": len(pre_xyz),
           "cube_world_x": float(cube_world[0]), "cube_world_y": float(cube_world[1]),
           "cube_world_z": float(cube_world[2]), "sampling_matches_adapter_max_error": 0.}
    for stage, indices in (("raw", valid_ids), ("crop", crop_ids), ("pre", pre_ids)):
        row.update({f"{stage}_{name}": count for name, count in classify(seg[indices.numpy()], labels).items()})
    group = output.create_group(f"frame_{frame:05d}")
    group.attrs["mode"] = mode
    for name, value in {"rgb": rgb.reshape(camera.resolution, camera.resolution, 3),
                        "segmentation_image": seg.reshape(camera.resolution, camera.resolution),
                        "xyz_base_crop": cropped.numpy(), "crop_source_indices": crop_ids.numpy(),
                        "raw_cube_xyz_base": xyz[np.isin(seg[valid_ids.numpy()], labels["cube"])].numpy(),
                        "tcp_base": tcp_base.numpy(), "world_to_base": transform.numpy(),
                        "cube_world_pose": array(base.cube.pose.raw_pose)[0]}.items():
        group.create_dataset(name, data=value, compression="lzf")
    if len(pre_xyz) >= 512:
        total = min(1024, len(pre_xyz))
        chosen = farthest_point_indices(pre_xyz[None], total)[0]
    else:
        chosen = torch.empty(0, dtype=torch.long)
    for n in (512, 1024):
        row[f"fps{n}_valid"] = len(chosen) >= n
        if len(chosen) < n:
            row.update({f"fps{n}_{name}": 0 for name in [*labels, "unknown"]})
            continue
        selected_ids = pre_ids[chosen[:n]]
        selected_xyz = pre_xyz[chosen[:n]]
        features_xyz = selected_xyz - tcp_base
        features = torch.cat((features_xyz, features_xyz.norm(dim=-1, keepdim=True)), -1)
        selected_seg = seg[selected_ids.numpy()]
        row.update({f"fps{n}_{name}": count for name, count in classify(selected_seg, labels).items()})
        assert len(np.unique(selected_ids.numpy())) == n
        assert row[f"fps{n}_ground"] == 0
        assert torch.isfinite(features).all()
        for name, value in {f"fps{n}_features": features.numpy(),
                            f"fps{n}_source_indices": selected_ids.numpy(),
                            f"fps{n}_segmentation": selected_seg}.items():
            group.create_dataset(name, data=value, compression="lzf")
        if frame == 0:
            config = ObservationConfig(num_points=n, crop_min=CROP_MIN, crop_max=CROP_MAX)
            expected, _ = pointcloud_features(obs, base.agent, config)
            error = float((expected[0] - features).abs().max())
            assert error < 1e-6, (camera.name, n, error)
            row["sampling_matches_adapter_max_error"] = max(row["sampling_matches_adapter_max_error"], error)
    params = obs["sensor_param"]["base_camera"]
    extrinsic = array(params["extrinsic_cv"])[0]
    intrinsic = array(params["intrinsic_cv"])[0]
    corners = np.array(list(itertools.product((-.02, .02), repeat=3)), dtype=np.float32)
    pose = array(base.cube.pose.to_transformation_matrix())[0]
    corners_world = corners @ pose[:3, :3].T + pose[:3, 3]
    camera_xyz = np.c_[corners_world, np.ones(8)] @ extrinsic.T
    image_xyz = camera_xyz @ intrinsic.T
    pixels = image_xyz[:, :2] / image_xyz[:, 2:3]
    visible = (camera_xyz[:, 2] >= .01) & ((pixels >= 0) & (pixels < camera.resolution)).all(-1)
    row["cube_vertices_in_frustum"] = int(visible.sum())
    group.create_dataset("camera_intrinsic_cv", data=intrinsic)
    group.create_dataset("camera_extrinsic_cv", data=extrinsic)
    group.attrs["statistics"] = json.dumps(row)
    return row


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {"min": float(values.min()), "p05": float(np.quantile(values, .05)),
            "median": float(np.median(values)), "max": float(values.max()),
            "zero_frames": int((values == 0).sum())}


def summarize(rows):
    count_keys = [name for name in rows[0] if name.startswith(("raw_", "crop_", "pre_", "fps512_", "fps1024_"))]
    result = {"frames": len(rows), "statistics": {k: statistics([r[k] for r in rows]) for k in count_keys},
              "frustum_clipped_frames": sum(r["cube_vertices_in_frustum"] != 8 for r in rows),
              "crop_cube_retention": sum(r["crop_cube"] for r in rows) / max(1, sum(r["raw_cube"] for r in rows)),
              "max_sampling_adapter_error": max(r["sampling_matches_adapter_max_error"] for r in rows)}
    for n in (512, 1024):
        result[f"fps{n}_sampling_lost_visible_cube_frames"] = sum(r["raw_cube"] > 0 and r[f"fps{n}_cube"] == 0 for r in rows)
        result[f"fps{n}_invalid_frames"] = sum(not r[f"fps{n}_valid"] for r in rows)
    result["by_phase"] = {phase: {"frames": len(subset), "raw_cube": statistics([r["raw_cube"] for r in subset]),
                                  "fps512_cube": statistics([r["fps512_cube"] for r in subset]),
                                  "fps1024_cube": statistics([r["fps1024_cube"] for r in subset])}
                          for phase in sorted(set(r["phase"] for r in rows))
                          if (subset := [r for r in rows if r["phase"] == phase])}
    return result


def source_state(group, index):
    from mani_skill.trajectory.utils import index_dict
    return index_dict(group["env_states"], index)


def state_error(env, expected):
    current = env.unwrapped.get_state_dict()
    return max(float(np.max(np.abs(array(current[category][name]).reshape(-1) - value.reshape(-1))))
               for category in ("actors", "articulations") for name, value in expected[category].items())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    destination = args.output.resolve()
    if not destination.is_relative_to(HERE):
        raise ValueError("All experiment outputs must be inside testpointcloud")
    destination.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(args.source.with_suffix(".json").read_text())
    by_id = {ep["episode_id"]: ep for ep in metadata["episodes"]}
    configs = cameras()
    selected = list(configs) if not args.cameras else args.cameras
    for name in selected:
        if name not in configs:
            raise ValueError(f"Unknown camera {name}; choices: {list(configs)}")
    manifest = {"source": str(args.source.resolve()), "source_sha256": sha256(args.source),
                "source_json_sha256": sha256(args.source.with_suffix(".json")),
                "env_id": "PickCube-v1", "robot": "panda", "control_mode": "pd_joint_pos",
                "mode": args.mode, "episodes": args.episodes, "frames_per_episode": args.frames,
                "crop_frame": "robot_base_before_TCP_subtraction", "crop_min": CROP_MIN,
                "crop_max": CROP_MAX, "sampling": {"counts": [512, 1024], "pre_sample_points": 4096, "seed": 42,
                "uses_segmentation": False}, "cameras": {name: asdict(configs[name]) for name in selected},
                "diagnostic_state_pose_and_segmentation": True,
                "snapshot_results_are_not_policy_success": True}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = {};all_rows = [];started = time.monotonic()
    with h5py.File(args.source, "r") as source:
        for name in selected:
            camera = configs[name];env = make_env(camera);rows = [];episodes = []
            try:
                with h5py.File(destination / f"{name}.h5", "w") as recordings:
                    recordings.attrs["configuration"] = json.dumps(manifest["cameras"][name])
                    recordings.attrs["experiment_manifest"] = json.dumps(manifest)
                    for episode_id in args.episodes:
                        meta = by_id[episode_id];group = source[f"traj_{episode_id}"]
                        reset = dict(meta["reset_kwargs"])
                        if isinstance(reset["seed"], list): reset["seed"] = reset["seed"][0]
                        env.reset(**reset);labels, links = labels_for(env)
                        saved = recordings.create_group(f"traj_{episode_id}")
                        saved.attrs["entity_ids"] = json.dumps(labels)
                        saved.attrs["robot_links"] = json.dumps(links)
                        saved.attrs["source_metadata"] = json.dumps(meta)
                        length = len(group["actions"]);errors = [];success = False
                        if args.mode == "action_replay":
                            env.unwrapped.set_state_dict(source_state(group, 0))
                            obs = env.unwrapped.get_obs()
                            frames = range(length + 1)
                        else:
                            frames = range(length + 1) if args.frames == 0 else sorted(set(
                                np.linspace(0, length, args.frames).astype(int).tolist() +
                                [int(np.argmax(group["env_states/actors/cube"][:, 2]))]))
                        for frame in frames:
                            expected = source_state(group, frame)
                            if args.mode == "snapshot":
                                env.unwrapped.set_state_dict(expected);obs = env.unwrapped.get_obs()
                            elif frame:
                                obs, _, _, _, _ = env.step(array(group["actions"][frame - 1]))
                            error = state_error(env, expected);errors.append(error)
                            if args.mode == "snapshot": assert error < 1e-5, (name, episode_id, frame, error)
                            row = capture(env, obs, camera, saved, frame, length, episode_id,
                                          reset["seed"], labels, args.mode)
                            row["source_state_max_error"] = error
                            rows.append(row)
                            if args.mode == "action_replay":
                                success = bool(array(env.unwrapped.evaluate()["success"])[0])
                        episodes.append({"source_episode": episode_id, "seed": reset["seed"],
                                         "frames": len(frames), "max_state_error": max(errors),
                                         "replay_success_end": success if args.mode == "action_replay" else None})
                        print(f"{name}: episode {episode_id}, {len(frames)} frames, state error {max(errors):.3g}", flush=True)
                result = summarize(rows);result["episodes"] = episodes;result["camera"] = asdict(camera)
                result["recording"] = str((destination / f"{name}.h5").resolve())
                results[name] = result;all_rows += rows
                (destination / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
                with (destination / "counts.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
                print(f"RESULT {name}: raw cube {result['statistics']['raw_cube']}; FPS512 cube {result['statistics']['fps512_cube']}; FPS512 wrist {result['statistics']['fps512_wrist']}", flush=True)
            finally:
                env.close()
    manifest["elapsed_seconds"] = time.monotonic() - started
    manifest["completed"] = True
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Finished: {destination}, {len(all_rows)} frame/config pairs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".runtime/flow_dp3/pickcube-100-scene-v3.raw.h5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["snapshot", "action_replay"], default="snapshot")
    parser.add_argument("--cameras", nargs="+", choices=list(cameras()))
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 2, 4, 6, 8])
    parser.add_argument("--frames", type=int, default=14, help="Snapshot frames per episode; 0 means all; replay always records all")
    args = parser.parse_args()
    if args.frames < 0: parser.error("frames must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
