"""Decode all expert MP4s, check paired trajectories and compare existing references."""
from pathlib import Path
import argparse
import json
import sys

import h5py
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/baselines/flow_dp3"))
sys.path.insert(0, str(ROOT / "visual"))
from control_modes import DUAL_LAYOUT
from dataset_io import load_episode, episode_names


def validate(args):
    result = {"dataset": str(args.dataset.resolve()), "branches": {}}
    with h5py.File(args.dataset, "r") as stream:
        assert stream.attrs["layout"] == DUAL_LAYOUT
        manifest = json.loads(stream.attrs["manifest"])
        result["paired_source_episodes"] = manifest["paired_source_episodes"]
        result["video_every"] = manifest.get("video_every", 1)
        for mode, dimension in (("ee", 7), ("joint", 8)):
            entries = []
            reference = getattr(args, "reference_" + mode)
            references = {}
            if reference:
                for name in episode_names(reference):
                    ep = load_episode(reference, name)
                    references[ep.metadata["source_episode"]] = ep
            video_paths = []
            for index, name in enumerate(episode_names(args.dataset, mode)):
                ep = load_episode(args.dataset, name, mode)
                assert ep.actions.shape[1] == dimension
                assert ep.states.shape[1] == 28 and ep.features.shape[1:] == (512, 4)
                metrics = stream[f"{mode}/{name}/task_metrics"]
                assert np.all(metrics["success"][-ep.metadata["hold_steps"]:])
                assert ep.metadata["success_end"] and ep.metadata["hold_stable"]
                recorded = manifest.get("save_video", True) and index % result["video_every"] == 0
                assert ep.metadata.get("video_recorded", recorded) == recorded
                entry = {"episode": name, "source_episode": ep.metadata["source_episode"],
                         "steps": len(ep.actions), "video_recorded": recorded,
                         "hold_steps": ep.metadata["hold_steps"]}
                if recorded:
                    video = args.dataset.parent / ep.metadata["video_path"]
                    count, dimensions = 0, None
                    with imageio.get_reader(str(video)) as reader:
                        fps = reader.get_meta_data()["fps"]
                        for frame in reader:
                            assert frame.std() > 1, "empty/black video frame"
                            dimensions = list(frame.shape)
                            count += 1
                    assert count == len(ep.states) == ep.metadata["video_frames"]
                    assert abs(fps - ep.metadata["video_fps"]) < 1e-5
                    entry.update(video=str(video.resolve()), frames=count, fps=fps, frame_shape=dimensions)
                    video_paths.append(video.resolve())
                else:
                    assert not {"video_path", "video_frames", "video_fps"} & set(ep.metadata)
                if reference:
                    old = references[ep.metadata["source_episode"]]
                    errors = {key: float(np.max(np.abs(current - previous)))
                              for key, current, previous in (("state", ep.states, old.states),
                                                             ("pointcloud", ep.features, old.features),
                                                             ("action", ep.actions, old.actions))}
                    assert max(errors.values()) <= 1e-4, errors
                    entry["reference_max_errors"] = errors
                entries.append(entry)
            if video_paths:
                assert set(video_paths[0].parent.glob("*.mp4")) == set(video_paths)
            result["branches"][mode] = entries
        assert episode_names(args.dataset, "ee") == episode_names(args.dataset, "joint")
        for name in episode_names(args.dataset, "ee"):
            ee, joint = (load_episode(args.dataset, name, mode) for mode in ("ee", "joint"))
            assert ee.metadata["source_episode"] == joint.metadata["source_episode"]
            np.testing.assert_allclose(ee.states[0], joint.states[0], atol=1e-6, rtol=0)
            np.testing.assert_allclose(ee.features[0], joint.features[0], atol=1e-6, rtol=0)
    result["all_checks_passed"] = True
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"all_checks_passed": True, "episodes_per_branch": len(entries),
                      "decoded_videos": sum(ep["video_recorded"] for v in result["branches"].values() for ep in v)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-ee", type=Path)
    parser.add_argument("--reference-joint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    validate(args)
