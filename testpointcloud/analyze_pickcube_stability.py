"""Audit terminal sampling and physically recorded PickCube holding phases."""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/baselines/flow_dp3"))
from dataset import DemoDataset
from obs_adapter import config_from_contract, action_bounds
from runtime_utils import sha256


def quantiles(values):
    return dict(zip(("min", "median", "max"), np.quantile(values, (0, .5, 1)).tolist()))


def inspect(path):
    data = DemoDataset(path)
    rows, coverage = [], []
    try:
        with h5py.File(path, "r") as f:
            contract = json.loads(f.attrs["contract"])
            manifest = json.loads(f.attrs["manifest"])
            config_from_contract(contract)
            low, high = map(np.asarray, action_bounds(contract))
            source = Path(manifest["source"])
            assert sha256(source) == manifest["source_sha256"]
            assert sha256(source.with_suffix(".json")) == manifest["source_json_sha256"]
            for name in sorted(f):
                g = f[name]
                meta = json.loads(g.attrs["metadata"])
                s, a, p = (g[key][:] for key in ("state", "action", "pointcloud_distance"))
                t = len(a)
                assert s.shape == (t + 1, 28) and a.shape == (t, contract["action_dim"])
                assert p.shape == (t + 1, 512, 4)
                assert all(np.isfinite(x).all() for x in (s, a, p))
                assert np.all(a >= low - 1e-4) and np.all(a <= high + 1e-4)
                assert np.abs(p[..., 3] - np.linalg.norm(p[..., :3], axis=-1)).max() < 1e-5
                assert np.abs(s[:,25:28] - s[0,25:28]).max() < 1e-6
                close = int(np.flatnonzero(a[:, -1] < -.5)[0])
                proxy = ((np.linalg.norm(s[:,18:21] - s[:,25:28], axis=1) <= .025) &
                         (np.abs(s[:,9:16]).max(axis=1) <= .2))
                proxy[:close] = False
                currents = [start + data.n_obs_steps - 1 for n, start, _ in data.index if n == name]
                if name in data.episodes:
                    coverage.append({"episode": name, "windows": len(currents),
                                     "proxy_near_static_current_windows": int(proxy[currents].sum()),
                                     "actual_success_current_windows": int(g["task_metrics/success"][:][currents].sum()) if "task_metrics" in g else None})
                row = {"episode": name, "source_episode": meta["source_episode"], "steps": t,
                       "hold_steps": meta.get("hold_steps", 0), "final_tcp_goal_error_m": float(np.linalg.norm(s[-1,18:21]-s[-1,25:28])),
                       "final_arm_qvel_maxabs": float(np.abs(s[-1,9:16]).max()),
                       "proxy_near_static_frames": int(proxy.sum()), "success_end": meta["success_end"]}
                assert row["success_end"]
                if contract["version"] in (5, 6):
                    h = meta["motion_steps"] + 1
                    metrics = g["task_metrics"]
                    assert metrics["success"].shape == (t + 1,)
                    assert meta["hold_steps"] == manifest["hold_steps"] == 40
                    assert len(metrics["success"][h:]) == 40
                    assert metrics["success"][h:].all() and metrics["is_grasped"][h:].all()
                    row.update({"hold_goal_error_max_m": float(metrics["goal_error_m"][h:].max()),
                                "hold_arm_qvel_maxabs": float(metrics["arm_qvel_maxabs"][h:].max()),
                                "final_goal_error_m": float(metrics["goal_error_m"][-1]),
                                "hold_all_success": True})
                    if name in data.episodes:
                        index = next(i for i, entry in enumerate(data.index) if entry[0] == name and entry[1] == t-1)
                        sample = data[index]
                        np.testing.assert_array_equal(sample["obs"]["state"][-1].numpy(), s[-1])
                        future = sample["action"][1:]
                        if contract["version"] == 5:
                            assert torch.count_nonzero(future[:, :6]).item() == 0
                            assert torch.all(future[:, -1] == a[-1,-1])
                        else:
                            np.testing.assert_array_equal(future.numpy(), np.repeat(a[-1:,:], len(future), 0))
                rows.append(row)
        return {"dataset": str(path.resolve()), "sha256": sha256(path), "contract": contract,
                "attempted": manifest["attempted"], "saved": len(rows), "rejected": manifest["rejected"],
                "actions": sum(x["steps"] for x in rows), "train_episodes": len(data.episodes),
                "train_windows": len(data), "train_episodes_without_proxy_near_static_condition": sum(x["proxy_near_static_current_windows"] == 0 for x in coverage),
                "train_proxy_near_static_current_windows": sum(x["proxy_near_static_current_windows"] for x in coverage),
                "train_actual_success_current_windows": sum(x["actual_success_current_windows"] for x in coverage) if contract["version"] in (5,6) else None,
                "metrics": {key: quantiles([x[key] for x in rows]) for key in ("steps", "final_tcp_goal_error_m", "final_arm_qvel_maxabs")},
                "holding_metrics": {key: quantiles([x[key] for x in rows]) for key in ("hold_goal_error_max_m", "hold_arm_qvel_maxabs", "final_goal_error_m")} if contract["version"] in (5,6) else None,
                "rows": rows, "train_coverage": coverage, "verification": "passed"}
    finally:
        data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / "testpointcloud"):
        raise ValueError("报告必须保存在 testpointcloud")
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [inspect(path) for path in args.datasets]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"datasets": records}, ensure_ascii=False, indent=2)+"\n")
    for record in records:
        print(json.dumps({k: v for k,v in record.items() if k not in ("rows", "train_coverage", "contract")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
