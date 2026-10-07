"""Probe existing pose-policy commands on real held-out stable expert observations."""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/baselines/flow_dp3"))
from runtime_utils import load_policy, sha256
from obs_adapter import config_from_contract, clip_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--hold-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "testpointcloud"):
        raise ValueError("输出应为testpointcloud下的新文件")
    torch.set_num_threads(4)
    run = json.loads(args.training_run.read_text())
    with h5py.File(run["data"], "r") as old:
        val_sources = {json.loads(old[name].attrs["metadata"])["source_episode"] for name in run["val_episodes"]}
    policy, checkpoint = load_policy(args.checkpoint, torch.device("cpu"))
    assert run["data_sha256"] == checkpoint["data_sha256"]
    assert checkpoint["contract"]["control_mode"] == "pd_ee_delta_pose"
    rows = []
    with h5py.File(args.hold_dataset, "r") as f:
        contract = json.loads(f.attrs["contract"])
        config_from_contract(contract)
        assert contract["version"] == 5
        assert contract["pointcloud"] == checkpoint["contract"]["pointcloud"]
        assert contract["sensor_configs"] == checkpoint["contract"]["sensor_configs"]
        for name in sorted(f):
            group = f[name]
            meta = json.loads(group.attrs["metadata"])
            if meta["source_episode"] not in val_sources:
                continue
            assert group["task_metrics/success"][-1] and group["task_metrics/is_grasped"][-1]
            obs = {key: torch.from_numpy(group[key][-2:][None]) for key in ("state", "pointcloud_distance")}
            torch.manual_seed(42)
            predicted = clip_action(policy.predict_action(obs)["action"][0], checkpoint["contract"]).numpy()
            predicted[:,3:6] /= np.maximum(1, np.linalg.norm(predicted[:,3:6], axis=1, keepdims=True))
            expert = group["action"][-2:]
            rows.append({"episode": name, "source_episode": meta["source_episode"],
                         "cube_goal_error_m": float(group["task_metrics/goal_error_m"][-1]),
                         "predicted_first2_delta_pos_norm_mm": float(np.linalg.norm(predicted[:2,:3],axis=1).mean()*100),
                         "expert_last2_delta_pos_norm_mm": float(np.linalg.norm(expert[:,:3],axis=1).mean()*100),
                         "predicted_first2_scaled_rotation_norm_deg": float(np.linalg.norm(predicted[:2,3:6],axis=1).mean()*.1*180/np.pi),
                         "expert_last2_scaled_rotation_norm_deg": float(np.linalg.norm(expert[:,3:6],axis=1).mean()*.1*180/np.pi),
                         "predicted_first2_gripper_mean": float(predicted[:2,-1].mean()),
                         "first2_predictions": predicted[:2].tolist()})
    assert len(rows) == len(val_sources)
    metrics = {key: dict(zip(("min", "median", "max"), np.quantile([row[key] for row in rows], (0,.5,1)).tolist()))
               for key in rows[0] if key not in ("episode", "source_episode", "first2_predictions")}
    result = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
              "hold_dataset_sha256": sha256(args.hold_dataset), "validation_episodes": len(rows),
              "policy_seed":42,"metrics":metrics,"rows":rows,
              "note":"仅在真实稳定观测上测预测指令，没有执行这些动作；不能把指令增量当成实际物体位移或据此认定失败原因。保留物体真值只做诊断，模型输入仍为state28与512点距离云。"}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(metrics,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
