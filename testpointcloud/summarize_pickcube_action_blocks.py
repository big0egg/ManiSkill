"""Summarize paired real-policy traces; export a standalone diagnostic figure."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def classify(row):
    if row["success_end"]:
        return "final_success"
    if not row["grasped_once"]:
        return "never_grasped"
    if not row["grasped_end"]:
        return "grasped_then_lost"
    if row["final_goal_error_m"] > .025:
        return "held_end_but_off_goal"
    return "held_end_near_goal_but_moving"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block8", type=Path)
    parser.add_argument("block2", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if not args.output.resolve().is_relative_to(root) or args.output.exists():
        raise ValueError("使用testpointcloud下的新报告路径")
    reports = [json.loads(p.read_text()) for p in (args.block8, args.block2)]
    traces = [json.loads(p.with_name(p.stem + "-trace.json").read_text()) for p in (args.block8, args.block2)]
    for key in ("checkpoint_sha256", "policy_seed", "contract", "max_steps", "weights"):
        assert reports[0][key] == reports[1][key]
    assert [r["n_action_steps"] for r in reports] == [8,2]
    assert all(r["reset_policy_seed_per_episode"] for r in reports)
    seeds = [x["seed"] for x in reports[0]["episodes"]]
    assert seeds == [x["seed"] for x in reports[1]["episodes"]]
    records = []
    for report in reports:
        episodes = report["episodes"]
        records.append({"n_action_steps": report["n_action_steps"], "episodes": len(episodes),
                        "success_once_count": sum(x["success_once"] for x in episodes),
                        "success_end_count": sum(x["success_end"] for x in episodes),
                        "last40_all_success_count": sum(x["last40_success_fraction"] == 1 for x in episodes),
                        "grasped_once_count": sum(x["grasped_once"] for x in episodes),
                        "grasped_end_count": sum(x["grasped_end"] for x in episodes),
                        "final_goal_error_median_m": float(np.median([x["final_goal_error_m"] for x in episodes])),
                        "mean_episode_inference_seconds": float(np.mean([x["mean_inference_seconds"]*x["inferences"] for x in episodes])),
                        "final_classification": dict(Counter(classify(x) for x in episodes))})
    paired = [{"seed": a["seed"], "block8_success": a["success_end"], "block2_success": b["success_end"],
               "block8_class": classify(a), "block2_class": classify(b),
               "block8_goal_error_m": a["final_goal_error_m"], "block2_goal_error_m": b["final_goal_error_m"]}
              for a,b in zip(reports[0]["episodes"], reports[1]["episodes"])]
    result = {"checkpoint_sha256": reports[0]["checkpoint_sha256"], "checkpoint_step": reports[0]["checkpoint_step"],
              "device": reports[0]["device"], "policy_seed": reports[0]["policy_seed"], "scene_seeds": seeds,
              "note": "同一权重和场景种子，每局重置策略随机流。不同动作块改变后续预测次数，随机采样轨迹不会逐步相同。单个策略随机种子的20局不能证明普遍优劣。",
              "summary": records, "pairs": paired}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(records, ensure_ascii=False, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    # Fixed seed selected before full results: baseline has reached the goal and drifted away.
    for trace, label in zip(traces, ("8-step block", "2-step block")):
        rows = next(x["steps"] for x in trace["episodes"] if x["seed"] == 1016)
        time = np.asarray([x["step"] for x in rows]) / 20
        axes[0].plot(time, [x["goal_error_m"]*100 for x in rows], label=label)
        axes[1].plot(time, [x["arm_qvel_maxabs"] for x in rows], label=label)
    axes[0].axhline(2.5, color="black", ls="--", label="2.5 cm tolerance")
    axes[1].axhline(.2, color="black", ls="--", label="0.2 rad/s threshold")
    axes[0].set(title="Seed 1016: cube-to-goal error", xlabel="Simulation time (s)", ylabel="Error (cm)")
    axes[1].set(title="Seed 1016: arm motion", xlabel="Simulation time (s)", ylabel="Max |joint velocity| (rad/s)")
    x = np.asarray([p["block8_goal_error_m"] for p in paired])*100
    y = np.asarray([p["block2_goal_error_m"] for p in paired])*100
    axes[2].scatter(x,y)
    high = max(x.max(), y.max(), 2.5)*1.05
    axes[2].plot([0,high], [0,high], "k--", alpha=.5)
    axes[2].set(title="Paired final errors (20 scenes)", xlabel="8-step error (cm)", ylabel="2-step error (cm)", xlim=(0,high), ylim=(0,high))
    for ax in axes[:2]:
        ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    figure.savefig(args.output.with_suffix(".png"), dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
