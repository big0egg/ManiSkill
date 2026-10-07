"""Paired SA1/SA2 radius audit on recorded PickCube clouds, without rendering.

Production modules are imported read-only. Actor IDs label the audit; they never
participate in FPS or ball query. Optional checkpoint results use fixed trained
EMA weights and measure sensitivity, not task success or a retrained policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASELINE = ROOT / "examples/baselines/flow_dp3"
sys.path.insert(0, str(BASELINE))
from ee_relation_encoder import (EERelationPointNetPPEncoder, ball_query_indices,
                                 farthest_point_indices, gather_points)

R1 = (.03, .04, .05, .07, .10, .15)
R2 = (.06, .08, .10, .12, .16, .20, .30)
CONTEXTS = ((512, 128, 32), (1024, 128, 32), (1024, 256, 64))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"count": 0, "mean": None, "min": None, "p05": None,
                "median": None, "p95": None, "max": None}
    return {"count": len(a), "mean": float(a.mean()), "min": float(a.min()),
            "p05": float(np.quantile(a, .05)), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max())}


def context_name(n, c1, c2):
    return f"fps{n}_sa{c1}_{c2}"


def case_name(context, r1, r2):
    return f"{context}_r{r1:.2f}_{r2:.2f}"


def prepare(features, segmentation, labels, c1, c2):
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))[None]
    if not torch.isfinite(x).all() or x.shape[-1] != 4:
        raise ValueError("Malformed point features")
    if float((x[..., 3] - x[..., :3].norm(dim=-1)).abs().max()) > 1e-5:
        raise ValueError("Distance channel does not equal displacement norm")
    seg = np.asarray(segmentation).reshape(-1)
    cube = np.isin(seg, labels["cube"])
    robot = np.isin(seg, labels["robot"])
    table = np.isin(seg, labels["table"])
    if not cube.any() or not np.all(cube | robot | table):
        raise ValueError("Missing cube or unexpected entity in operation crop")
    u = x[..., :3]
    i1 = farthest_point_indices(u, c1)
    u1 = gather_points(u, i1)
    i2 = farthest_point_indices(u1, c2)
    u2 = gather_points(u1, i2)
    j1, m1 = ball_query_indices(u, u1, max(R1), 32)
    j2, m2 = ball_query_indices(u1, u2, max(R2), 32)
    d1 = ((gather_points(u, j1) - u1.unsqueeze(2)) ** 2).sum(-1)
    d2 = ((gather_points(u1, j2) - u2.unsqueeze(2)) ** 2).sum(-1)
    return dict(x=x, u=u, u1=u1, u2=u2, i1=i1, i2=i2, j1=j1, j2=j2,
                max_m1=m1, max_m2=m2, base_m1=m1 & (d1 <= .10**2),
                base_m2=m2 & (d2 <= .20**2), d1=d1, d2=d2, cube=cube,
                robot=robot, table=table)


def query(p, r1, r2):
    # All candidates are <= the cached maximum radius. Preserve the maximum
    # mask: padded safe index zero must never become an actual neighbor.
    m1 = p["max_m1"] & (p["d1"] <= r1 ** 2)
    m2 = p["max_m2"] & (p["d2"] <= r2 ** 2)
    return p["j1"].masked_fill(~m1, 0), m1, p["j2"].masked_fill(~m2, 0), m2


def verify_queries(p):
    for r1, r2 in ((.10, .20), (.05, .12), (.03, .06), (.15, .30)):
        j1, m1, j2, m2 = query(p, r1, r2)
        a1, b1 = ball_query_indices(p["u"], p["u1"], r1, 32)
        a2, b2 = ball_query_indices(p["u1"], p["u2"], r2, 32)
        for a, b in ((j1, a1), (m1, b1), (j2, a2), (m2, b2)):
            if not torch.equal(a, b):
                raise AssertionError("Cached query disagrees with production query")


def measure(p, r1, r2):
    j1, m1, j2, m2 = [v[0].numpy() for v in query(p, r1, r2)]
    i1 = p["i1"][0].numpy()
    i2 = p["i2"][0].numpy()
    cube = p["cube"]
    cube1, cube2 = cube[i1], cube[i1[i2]]
    n1, n2 = m1.sum(-1), m2.sum(-1)
    if not np.all(n1 >= 1) or not np.all(n2 >= 1):
        raise AssertionError("Each FPS center must include itself")
    cube_indices = np.flatnonzero(cube)
    # Each row tracks original input cube points with a path into that feature.
    reach1 = ((j1[:, :, None] == cube_indices) & m1[:, :, None]).any(1)
    reach2 = (reach1[j2] & m2[:, :, None]).any(1)
    touch1, touch2 = reach1.any(1), reach2.any(1)
    base1, base2 = p["base_m1"][0].numpy(), p["base_m2"][0].numpy()
    changed1 = (m1 != base1).any(-1)
    changed2 = (m2 != base2).any(-1)
    purity1 = (cube[j1] & m1).sum(-1) / n1
    table1 = (p["table"][j1] & m1).sum(-1) / n1
    robot1 = (p["robot"][j1] & m1).sum(-1) / n1
    purity2 = (cube1[j2] & m2).sum(-1) / n2
    touched_cube_fraction2 = (touch1[j2] & m2).sum(-1) / n2
    # These are path coverage and label composition, not learned attribution.
    return dict(input_cube_points=int(cube.sum()), cube_centers_sa1=int(cube1.sum()),
                cube_centers_sa2=int(cube2.sum()),
                mean_neighbors_sa1=float(n1.mean()), mean_neighbors_sa2=float(n2.mean()),
                sparse_le3_fraction_sa1=float((n1 <= 3).mean()),
                sparse_le3_fraction_sa2=float((n2 <= 3).mean()),
                changed_groups_sa1=float(changed1.mean()),
                changed_groups_sa2=float(changed2.mean()),
                changed_cube_center_groups_sa1=float(changed1[cube1].mean()) if cube1.any() else None,
                cube_fraction_at_cube_centers_sa1=float(purity1[cube1].mean()) if cube1.any() else None,
                table_fraction_at_cube_centers_sa1=float(table1[cube1].mean()) if cube1.any() else None,
                robot_fraction_at_cube_centers_sa1=float(robot1[cube1].mean()) if cube1.any() else None,
                cube_fraction_at_cube_receiving_groups_sa1=float(purity1[touch1].mean()) if touch1.any() else 0.,
                cube_center_fraction_at_cube_receiving_groups_sa2=float(purity2[touch2].mean()) if touch2.any() else 0.,
                cube_receiving_sa1_fraction_at_cube_receiving_groups_sa2=float(touched_cube_fraction2[touch2].mean()) if touch2.any() else 0.,
                cube_points_coverage_sa1=float(reach1.any(0).mean()),
                cube_points_coverage_sa2=float(reach2.any(0).mean()),
                cube_receiving_groups_sa1=int(touch1.sum()),
                cube_receiving_groups_sa2=int(touch2.sum()),
                zero_cube_path_sa2=int(not touch2.any()))


@torch.no_grad()
def cached_encode(encoder, p, r1, r2):
    j1, m1, j2, m2 = query(p, r1, r2)
    x, u, u1, u2 = p["x"], p["u"], p["u1"], p["u2"]
    delta1 = gather_points(u, j1) - u1.unsqueeze(2)
    anchor1 = torch.cat((u1, u1.norm(dim=-1, keepdim=True)), -1)
    g1 = torch.cat((delta1, gather_points(x[..., 3:4], j1),
                    anchor1.unsqueeze(2).expand(-1, -1, 32, -1)), -1)
    from ee_relation_encoder import masked_max
    h1 = masked_max(encoder.sa1(g1), m1)
    delta2 = gather_points(u1, j2) - u2.unsqueeze(2)
    rho2 = u2.norm(dim=-1, keepdim=True)
    anchor2 = torch.cat((u2, rho2), -1)
    g2 = torch.cat((delta2, gather_points(h1, j2),
                    anchor2.unsqueeze(2).expand(-1, -1, 32, -1)), -1)
    h2 = masked_max(encoder.sa2(g2), m2)
    f = encoder.region_projection(torch.cat((h2, u2, rho2), -1))
    scores = encoder.score_a(torch.cat((h2, u2), -1)) + encoder.score_b(rho2)
    alpha = torch.softmax(scores.float(), 1)
    z = torch.cat((f.max(1).values, (alpha * f.float()).sum(1).to(f.dtype)), -1)
    return encoder.output_projection(z)


def trained_probe(checkpoint, source, output, frame_count):
    from runtime_utils import load_policy
    policy, saved = load_policy(checkpoint, torch.device("cpu"))
    contract = saved["contract"]
    if contract["env_id"] != "PickCube-v1" or contract["pointcloud"]["length_scale"] != 1:
        raise ValueError("Probe needs PickCube and metre-valued features")
    from mani_skill.utils.task_pointcloud import pointcloud_sensor_configs
    if contract.get("sensor_configs") != pointcloud_sensor_configs():
        raise ValueError("Checkpoint camera does not match recorded camera-v5 clouds")
    if tuple(contract["pointcloud"]["crop_min"]) != (.44, -.23, -.03) or tuple(contract["pointcloud"]["crop_max"]) != (.79, .25, .52):
        raise ValueError("Checkpoint crop does not match recorded clouds")
    r0 = (policy.config.radius1_m, policy.config.radius2_m)
    if r0 != (.10, .20):
        raise ValueError("Current probe reference expects trained radii .10/.20")
    radii = ((.10, .20), (.05, .20), (.10, .12), (.05, .12), (.03, .06))
    records = []
    with h5py.File(source, "r") as f:
        available = [(tn, fn) for tn in sorted(f) for fn in sorted(f[tn])]
        for index in np.unique(np.linspace(0, len(available)-1, min(frame_count, len(available)), dtype=int)):
            tn, fn = available[index]
            g = f[tn][fn]
            labels = json.loads(f[tn].attrs["entity_ids"])
            for n, c1, c2 in CONTEXTS:
                p = prepare(g[f"fps{n}_features"][:], g[f"fps{n}_segmentation"][:], labels, c1, c2)
                encoder = policy.distance_encoder
                encoder.num_centers = (c1, c2)
                encoder.radius1, encoder.radius2 = r0
                reference = cached_encode(encoder, p, *r0)
                direct = encoder(p["x"])
                error = float((reference-direct).abs().max())
                if not torch.allclose(reference, direct, atol=1e-6, rtol=1e-5):
                    raise AssertionError("Cached encoder disagrees with production forward")
                for r1, r2 in radii:
                    value = cached_encode(encoder, p, r1, r2)
                    relative_l2 = float((value-reference).norm()/reference.norm().clamp_min(1e-12))
                    cosine = float(torch.nn.functional.cosine_similarity(value, reference).item())
                    if not math.isfinite(relative_l2 + cosine):
                        raise ValueError("Nonfinite trained feature")
                    records.append(dict(trajectory=tn, frame=fn, context=context_name(n,c1,c2),
                                        radius1=r1, radius2=r2, relative_l2=relative_l2,
                                        cosine=cosine, cached_forward_error=error))
    with (output/"trained_features.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    summary = {}
    for row in records:
        key = case_name(row["context"],row["radius1"],row["radius2"])
        summary.setdefault(key, []).append(row)
    summary = {k: {m: stats([r[m] for r in rows]) for m in ("relative_l2","cosine","cached_forward_error")} for k,rows in summary.items()}
    result = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                  weights="ema", checkpoint_step=saved["step"], measurements=summary,
                  limitation="Same weights, changed inference geometry; not a success-rate test or radius retraining.")
    (output/"trained_features.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def plots(cases, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    specifications = (("mean_neighbors_sa1","SA1 valid neighbors",False),
                      ("mean_neighbors_sa2","SA2 valid neighbors",False),
                      ("cube_fraction_at_cube_centers_sa1","Cube fraction at SA1 cube centers",True),
                      ("changed_cube_center_groups_sa1","SA1 cube groups changed vs .10",True),
                      ("cube_points_coverage_sa2","Input cube points reaching SA2",True),
                      ("cube_receiving_groups_sa2","SA2 groups receiving cube paths",False))
    for n,c1,c2 in CONTEXTS:
        context = context_name(n,c1,c2)
        fig,axes = plt.subplots(2,3,figsize=(15,8),layout="constrained")
        for ax,(metric,title,fraction) in zip(axes.flat,specifications):
            grid = np.array([[cases[case_name(context,a,b)]["all"][metric]["mean"] for a in R1] for b in R2])
            shown = grid*100 if fraction else grid
            im = ax.imshow(shown,origin="lower",aspect="auto",cmap="viridis")
            ax.set_xticks(range(len(R1)),[f"{v:.2f}" for v in R1]);ax.set_yticks(range(len(R2)),[f"{v:.2f}" for v in R2])
            ax.set_xlabel("SA1 radius (m)");ax.set_ylabel("SA2 radius (m)")
            ax.set_title(title+(" (%)" if fraction else ""))
            for iy in range(len(R2)):
                for ix in range(len(R1)):
                    ax.text(ix,iy,f"{shown[iy,ix]:.1f}",ha="center",va="center",fontsize=7,
                            color="white" if shown[iy,ix] < (shown.min()+shown.max())/2 else "black")
            fig.colorbar(im,ax=ax,shrink=.8)
        frames = cases[case_name(context, .10, .20)]["all"]["frames"]
        fig.suptitle(f"{n} input points; {c1}/{c2} centers; {frames}-frame expert geometry audit")
        fig.savefig(output/f"{context}-radius-grid.png",dpi=160);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=HERE/"runs/validation01/low_left256_60.h5")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--frames-per-episode",type=int,default=0,help="0 = all frames")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--feature-frames",type=int,default=42)
    parser.add_argument("--no-plots",action="store_true")
    args = parser.parse_args()
    if args.frames_per_episode < 0 or args.feature_frames < 1:
        parser.error("Invalid frame count")
    output = args.output.resolve()
    if not output.is_relative_to(HERE) or output == HERE:
        parser.error("Output must be a new directory under testpointcloud")
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    start = time.monotonic()
    manifest = dict(completed=False,source=str(args.source.resolve()),source_sha256=digest(args.source),
                    production_encoder_sha256=digest(BASELINE/"ee_relation_encoder.py"),
                    script_sha256=digest(__file__),torch_version=torch.__version__,numpy_version=np.__version__,
                    radii1_m=R1,radii2_m=R2,contexts=CONTEXTS,neighbors=[32,32],length_scale=1.,
                    frames_per_episode=args.frames_per_episode,
                    camera="low_left256_60; world eye=(.30,-.30,.35), target=(0,0,.12), FOV60, 256x256",
                    crop_min=[.44,-.23,-.03],crop_max=[.79,.25,.52],
                    segmentation_usage="Labels audit only; not fed to FPS, ball query or encoder.",
                    scope="Expert snapshot geometry; not learned attribution or task success.")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    collected = {}; effective = {}; validation_queries = 0; frame_count = 0
    with h5py.File(args.source,"r") as source, (output/"frames.csv").open("w",newline="") as stream:
        writer = None
        for tn in sorted(source):
            group = source[tn];labels = json.loads(group.attrs["entity_ids"])
            names = sorted(group)
            indices = (range(len(names)) if not args.frames_per_episode else
                       np.unique(np.linspace(0,len(names)-1,min(args.frames_per_episode,len(names)),dtype=int)))
            for frame_index in indices:
                frame = group[names[frame_index]]
                phase = json.loads(frame.attrs["statistics"])["phase"]
                for n,c1,c2 in CONTEXTS:
                    context = context_name(n,c1,c2)
                    p = prepare(frame[f"fps{n}_features"][:],frame[f"fps{n}_segmentation"][:],labels,c1,c2)
                    if frame_index == 0:
                        verify_queries(p);validation_queries += 8
                    i1 = p["i1"][0].numpy();cube1 = p["cube"][i1]
                    sat1 = p["base_m1"][0].numpy().all(-1)
                    sat2 = p["base_m2"][0].numpy().all(-1)
                    e = effective.setdefault(context,dict(sa1_all=[],sa1_cube_centers=[],sa2_all=[]))
                    distance1 = p["d1"][0,:, -1].sqrt().numpy()
                    distance2 = p["d2"][0,:, -1].sqrt().numpy()
                    e["sa1_all"].extend(distance1[sat1].tolist())
                    e["sa1_cube_centers"].extend(distance1[sat1 & cube1].tolist())
                    e["sa2_all"].extend(distance2[sat2].tolist())
                    for r1 in R1:
                        for r2 in R2:
                            row = dict(trajectory=tn,frame=names[frame_index],phase=phase,context=context,radius1_m=r1,radius2_m=r2,
                                       **measure(p,r1,r2))
                            if writer is None:
                                writer = csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                            writer.writerow(row)
                            collected.setdefault(case_name(context,r1,r2),[]).append(row)
                frame_count += 1
            print(f"Completed {tn}: {frame_count} paired scene frames",flush=True)
    cases = {}
    for key,rows in collected.items():
        base = rows[0];case = dict(context=base["context"],radius1_m=base["radius1_m"],radius2_m=base["radius2_m"])
        for phase in ("all","approach","near_contact","lifted"):
            chosen = rows if phase == "all" else [r for r in rows if r["phase"] == phase]
            if not chosen:continue
            case[phase] = {metric:stats([r[metric] for r in chosen]) for metric in list(base)[6:]}
            case[phase]["frames"] = len(chosen)
            case[phase]["zero_cube_path_frames_sa2"] = sum(r["zero_cube_path_sa2"] for r in chosen)
        cases[key] = case
    summary = dict(scene_frames=frame_count,cases=cases,
                   effective_32nd_neighbor_distance_m={k:{m:stats(v) for m,v in e.items()} for k,e in effective.items()},
                   direct_production_query_comparisons=validation_queries,
                   trained_features=None)
    if args.checkpoint:
        summary["trained_features"] = trained_probe(args.checkpoint,args.source,output,args.feature_frames)
    if not args.no_plots:plots(cases,output)
    manifest.update(completed=True,scene_frames=frame_count,paired_measurements=frame_count*len(CONTEXTS)*len(R1)*len(R2),
                    elapsed_seconds=time.monotonic()-start,trained_checkpoint=str(args.checkpoint) if args.checkpoint else None)
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:manifest[k] for k in ("completed","scene_frames","paired_measurements","elapsed_seconds")}))


if __name__ == "__main__":
    main()
