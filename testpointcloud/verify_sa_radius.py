"""Verify paired radius measurements and cached production arithmetic."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from sa_radius_experiment import (CONTEXTS, EERelationPointNetPPEncoder, HERE,
                                  R1, R2, cached_encode, digest, prepare)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(HERE):
        parser.error("Run must be under testpointcloud")
    destination = run/"verification.json"
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((run/"manifest.json").read_text())
    summary = json.loads((run/"summary.json").read_text())
    assert manifest["completed"]
    assert digest(manifest["source"]) == manifest["source_sha256"]
    assert manifest["radii1_m"] == list(R1) and manifest["radii2_m"] == list(R2)
    import sa_radius_experiment as audit
    assert digest(audit.__file__) == manifest["script_sha256"]
    assert digest(audit.BASELINE/"ee_relation_encoder.py") == manifest["production_encoder_sha256"]
    paired = {}; values = {}; row_count = 0
    with (run/"frames.csv").open() as f:
        for row in csv.DictReader(f):
            row_count += 1
            key = (row["trajectory"], row["frame"], row["context"])
            invariant = tuple(int(row[k]) for k in ("input_cube_points","cube_centers_sa1","cube_centers_sa2"))
            radii = (float(row["radius1_m"]), float(row["radius2_m"]))
            if key not in paired:
                paired[key] = (invariant,set())
            assert paired[key][0] == invariant, "Radius unexpectedly changed FPS centers"
            assert radii not in paired[key][1], "Duplicate paired radius record"
            paired[key][1].add(radii)
            for k in ("mean_neighbors_sa1","mean_neighbors_sa2"):
                assert 1 <= float(row[k]) <= 32
            for k in ("cube_points_coverage_sa1","cube_points_coverage_sa2"):
                assert 0 <= float(row[k]) <= 1
            assert float(row["cube_points_coverage_sa2"]) <= float(row["cube_points_coverage_sa1"]) + 1e-10
            name = f'{row["context"]}_r{radii[0]:.2f}_{radii[1]:.2f}'
            v = values.setdefault(name,dict(cube_fraction=[],coverage=[],n1=[],n2=[]))
            if row["cube_fraction_at_cube_centers_sa1"]:
                v["cube_fraction"].append(float(row["cube_fraction_at_cube_centers_sa1"]))
            v["coverage"].append(float(row["cube_points_coverage_sa2"]))
            v["n1"].append(float(row["mean_neighbors_sa1"]))
            v["n2"].append(float(row["mean_neighbors_sa2"]))
    assert row_count == manifest["paired_measurements"]
    assert len(paired) == manifest["scene_frames"] * len(CONTEXTS)
    for _,radii in paired.values():
        assert radii == {(a,b) for a in R1 for b in R2}
    for name,v in values.items():
        s = summary["cases"][name]["all"]
        for key,metric in (("cube_fraction","cube_fraction_at_cube_centers_sa1"),
                           ("coverage","cube_points_coverage_sa2"),
                           ("n1","mean_neighbors_sa1"),("n2","mean_neighbors_sa2")):
            assert abs(float(np.mean(v[key]))-s[metric]["mean"]) < 1e-12
    torch.set_num_threads(1)
    torch.manual_seed(7)
    forward_checks = []
    with h5py.File(manifest["source"],"r") as f:
        first = f[sorted(f)[0]];frame = first[sorted(first)[0]]
        labels = json.loads(first.attrs["entity_ids"])
        for n,c1,c2 in CONTEXTS:
            p = prepare(frame[f"fps{n}_features"][:],frame[f"fps{n}_segmentation"][:],labels,c1,c2)
            encoder = EERelationPointNetPPEncoder(radius1=.1,radius2=.2,num_centers=(c1,c2)).eval()
            for r1,r2 in ((.10,.20),(.05,.12),(.03,.06),(.15,.30)):
                encoder.radius1,encoder.radius2 = r1,r2
                with torch.no_grad():
                    direct = encoder(p["x"]);cached = cached_encode(encoder,p,r1,r2)
                error = float((direct-cached).abs().max())
                assert torch.allclose(direct,cached,atol=1e-6,rtol=1e-5)
                forward_checks.append(dict(points=n,centers=[c1,c2],radii=[r1,r2],max_error=error))
    result = dict(passed=True,rows=row_count,unique_scene_frames=manifest["scene_frames"],
                  radius_invariance_of_centers=True,csv_summary_recomputed=True,
                  source_and_production_hashes_match=True,
                  direct_production_query_comparisons=summary["direct_production_query_comparisons"],
                  cached_forward_checks=forward_checks,
                  random_weights_usage="Arithmetic verification only; no learned performance claim.")
    destination.write_text(json.dumps(result,indent=2)+"\n")
    print(f"PASS: {row_count} paired rows, all radius/center invariants, CSV summaries and {len(forward_checks)} encoder comparisons")


if __name__ == "__main__":
    main()
