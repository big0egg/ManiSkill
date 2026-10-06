"""Check camera frusta against all recorded cube poses, without scene rendering."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path

from camera_experiment import Camera, HERE, ROOT, cameras
import h5py
import numpy as np


def rotation(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q.T
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def projection(camera):
    eye = np.array(camera.eye);forward = np.array(camera.target) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1]);right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    r = np.stack((right, down, forward));extrinsic = np.c_[r, -r @ eye]
    f = camera.resolution / (2 * np.tan(np.radians(camera.fov_degrees) / 2))
    intrinsic = np.array([[f, 0, camera.resolution / 2], [0, f, camera.resolution / 2], [0, 0, 1]])
    return intrinsic, extrinsic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / '.runtime/flow_dp3/pickcube-100-scene-v3.raw.h5')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(HERE) and not args.output.exists()
    cubes = [];episodes = []
    with h5py.File(args.source, 'r') as h:
        for name in sorted(h):
            c = h[name + '/env_states/actors/cube'][:, :7]
            cubes.append(c);episodes.extend([int(name.split('_')[1])] * len(c))
    c = np.concatenate(cubes);episodes = np.array(episodes)
    corners = np.array(list(itertools.product((-.02, .02), repeat=3)))
    world = np.einsum('nij,kj->nki', rotation(c[:, 3:7]), corners) + c[:, None, :3]
    world = np.concatenate((world, np.ones((*world.shape[:2], 1))), -1)
    results = {}
    for name, camera in cameras().items():
        k, extrinsic = projection(camera)
        projected = world @ extrinsic.T;pixels = projected @ k.T
        pixels = pixels[:, :, :2] / pixels[:, :, 2:3]
        visible = (projected[:, :, 2] >= .01) & ((pixels >= 0) & (pixels < camera.resolution)).all(-1)
        clipped = ~visible.all(-1)
        results[name] = {'camera': asdict(camera), 'frames': len(c),
                         'clipped_frames': int(clipped.sum()),
                         'clipped_episodes': sorted(set(episodes[clipped].tolist())),
                         'minimum_vertices_visible': int(visible.sum(-1).min())}
        print(name, 'clipped frames',int(clipped.sum()), '/',len(c),'episodes',results[name]['clipped_episodes'])
    # Verify the analytic camera convention against a real rendered sample.
    sample = HERE / 'runs/smoke02/legacy128_90.h5'
    if sample.exists():
        with h5py.File(sample, 'r') as h:
            g = h['traj_0/frame_00000'];k, e = projection(cameras()['legacy128_90'])
            np.testing.assert_allclose(k, g['camera_intrinsic_cv'][:], atol=1e-5, rtol=0)
            np.testing.assert_allclose(e, g['camera_extrinsic_cv'][:], atol=1e-5, rtol=0)
    args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
