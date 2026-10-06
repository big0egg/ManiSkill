"""Compare runtime cameras on saved Push/Stack/Peg/Draw states, without task edits.

All native cameras are retained (Peg includes its existing hand camera). Entity IDs
label diagnostics only; point selection remains crop/random4096/global FPS512.
"""
import argparse
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from camera_experiment import (HERE, ROOT, WRIST_LINKS, array, ids, source_state,
                               state_error, statistics, sha256)
from ee_relation_encoder import farthest_point_indices
from obs_adapter import ObservationConfig, pointcloud_features
import gymnasium as gym
import h5py
import numpy as np
import torch


@dataclass(frozen=True)
class Camera:
    name: str
    eye: tuple
    target: tuple
    width: int = 256
    height: int = 256
    fov: float = math.pi / 3


TASKS = {
    'PushCube-v1': {'robot': 'panda', 'source': 'diagnostic',
                    'crop': ((.40, -.23, -.03), (1.06, .25, .52)), 'target': (.10, 0, .08),
                    'objects': ['cube', 'goal_region'], 'primary': 'cube'},
    'StackCube-v1': {'robot': 'panda', 'source': 'diagnostic',
                     'crop': ((.36, -.36, -.03), (.87, .36, .52)), 'target': (0, 0, .10),
                     'objects': ['red_cube', 'green_cube'], 'primary': 'red_cube'},
    'PegInsertionSide-v1': {'robot': 'panda_wristcam', 'source': 'diagnostic',
                           'crop': ((.34, -.40, -.03), (.90, .62, .52)), 'target': (0, .10, .12),
                           'objects': ['peg', 'hole_box'], 'primary': 'peg'},
    'DrawTriangle-v1': {'robot': 'panda_stick', 'source': 'diagnostic-base-interface',
                        'crop': ((.28, -.35, -.03), (.80, .18, .52)), 'target': (-.10, -.10, .04),
                        'objects': ['goal_triangle', 'ink'], 'primary': 'goal_triangle'},
}


def candidates(task):
    s = TASKS[task]; target = s['target']
    if task == 'DrawTriangle-v1':
        baseline = Camera('baseline', (.3, 0, .8), (0, 0, .1), 320, 240, 1.2)
        result = [baseline,
                  Camera('left256_60', (.25, -.4, .50), target),
                  Camera('right256_60', (.25, .3, .50), target),
                  Camera('overhead256_50', (-.10, -.10, .60), target, fov=math.radians(50)),
                  Camera('overhead384_50', (-.10, -.10, .60), target, 384, 384, math.radians(50)),
                  Camera('overhead_close256_50', (-.10, -.10, .45), target, fov=math.radians(50)),
                  Camera('front256_50', (.25, -.10, .55), target, fov=math.radians(50))]
    else:
        baseline = (Camera('baseline', (0, -.3, .2), (0, 0, .1), 128, 128, math.pi/2)
                    if task == 'PegInsertionSide-v1' else
                    Camera('baseline', (.3, 0, .6), (-.1, 0, .1), 128, 128, math.pi/2))
        result = [baseline,
                  Camera('low_left256_60', (.30, -.30, .35), target),
                  Camera('low_right256_60', (.30, .30, .35), target),
                  Camera('high_left256_60', (.30, -.35, .55), target),
                  Camera('high_right256_60', (.30, .35, .55), target),
                  Camera('front256_60', (.4, .05, .50), target)]
        if task == 'PushCube-v1':
            result += [Camera('wide_low_left256_75', (.30, -.30, .35), (.15, 0, .08), fov=math.radians(75)),
                       Camera('wide_low_right256_75', (.30, .30, .35), (.15, 0, .08), fov=math.radians(75))]
        if task == 'StackCube-v1':
            result += [Camera('lower_right256_65', (.30, .35, .28), (0, 0, .07), fov=math.radians(65))]
        if task == 'PegInsertionSide-v1':
            result += [Camera('wide_high_left256_75', (.30, -.35, .55), target, fov=math.radians(75)),
                       Camera('wide_high_right256_75', (.30, .35, .55), target, fov=math.radians(75)),
                       Camera('center_high256_60', (.35, 0, .70), target),
                       Camera('distant_left256_60', (.35, -.45, .70), target)]
    return {c.name: c for c in result}


def camera_pose(camera):
    from mani_skill.utils import sapien_utils
    direction = np.array(camera.target)-camera.eye
    up = (0, 1, 0) if np.linalg.norm(np.cross(direction, (0, 0, 1))) < 1e-6 else (0, 0, 1)
    pose = sapien_utils.look_at(eye=camera.eye, target=camera.target, up=up)
    assert abs(np.linalg.norm(array(pose.q))-1) < 1e-6
    return pose


def make_env(task, camera):
    import mani_skill.envs  # noqa: F401
    pose = camera_pose(camera)
    settings = {'pose': np.r_[array(pose.p).ravel(), array(pose.q).ravel()].tolist(),
                'width': camera.width, 'height': camera.height, 'fov': camera.fov}
    return gym.make(task, robot_uids=TASKS[task]['robot'], num_envs=1,
                    obs_mode='pointcloud', control_mode='pd_joint_pos',
                    sim_backend='physx_cpu', render_backend='cpu',
                    sensor_configs={'shader_pack': 'default', 'base_camera': settings},
                    reconfiguration_freq=1, max_episode_steps=1000)


def actors_for(base, task):
    if task == 'PushCube-v1': return {'cube': [base.obj], 'goal_region': [base.goal_region]}
    if task == 'StackCube-v1': return {'red_cube': [base.cubeA], 'green_cube': [base.cubeB]}
    if task == 'PegInsertionSide-v1': return {'peg': [base.peg], 'hole_box': [base.box]}
    return {'goal_triangle': [base.goal_tri], 'ink': base.dots}


def labels_for(base, task):
    actors = actors_for(base, task)
    links = {link.name: ids(link) for link in base.agent.robot.links}
    labels = {name: sum((ids(a) for a in values), []) for name, values in actors.items()}
    labels.update(robot=sum(links.values(), []),
                  wrist=sum((v for k, v in links.items() if k in WRIST_LINKS or 'stick' in k), []),
                  table=ids(base.table_scene.table), ground=ids(base.table_scene.ground))
    if task == 'DrawTriangle-v1': labels['canvas'] = ids(base.canvas)
    return labels, links


def counts(seg, labels):
    result = {name: int(np.isin(seg, v).sum()) for name, v in labels.items()}
    known = sum((v for k, v in labels.items() if k != 'wrist'), [])
    result['unknown'] = int((~np.isin(seg, known)).sum())
    assert sum(v for k, v in result.items() if k != 'wrist') == len(seg)
    return result


def object_corners(base, task):
    """Conservative object bounds: triangle/target geometry, not render visibility."""
    result = {}
    for name, objects in actors_for(base, task).items():
        parts = []
        for actor in objects:
            if name == 'ink' and float(array(actor.pose.p)[0, 2]) < .01: continue
            if name == 'goal_triangle':
                # Each outline side has half width .005; use all triangle vertices + margin.
                verts = base.original_verts
                local = np.concatenate([verts + d for d in itertools.product((-.005, .005), repeat=3)])
            else:
                half = (array(base.peg_half_sizes)[0] if name == 'peg' else
                        np.full(3, float(array(base.peg_half_sizes)[0, 0])) if name == 'hole_box' else
                        (.00005, .1, .1) if name == 'goal_region' else
                        (.002, .01, .01) if name == 'ink' else (.02, .02, .02))
                local = np.array(list(itertools.product(*[(-h, h) for h in half])))
            pose = array(actor.pose.to_transformation_matrix())[0]
            parts.append(local @ pose[:3, :3].T + pose[:3, 3])
        result[name] = np.concatenate(parts) if parts else np.empty((0, 3))
    return result


def capture(base, task, camera, obs, group, frame, episode, labels):
    crop_min, crop_max = TASKS[task]['crop'];objects = TASKS[task]['objects']
    pc = obs['pointcloud'];xyzw = torch.as_tensor(pc['xyzw']).detach().cpu().float()[0]
    seg = array(pc['segmentation'])[0].ravel();rgb = array(pc['rgb'])[0].reshape(-1, 3)
    valid = (xyzw[:, 3] > .5) & torch.isfinite(xyzw).all(-1)
    valid_ids = torch.nonzero(valid).flatten()
    tf = base.agent.robot.pose.inv().to_transformation_matrix().detach().cpu()[0]
    xyz = xyzw[valid, :3] @ tf[:3, :3].T + tf[:3, 3]
    keep = ((xyz >= torch.tensor(crop_min)) & (xyz <= torch.tensor(crop_max))).all(-1)
    cropped = xyz[keep];crop_ids = valid_ids[keep]
    pre = torch.arange(len(cropped))
    if len(pre) > 4096: pre = torch.randperm(len(cropped), generator=torch.Generator().manual_seed(42))[:4096]
    prexyz = cropped[pre];pre_ids = crop_ids[pre]
    chosen = farthest_point_indices(prexyz[None], 512)[0] if len(prexyz) >= 512 else torch.empty(0, dtype=torch.long)
    tcp_pose = base.agent.tcp.pose if task == 'DrawTriangle-v1' else base.agent.tcp_pose
    tcp = tf[:3, :3] @ tcp_pose.p.detach().cpu()[0] + tf[:3, 3]
    sampled_ids = pre_ids[chosen];sampled_seg = seg[sampled_ids.numpy()]
    centered = prexyz[chosen] - tcp
    features = torch.cat((centered, centered.norm(dim=-1, keepdim=True)), -1)
    row = {'task': task, 'camera': camera.name, 'episode': episode, 'frame': frame,
           'fps512_valid': len(chosen) == 512, 'crop_total': len(cropped), 'presample_total': len(prexyz)}
    for stage, indices in [('raw', valid_ids), ('crop', crop_ids), ('pre', pre_ids), ('fps512', sampled_ids)]:
        row.update({f'{stage}_{k}': v for k, v in counts(seg[indices.numpy()], labels).items()})
    assert row['fps512_ground'] == 0
    assert len(np.unique(sampled_ids.numpy())) == len(chosen)
    g = group.create_group(f'frame_{frame:05d}')
    for k, v in {'rgb_flat': rgb, 'segmentation_flat': seg, 'xyz_base_crop': cropped.numpy(),
                 'crop_source_indices': crop_ids.numpy(), 'fps512_features': features.numpy(),
                 'fps512_source_indices': sampled_ids.numpy(), 'fps512_segmentation': sampled_seg,
                 'tcp_base': tcp.numpy()}.items(): g.create_dataset(k, data=v, compression='lzf')
    offset = 0;sensors = g.create_group('sensors')
    for uid, sensor in base._sensors.items():
        config = sensor.config;size = config.width * config.height
        s = sensors.create_group(uid);local = slice(offset, offset + size)
        s.create_dataset('rgb', data=rgb[local].reshape(config.height, config.width, 3), compression='lzf')
        s.create_dataset('segmentation', data=seg[local].reshape(config.height, config.width), compression='lzf')
        s.attrs['pixel_offset'] = offset
        row.update({f'{uid}_raw_{k}': int(np.isin(seg[local][array(valid)[local]], labels[k]).sum()) for k in objects})
        params = obs['sensor_param'][uid]
        ext = array(params['extrinsic_cv'])[0];intr = array(params['intrinsic_cv'])[0]
        s.create_dataset('extrinsic_cv', data=ext);s.create_dataset('intrinsic_cv', data=intr)
        if uid == 'base_camera':
            actual_eye = -ext[:, :3].T @ ext[:, 3]
            assert np.allclose(actual_eye, camera.eye, atol=1e-6), (camera.name, actual_eye, camera.eye)
            for name, corners in object_corners(base, task).items():
                if not len(corners): row[f'frustum_clipped_{name}'] = False;continue
                cx = np.c_[corners, np.ones(len(corners))] @ ext.T
                px = cx @ intr.T;uv = px[:, :2] / px[:, 2:3]
                inside = (cx[:, 2] > .01) & (uv[:, 0] >= 0) & (uv[:, 0] < config.width) & (uv[:, 1] >= 0) & (uv[:, 1] < config.height)
                row[f'frustum_clipped_{name}'] = not bool(inside.all())
                bx = corners @ array(tf)[:3, :3].T + array(tf)[:3, 3]
                row[f'crop_corners_outside_{name}'] = int(((bx < crop_min) | (bx > crop_max)).any(-1).sum())
                g.create_dataset(f'{name}_world_corners', data=corners, compression='lzf')
        offset += size
    assert offset == len(seg)
    error = float(np.abs(features[:, :3].numpy() + tcp.numpy() - cropped[pre][chosen].numpy()).max()) if len(chosen) else 0.
    norm_error = float(np.abs(features[:, 3].numpy() - np.linalg.norm(features[:, :3].numpy(), axis=-1)).max()) if len(chosen) else 0.
    assert error < 1e-6 and norm_error < 1e-6
    assert np.array_equal(seg[sampled_ids.numpy()], sampled_seg)
    for name in objects: assert row[f'crop_{name}'] <= row[f'raw_{name}']
    row['audit_xyz_error'] = error;row['audit_distance_error'] = norm_error
    if frame == 0 and len(chosen) == 512:
        agent_view = SimpleNamespace(robot=base.agent.robot, tcp_pose=tcp_pose)
        expected, _ = pointcloud_features(obs, agent_view, ObservationConfig(num_points=512, crop_min=crop_min, crop_max=crop_max))
        row['adapter_error'] = float((expected[0] - features).abs().max());assert row['adapter_error'] < 1e-6
    g.attrs['statistics'] = json.dumps(row)
    return row


def summarize(rows, task):
    objects = TASKS[task]['objects']
    keys = [k for k in rows[0] if k.startswith(('raw_', 'crop_', 'pre_', 'fps512_')) and k != 'fps512_valid']
    return {'frames': len(rows), 'statistics': {k: statistics([r[k] for r in rows]) for k in keys},
            'fps512_invalid_frames': sum(not r['fps512_valid'] for r in rows),
            'object_crop_retention': {k: sum(r[f'crop_{k}'] for r in rows) / max(1, sum(r[f'raw_{k}'] for r in rows)) for k in objects},
            'frustum_clipped_frames': {k: sum(r[f'frustum_clipped_{k}'] for r in rows) for k in objects},
            'max_crop_corners_outside': {k: max(r.get(f'crop_corners_outside_{k}', 0) for r in rows) for k in objects},
            'max_xyz_audit_error': max(r['audit_xyz_error'] for r in rows),
            'max_distance_audit_error': max(r['audit_distance_error'] for r in rows),
            'base_camera_raw': {k: statistics([r[f'base_camera_raw_{k}'] for r in rows]) for k in objects},
            'by_episode': {str(e): {k: statistics([r[f'fps512_{k}'] for r in rows if r['episode'] == e]) for k in objects} for e in sorted(set(r['episode'] for r in rows))}}


def run(args):
    output = args.output.resolve();assert output.is_relative_to(HERE)
    output.mkdir(parents=True, exist_ok=False);started = time.monotonic();results = {}
    manifest = {'mode': 'snapshot', 'frames_per_episode': args.frames, 'tasks': args.tasks,
                'camera_pose_unit_quaternion_and_actual_eye_verified': True,
                'segmentation_used_for_sampling': False, 'FPS': 512, 'pre_sample_points': 4096, 'seed': 42,
                'all_native_sensors_retained': True, 'source': {}, 'crop_frame': 'robot_base_before_TCP_subtraction',
                'task_settings': TASKS, 'cameras': {t: {k: asdict(v) for k, v in candidates(t).items()} for t in args.tasks}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for task in args.tasks:
        source = ROOT/'.runtime/crop-inspection-20261006-1'/task/(TASKS[task]['source']+'.h5')
        meta = json.loads(source.with_suffix('.json').read_text());choices = candidates(task)
        manifest['source'][task] = {'path': str(source), 'sha256': sha256(source), 'json_sha256': sha256(source.with_suffix('.json'))}
        results[task] = {}
        with h5py.File(source, 'r') as h:
            for name in args.cameras or choices:
                camera = choices[name];env = make_env(task, camera);rows = [];errors = []
                path = output/f'{task}--{name}.h5'
                try:
                    with h5py.File(path, 'w') as recorded:
                        recorded.attrs['task'] = task;recorded.attrs['camera'] = json.dumps(asdict(camera))
                        recorded.attrs['crop'] = json.dumps(TASKS[task]['crop'])
                        for ep in meta['episodes']:
                            e = ep['episode_id'];saved = h[f'traj_{e}'];length = len(saved['actions'])
                            env.reset(**ep['reset_kwargs']);base = env.unwrapped
                            labels, links = labels_for(base, task)
                            group = recorded.create_group(f'traj_{e}');group.attrs['entity_ids'] = json.dumps(labels)
                            group.attrs['robot_links'] = json.dumps(links);group.attrs['source_metadata'] = json.dumps(ep)
                            frames = range(length+1) if args.frames == 0 else sorted(set(np.linspace(0, length, args.frames).astype(int).tolist()))
                            for frame in frames:
                                state = source_state(saved, frame);base.set_state_dict(state)
                                if task == 'DrawTriangle-v1': base.draw_step = frame
                                err = state_error(env, state);assert err < 1e-5;errors.append(err)
                                row = capture(base, task, camera, base.get_obs(), group, frame, e, labels)
                                row['state_error'] = err;rows.append(row)
                            print(f'{task}/{name}: episode {e}, {len(frames)} frames', flush=True)
                    result = summarize(rows, task);result['camera'] = asdict(camera);result['recording'] = str(path)
                    result['max_state_error'] = max(errors);results[task][name] = result
                    (output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
                    primary = TASKS[task]['primary']
                    print(f"RESULT {task}/{name}: raw {result['statistics']['raw_'+primary]}; FPS {result['statistics']['fps512_'+primary]}", flush=True)
                finally: env.close()
    manifest['completed'] = True;manifest['elapsed_seconds'] = time.monotonic()-started
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', nargs='+', choices=list(TASKS), default=list(TASKS))
    p.add_argument('--cameras', nargs='+')
    p.add_argument('--frames', type=int, default=18, help='0 records all states')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.frames < 0: p.error('frames must be nonnegative')
    run(a)
