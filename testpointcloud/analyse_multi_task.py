"""Audit experimental multi-task recordings; export camera ranking and paired plots."""
import argparse
import json
from pathlib import Path

from multi_task_experiment import HERE, TASKS, Camera, camera_pose, array
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def rank(task, results):
    primary = TASKS[task]['primary']
    secondary = [k for k in TASKS[task]['objects'] if k != primary and k != 'ink']
    def key(item):
        _, r = item;s = r['statistics'];groups = [primary] + secondary
        missing = sum(s[f'fps512_{k}']['zero_frames'] for k in groups)
        clipped = sum(r['frustum_clipped_frames'][k] for k in groups)
        eligible = r['fps512_invalid_frames'] == 0 and missing == 0 and clipped == 0
        return (eligible, -missing, -clipped, s[f'fps512_{primary}']['p05'],
                min(s[f'fps512_{k}']['p05'] for k in secondary) if secondary else 0,
                s[f'fps512_{primary}']['median'], s['fps512_wrist']['p05'])
    return [k for k, _ in sorted(results.items(), key=key, reverse=True)]


def audit(task, entry):
    bounds = np.array(TASKS[task]['crop']);number = 0;maxerr = 0.
    losses = {k: {'raw_zero_frames': 0, 'crop_lost_visible_frames': 0,
                  'presample_lost_visible_frames': 0, 'fps512_lost_visible_frames': 0}
              for k in TASKS[task]['objects']}
    with h5py.File(entry['recording'], 'r') as h:
        for traj in h.values():
            labels = json.loads(traj.attrs['entity_ids'])
            for g in traj.values():
                row = json.loads(g.attrs['statistics']);seg = g['segmentation_flat'][:]
                xyz = g['xyz_base_crop'][:];crop_ids = g['crop_source_indices'][:]
                assert ((xyz >= bounds[0]-1e-6)&(xyz <= bounds[1]+1e-6)).all()
                assert len(xyz) == len(crop_ids) == row['crop_total'] and np.all(np.diff(crop_ids)>0)
                ids = g['fps512_source_indices'][:];p = g['fps512_features'][:];ss = g['fps512_segmentation'][:]
                assert len(ids) == len(np.unique(ids)) and p.shape == (len(ids), 4)
                if row['fps512_valid']: assert len(ids) == 512
                assert np.array_equal(seg[ids], ss)
                locations = np.searchsorted(crop_ids, ids);assert np.array_equal(crop_ids[locations], ids)
                if len(ids):
                    error = float(np.abs(p[:, :3]+g['tcp_base'][:]-xyz[locations]).max())
                    assert error < 1e-6;maxerr = max(maxerr, error)
                    assert np.abs(p[:, 3]-np.linalg.norm(p[:, :3], axis=-1)).max() < 1e-6
                for name, values in labels.items():
                    assert int(np.isin(ss, values).sum()) == row[f'fps512_{name}']
                    # No native far-depth foreground in these operation regions.
                    assert int(np.isin(seg, values).sum()) == row[f'raw_{name}']
                assert row['fps512_ground'] == 0
                for name, counts in losses.items():
                    counts['raw_zero_frames'] += row['raw_'+name] == 0
                    counts['crop_lost_visible_frames'] += row['raw_'+name] > 0 and row['crop_'+name] == 0
                    counts['presample_lost_visible_frames'] += row['crop_'+name] > 0 and row['pre_'+name] == 0
                    counts['fps512_lost_visible_frames'] += row['pre_'+name] > 0 and row['fps512_'+name] == 0
                for sensor in g['sensors'].values():
                    offset = sensor.attrs['pixel_offset'];local = sensor['segmentation'][:].ravel()
                    assert np.array_equal(seg[offset:offset+len(local)], local)
                extrinsic = g['sensors/base_camera/extrinsic_cv'][:]
                assert np.allclose(-extrinsic[:, :3].T @ extrinsic[:, 3], entry['camera']['eye'], atol=1e-6)
                number += 1
    assert number == entry['frames']
    return {'passed': True, 'frames': number, 'max_xyz_error': maxerr, 'object_loss_stages': losses}


def plot(task, results, names, episode, frame, path):
    objects = TASKS[task]['objects'];color = {objects[0]: '#f84938', objects[1]: '#56ce83',
        'robot': '#49a4d8', 'table': '#a8adb5', 'canvas': '#c7c8ce', 'unknown': '#bd78c4'}
    fig = plt.figure(figsize=(13, 4*len(names)))
    for i, name in enumerate(names):
        with h5py.File(results[name]['recording'], 'r') as h:
            traj = h[f'traj_{episode}'];g = traj[f'frame_{frame:05d}']
            row = json.loads(g.attrs['statistics']);labels = json.loads(traj.attrs['entity_ids'])
            xyz = g['fps512_features'][:, :3]+g['tcp_base'][:];seg = g['fps512_segmentation'][:]
            rgb = g['sensors/base_camera/rgb'][:]
        ax = fig.add_subplot(len(names), 2, 2*i+1);ax.imshow(rgb);ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(name+' | raw '+', '.join(f'{o}={row["raw_"+o]}' for o in objects), fontsize=10)
        ax = fig.add_subplot(len(names), 2, 2*i+2, projection='3d')
        for k, values in labels.items():
            if k in ('wrist', 'ground'): continue
            mask = np.isin(seg, values)
            ax.scatter(*xyz[mask].T, c=color.get(k, '#bd78c4'), s=32 if k in objects else 6,
                       alpha=.85 if k in objects else .4, label=f'{k} {int(mask.sum())}')
        ax.set_title('Global FPS512 (all native sensors)', fontsize=10)
        ax.set_xlabel('Base X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)')
        for setter, a, b in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), *TASKS[task]['crop']): setter(a,b)
        ax.view_init(elev=28, azim=-55);ax.legend(loc='upper left', fontsize=7)
    fig.suptitle(f'{task} | identical source state: trajectory {episode}, frame {frame}', fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(path, dpi=140);plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args();out = args.output.resolve();assert out.is_relative_to(HERE)
    out.mkdir(parents=True, exist_ok=False);screen = {};validation = {};audits = {};frames = 0;excluded = []
    for folder in args.inputs:
        manifest = json.loads((folder/'manifest.json').read_text());assert manifest.get('completed')
        summaries = json.loads((folder/'summary.json').read_text())
        target = validation if manifest['frames_per_episode'] == 0 else screen
        for task, entries in summaries.items():
            for name, entry in entries.items():
                if task == 'DrawTriangle-v1' and name.startswith('overhead') and not manifest.get('camera_pose_unit_quaternion_and_actual_eye_verified'):
                    excluded.append({'run': str(folder), 'task': task, 'camera': name,
                                     'reason': 'Degenerate look_at: default up parallel to vertical view; actual eye differed from requested. Repeated with explicit up.'})
                    continue
                target.setdefault(task, {})[name] = entry
                report = audit(task, entry);frames += report['frames']
                entry['object_loss_stages'] = report['object_loss_stages']
                audits[f'{folder}/{task}/{name}'] = report
    report = {'screen': screen, 'validation': validation,
              'ranking_rule': 'Prefer no zero target frames and no geometric frustum clipping; then primary and secondary FPS512 P05, primary median and wrist P05.',
              'tasks': {}, 'integrity': {'passed': True, 'frames': frames, 'runs': audits},
              'excluded_invalid_camera_runs': excluded,
              'scope': '3 source expert trajectories per task; ranking and full re-render use same episodes; no held-out policy evaluation.'}
    for task in screen:
        entries = validation.get(task) or screen[task];order = rank(task, entries);winner = order[0]
        camera = entries[winner]['camera'];pose = camera_pose(Camera(**camera))
        primary = TASKS[task]['primary'];other = [o for o in TASKS[task]['objects'] if o != 'ink']
        preset = {'status': 'experiment_only', 'camera': camera, 'sensor_configs': {'shader_pack': 'default', 'base_camera': {
            'pose': np.r_[array(pose.p).ravel(), array(pose.q).ravel()].tolist(), 'width': camera['width'], 'height': camera['height'], 'fov': camera['fov']}},
            'crop': {'frame': 'robot_base_before_TCP_subtraction', 'min': TASKS[task]['crop'][0], 'max': TASKS[task]['crop'][1]},
            'sampling': {'num_points':512, 'pre_sample_points':4096, 'seed':42},
            'native_hand_camera_retained': task == 'PegInsertionSide-v1',
            'camera_has_zero_target_frames': any(entries[winner]['statistics']['raw_'+o]['zero_frames'] for o in other),
            'camera_has_frustum_clipping': any(entries[winner]['frustum_clipped_frames'][o] for o in other)}
        (out/f'{task}-camera-preset.json').write_text(json.dumps(preset, indent=2)+'\n')
        report['tasks'][task] = {'screen_ranking': rank(task, screen[task]), 'validation_ranking': order,
                               'recommended_camera': winner, 'preset': preset, 'result': entries[winner]}
        names = ['baseline']+([winner] if winner != 'baseline' else [])
        if len(names) == 1 and len(order)>1: names.append(order[1])
        with h5py.File(entries[winner]['recording'], 'r') as h:
            candidates = [(int(t.rsplit('_',1)[-1]), int(f.rsplit('_',1)[-1]), json.loads(g.attrs['statistics']))
                          for t,traj in h.items() for f,g in traj.items()]
        # Representative mid-frame and worst-target frame; all views use identical states.
        ep, frame, _ = candidates[len(candidates)//2]
        plot(task, entries, names, ep, frame, out/f'{task}-comparison-middle.png')
        ep, frame, _ = min(candidates, key=lambda x: x[2]['fps512_'+primary])
        plot(task, entries, names, ep, frame, out/f'{task}-comparison-worst.png')
        worst_frames = {}
        for obj in TASKS[task]['objects']:
            ep, frame, row = min(candidates, key=lambda x: x[2]['fps512_'+obj])
            worst_frames[obj] = {'episode': ep, 'frame': frame, 'raw_points': row['raw_'+obj],
                                 'crop_points': row['crop_'+obj], 'presample_points': row['pre_'+obj],
                                 'fps512_points': row['fps512_'+obj]}
            if obj != primary and obj != 'ink':
                plot(task, entries, names, ep, frame, out/f'{task}-comparison-worst-{obj}.png')
        report['tasks'][task]['worst_frames'] = worst_frames
        print(task, winner, entries[winner]['statistics']['fps512_'+primary], flush=True)
    (out/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Passed audit: {frames} frame/config pairs', flush=True)


if __name__ == '__main__': main()
