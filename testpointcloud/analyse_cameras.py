"""Summarize camera runs and export paired RGB / semantic point-cloud comparisons."""
import argparse
import csv
import json
from pathlib import Path

from camera_experiment import HERE, CROP_MIN, CROP_MAX, array
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def rank(results, frusta):
    def key(item):
        name, v = item;s = v['statistics']
        eligible = (frusta[name]['clipped_frames'] == 0 and s['raw_cube']['zero_frames'] == 0
                    and v['fps512_invalid_frames'] == 0 and v['fps512_sampling_lost_visible_cube_frames'] == 0)
        return (eligible, s['fps512_cube']['p05'], s['fps512_cube']['median'],
                s['fps512_wrist']['p05'], s['raw_cube']['p05'], -v['camera']['resolution'])
    return [name for name, _ in sorted(results.items(), key=key, reverse=True)]


def style(ax):
    ax.set_facecolor('#101725');ax.tick_params(colors='#cdd6e2', labelsize=8)
    for spine in ax.spines.values(): spine.set_color('#536173')
    for label in [ax.xaxis.label, ax.yaxis.label, ax.title]: label.set_color('#eef2f7')


def plot_ranking(results, order, frusta, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(order)*.31)), facecolor='#101725', sharey=True)
    y = np.arange(len(order));labels = []
    for name in order:
        v = results[name];bad = frusta[name]['clipped_frames'] or v['statistics']['raw_cube']['zero_frames']
        labels.append(name + (' [clipping/occlusion]' if bad else ''))
    for ax, field, title in zip(axes, ['raw_cube', 'fps512_cube'], ['Raw visible cube pixels', 'Cube points after FPS512']):
        style(ax);med = [results[n]['statistics'][field]['median'] for n in order]
        p05 = [results[n]['statistics'][field]['p05'] for n in order]
        ax.barh(y, med, color='#5a9de2', alpha=.8, label='Median')
        ax.scatter(p05, y, color='#ffb454', s=23, label='5th percentile', zorder=3)
        ax.set_title(title);ax.set_xlabel('Points per frame');ax.grid(axis='x', alpha=.15)
        ax.legend(facecolor='#101725', labelcolor='white', loc='lower right', fontsize=8)
    axes[0].set_yticks(y, labels, fontsize=8);axes[0].invert_yaxis()
    fig.suptitle('Identical scene states, fixed operation crop; segmentation only labels diagnostics', color='white')
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(path, dpi=160);plt.close(fig)


def cloud_axes(ax, small=False):
    style(ax);ax.set_xlabel('Base X (m)', fontsize=8);ax.set_ylabel('Y (m)', fontsize=8)
    ax.set_zlabel('Z (m)', fontsize=8, color='#eef2f7');ax.view_init(elev=28, azim=-55)
    ax.set_box_aspect([1,1,1])
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((.06,.09,.14,1));axis.label.set_color('white')
    if not small:
        for setlim, a, b in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], CROP_MIN, CROP_MAX): setlim(a,b)


def plot_frame(results, names, source_id, frame, title, path):
    fig = plt.figure(figsize=(15, 4.4*len(names)), facecolor='#101725')
    for i, name in enumerate(names):
        with h5py.File(results[name]['recording'], 'r') as h:
            traj = h[f'traj_{source_id}'];g = traj[f'frame_{frame:05d}'];r = json.loads(g.attrs['statistics'])
            labels = json.loads(traj.attrs['entity_ids']);rgb = g['rgb'][:];segim = g['segmentation_image'][:]
            features = g['fps512_features'][:];xyz = features[:,:3]+g['tcp_base'][:]
            seg = g['fps512_segmentation'][:];cube = np.isin(seg, labels['cube'])
            robot = np.isin(seg, labels['robot']);table = np.isin(seg, labels['table'])
            raw_cube = g['raw_cube_xyz_base'][:]
        ax = fig.add_subplot(len(names), 3, 3*i+1);style(ax);ax.imshow(rgb);ax.set_xticks([]);ax.set_yticks([])
        mask = np.isin(segim, labels['cube']);yy, xx = np.nonzero(mask)
        if len(xx):ax.add_patch(Rectangle((xx.min()-.5,yy.min()-.5), xx.max()-xx.min()+1, yy.max()-yy.min()+1,
                                        fill=False, edgecolor='white', linewidth=1.2))
        camera = results[name]['camera']
        ax.set_title(f"{name}\nNative {camera['resolution']}x{camera['resolution']} RGB; raw cube={r['raw_cube']}", fontsize=10)
        ax = fig.add_subplot(len(names), 3, 3*i+2, projection='3d');cloud_axes(ax)
        for mask, color, size, label in [(table,'#a3aab5',5,'table'),(robot,'#54b6eb',9,'robot'),(cube,'#ff5547',55,'cube')]:
            ax.scatter(*xyz[mask].T,c=color,s=size,alpha=.75 if label!='table' else .4,label=f'{label}: {int(mask.sum())}')
        ax.set_title('512 sampled points, entity-colored for inspection', fontsize=10)
        ax.legend(loc='upper left',facecolor='#101725',labelcolor='white',fontsize=8)
        ax = fig.add_subplot(len(names), 3, 3*i+3, projection='3d');cloud_axes(ax, small=True)
        if len(raw_cube):
            ax.scatter(*raw_cube.T,c='#f39b8b',s=4,alpha=.4,label=f'raw cube: {len(raw_cube)}')
            center=(raw_cube.min(0)+raw_cube.max(0))/2
        else:center=xyz.mean(0)
        ax.scatter(*xyz[cube].T,c='#ff3f32',s=55,edgecolors='white',linewidths=.4,label=f'FPS cube: {int(cube.sum())}')
        for setlim,c in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],center):setlim(c-.027,c+.027)
        ax.set_title('Cube close-up: raw surface and sampled points', fontsize=10)
        ax.legend(loc='upper left',facecolor='#101725',labelcolor='white',fontsize=8)
    fig.suptitle(f'{title} | source trajectory {source_id}, frame {frame} | same physical state',color='white',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(path,dpi=140);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--frustum', type=Path, default=HERE/'runs/frustum01.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve();assert destination.is_relative_to(HERE)
    destination.mkdir(parents=True,exist_ok=False)
    frusta = json.loads(args.frustum.read_text());runs = {};screen = {};validation = {};replay = {}
    for path in args.inputs:
        manifest = json.loads((path/'manifest.json').read_text())
        assert manifest.get('completed')
        data = json.loads((path/'summary.json').read_text());runs[str(path)] = data
        if manifest['mode']=='action_replay':replay.update(data)
        elif manifest['frames_per_episode']==0:validation.update(data)
        else:screen.update(data)
    ranked = rank(validation or screen,frusta);winner = ranked[0]
    report = {'recommended_camera': (validation or screen)[winner]['camera'],
              'ranking_rule': 'Reject full-source frustum clipping, observed raw zeros and FPS invalid/lost-visible frames; then prioritize FPS512 cube p05, median, wrist p05.',
              'screen_ranking':rank(screen,frusta) if screen else [],'validation_ranking':ranked,
              'fixed_crop':{'frame':'robot_base before subtracting TCP','min':CROP_MIN,'max':CROP_MAX},
              'full_source_frustum':frusta[winner], 'screen':screen,'validation':validation,'replay':replay,
              'no_policy_training_or_success_rate_claim':True}
    (destination/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    from mani_skill.utils import sapien_utils
    config=report['recommended_camera']
    pose=sapien_utils.look_at(eye=config['eye'],target=config['target'])
    preset={'env_id':'PickCube-v1','robot_uids':'panda','camera_coordinate_frame':'world',
            'camera':config,
            'sensor_configs':{'shader_pack':'default','base_camera':{
                'pose':np.r_[array(pose.p).reshape(-1),array(pose.q).reshape(-1)].tolist(),
                'width':config['resolution'],'height':config['resolution'],
                'fov':float(np.radians(config['fov_degrees']))}},
            'pointcloud':{'num_points':512,'crop_min':CROP_MIN,'crop_max':CROP_MAX,
                          'crop_coordinate_frame':'robot_base_before_TCP_subtraction',
                          'pre_sample_points':4096,'sampling_seed':42,'length_scale':1.0},
            'status':'experiment_only; production observation contract has not been changed'}
    (destination/'camera-preset.json').write_text(json.dumps(preset,indent=2)+'\n')
    plot_ranking(screen,rank(screen,frusta),frusta,destination/'screen-ranking.png')
    if validation:plot_ranking(validation,ranked,frusta,destination/'validation-ranking.png')
    comparison = validation or screen
    names = list(dict.fromkeys([n for n in ['legacy128_90',winner,*ranked[:2]] if n in comparison]))
    names=names[:3]
    # Paired illustrations at worst object-sampling frames in each important phase.
    origin=Path(comparison[winner]['recording']).parent
    with (origin/'counts.csv').open() as f: rows=[r for r in csv.DictReader(f) if r['camera']==winner]
    examples=[]
    for phase in ('approach','near_contact','lifted'):
        candidates=[r for r in rows if r['phase']==phase]
        if not candidates:continue
        row=min(candidates,key=lambda r:(int(r['fps512_cube']),int(r['raw_cube'])))
        source_id=int(row['source_episode']);frame=int(row['frame'])
        filename=f'comparison-{phase}.png'
        plot_frame(comparison,names,source_id,frame,phase,destination/filename)
        examples.append({'phase':phase,'source_episode':source_id,'frame':frame,'cameras':names,'figure':filename})
    (destination/'plot-examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    print('Recommended:',winner,json.dumps(report['recommended_camera']))
    print('Report:',destination)


if __name__=='__main__':main()
