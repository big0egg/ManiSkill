"""回放训练动作，补录场景/相机/点云/状态视频，并检查观测一致性。"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np

from dataset_io import DEFAULT_DATASET, ROOT, default_output, episode_names, load_episode, require_new, sha256
from control_modes import CONTROL_CHOICES


def inspect_dataset(dataset, near_radius, control_mode=None):
    names = episode_names(dataset, control_mode)
    lengths, distances, near_counts, details = [], [], [], []
    for name in names:
        ep = load_episode(dataset, name, control_mode)
        d = ep.features[..., 3] * ep.scale
        near = np.sum(d <= near_radius, axis=1)
        norm_error = float(np.max(np.abs(np.linalg.norm(ep.features[..., :3], axis=-1) - ep.features[..., 3])))
        lengths.append(len(ep.actions))
        distances.append(d.ravel())
        near_counts.append(near)
        details.append({'episode': name, 'source_episode': ep.metadata['source_episode'],
                        'seed': ep.metadata.get('seed'), 'actions': len(ep.actions), 'observations': len(ep.states),
                        'recorded_success_end': bool(ep.metadata['success_end']),
                        'max_distance_norm_error': norm_error,
                        'near_points_min': int(near.min()), 'near_points_median': float(np.median(near)),
                        'terminal_tcp_goal_distance_m': None if ep.goals is None else float(np.linalg.norm(ep.tcp[-1] - ep.goals[-1]))})
    if not names:
        raise ValueError('数据集没有 episode')
    d = np.concatenate(distances)
    near = np.concatenate(near_counts)
    return {'dataset': str(Path(dataset).resolve()), 'dataset_sha256': sha256(dataset),
            'env_id': ep.contract['env_id'], 'robot_uids': ep.contract['robot_uids'],
            'control_mode': ep.contract['control_mode'],
            'episode_count': len(names), 'actions': sum(lengths), 'observations': sum(lengths) + len(names),
            'steps_min_median_max': [min(lengths), float(np.median(lengths)), max(lengths)],
            'finite_and_shapes_valid': True, 'near_radius_m': near_radius,
            'near_points_min_median_max': [int(near.min()), float(np.median(near)), int(near.max())],
            'distance_m_quantiles': dict(zip(['min', 'p10', 'median', 'p90', 'p99', 'max'],
                                           map(float, np.quantile(d, [0, .1, .5, .9, .99, 1])))),
            'recorded_success_count': sum(e['recorded_success_end'] for e in details),
            'note': '结构有效和记录成功不等同于训练数据高质量；需检查视频、点云覆盖和任务多样性。',
            'episodes': details}


class Dashboard:
    """无桌面 Matplotlib 面板。视频中的点云始终取自保存的 HDF5。"""
    def __init__(self, episode, near_radius):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        self.ep, self.radius = episode, near_radius
        self.figure = Figure(figsize=(12, 12), dpi=100, facecolor='#101725')
        self.canvas = FigureCanvasAgg(self.figure)
        grid = self.figure.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=.32, wspace=.23)
        self.cloud_axes = [self.figure.add_subplot(grid[0, i], projection='3d') for i in range(2)]
        self.curve_axes = [self.figure.add_subplot(grid[i + 1, j]) for i in range(2) for j in range(2)]
        self.cloud_artists = []
        xyz = episode.features[..., :3].astype(np.float64) * episode.scale + episode.tcp[:, None, :]
        markers = np.vstack((np.zeros((1, 3)), episode.tcp))
        if episode.goals is not None:
            markers = np.vstack((markers, episode.goals))
        self.global_bounds = np.vstack((xyz.reshape(-1, 3), markers))
        self.near_center = (episode.tcp.min(axis=0) + episode.tcp.max(axis=0)) / 2
        self.limit = float(episode.features[..., 3].max() * episode.scale)
        for ax, title in zip(self.cloud_axes, ['Saved point cloud: full scene (base frame)', f'Saved point cloud: within {near_radius:g} m of TCP']):
            self.style(ax)
            ax.set_title(title, color='white', fontsize=10)
            ax.set_xlabel('X (m)', color='white'); ax.set_ylabel('Y (m)', color='white'); ax.set_zlabel('Z (m)', color='white')
            ax.view_init(elev=30, azim=-55)
            ax.set_box_aspect([1, 1, 1])
        center = (self.global_bounds.min(axis=0) + self.global_bounds.max(axis=0)) / 2
        span = max(float(np.ptp(self.global_bounds, axis=0).max()) / 2, .4)
        for ax, c, r in zip(self.cloud_axes, [center, self.near_center], [span, near_radius + .25]):
            ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        s = episode.states
        t = np.arange(len(s))
        field = episode.contract['state_fields']
        lo, hi = field['qpos']; qpos = s[:, lo:hi]
        lo, hi = field['qvel']; qvel = s[:, lo:hi]
        auxiliary = episode.tcp if episode.goals is None else np.column_stack((qpos[:, 7:9], np.linalg.norm(episode.tcp - episode.goals, axis=1)))
        curves = [qpos[:, :7], qvel[:, :7], auxiliary, episode.actions]
        action_title = ('Saved joint targets (rad) / gripper (normalized)' if episode.contract['control_mode'] == 'pd_joint_pos'
                        else 'Saved action (normalized)')
        titles = ['Joint position (rad)', 'Joint velocity (rad/s)', 'TCP position XYZ (m)' if episode.goals is None else 'Finger positions / TCP-goal distance (m)', action_title]
        self.cursors = []
        for ax, data, title in zip(self.curve_axes, curves, titles):
            self.style(ax)
            labels = (['X', 'Y', 'Z'] if episode.goals is None else ['finger 1', 'finger 2', 'TCP-goal']) if title == titles[2] else [str(i) for i in range(data.shape[1])]
            for i, label in enumerate(labels):
                ax.plot(t[:len(data)], data[:, i], label=label, linewidth=1)
            ax.set_title(title, color='white', fontsize=10)
            ax.set_xlabel('Observation frame / action step', color='white', fontsize=8)
            ax.set_xlim(0, max(len(s) - 1, 1))
            ax.legend(fontsize=6, loc='upper right', ncol=3)
            self.cursors.append(ax.axvline(0, color='#ffdddd', linewidth=1.5))

    @staticmethod
    def style(ax):
        ax.set_facecolor('#101725')
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#8899aa')
        ax.grid(alpha=.15)

    def render(self, frame):
        for artist in self.cloud_artists:
            artist.remove()
        self.cloud_artists = []
        ep = self.ep
        points = ep.positions(frame)
        distances = ep.features[frame, :, 3] * ep.scale
        for ax, mask, limit in zip(self.cloud_axes, [np.ones(len(points), bool), distances <= self.radius], [self.limit, self.radius]):
            p = points[mask]
            self.cloud_artists.append(ax.scatter(*p.T, c=distances[mask], cmap='turbo', vmin=0, vmax=limit, s=8, depthshade=False))
            for position, color, marker in [(ep.tcp[frame], 'cyan', 'o'), (None if ep.goals is None else ep.goals[frame], 'lime', '*'), (np.zeros(3), 'white', '+')]:
                if position is None:
                    continue
                self.cloud_artists.append(ax.scatter(*position, c=color, marker=marker, s=35))
            for axis, color in zip(np.eye(3) * .25, ['red', 'lime', 'blue']):
                line, = ax.plot([0, axis[0]], [0, axis[1]], [0, axis[2]], color=color)
                self.cloud_artists.append(line)
        for cursor in self.cursors:
            cursor.set_xdata([frame, frame])
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].copy()


def rgb_array(image):
    if hasattr(image, 'detach'):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError('只支持单环境录像')
        image = image[0]
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255, 0, 255)
    return image[..., :3].astype(np.uint8)


def compose(scene, sensor, dashboard, lines):
    from PIL import Image, ImageDraw, ImageFont
    result = Image.new('RGB', (1800, 1200), '#101725')
    result.paste(Image.fromarray(scene).resize((600, 600)), (0, 0))
    # 最近邻放大，保留所录制相机的原生像素细节。
    result.paste(Image.fromarray(sensor).resize((480, 480), Image.Resampling.NEAREST), (60, 630))
    result.paste(Image.fromarray(dashboard), (600, 0))
    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype(str(ROOT / 'mani_skill/utils/visualization/UbuntuSansMono-Regular.ttf'), 17)
    draw.rectangle((0, 0, 600, 105), fill='#101725')
    draw.multiline_text((12, 10), '\n'.join(lines), fill='white', font=font)
    draw.text((12, 608), f'Replayed base_camera RGB (native {sensor.shape[1]}x{sensor.shape[0]})', fill='white', font=font)
    return np.asarray(result)


def replay_episode(ep, env, raw, raw_episodes, args, destination):
    import imageio.v2 as imageio
    import torch
    from mani_skill.trajectory import utils as trajectory_utils
    from obs_adapter import config_from_contract, adapt_observation
    source_id = ep.metadata['source_episode']
    meta = raw_episodes[source_id]
    initial = trajectory_utils.dict_to_list_of_dicts(raw[f'traj_{source_id}/env_states'])[0]
    reset = dict(meta['reset_kwargs'])
    if isinstance(reset.get('seed'), list):
        reset['seed'] = reset['seed'][0]
    env.reset(**reset)
    env.unwrapped.set_state_dict(initial)
    config = config_from_contract(ep.contract)
    dashboard = Dashboard(ep, args.near_radius)
    state_errors = {key: 0. for key in ep.contract['state_fields']}
    point_error = 0.
    mismatch_frames = []
    success_once = False
    fps = args.fps or env.unwrapped.control_freq
    info = env.unwrapped.evaluate()
    obs = env.unwrapped.get_obs()
    started = time.monotonic()
    with imageio.get_writer(str(destination), fps=fps, codec='libx264', quality=8, macro_block_size=1) as writer:
        for frame in range(len(ep.states)):
            if frame:
                obs, _, _, _, info = env.step(torch.as_tensor(ep.actions[frame - 1]))
            adapted, _ = adapt_observation(obs, env.unwrapped.agent, config, contract=ep.contract)
            state = adapted['state'][0].numpy()
            features = adapted['pointcloud_distance'][0].numpy()
            current_errors = {key: float(np.max(np.abs(state[lo:hi] - ep.states[frame, lo:hi])))
                              for key, (lo, hi) in ep.contract['state_fields'].items()}
            # 逐索引严格比较，采样顺序变化也算不一致，避免把回放视频当成原始 RGB。
            current_point_error = float(np.max(np.abs(features - ep.features[frame])))
            for key, error in current_errors.items():
                state_errors[key] = max(state_errors[key], error)
            point_error = max(point_error, current_point_error)
            matching = all(value <= (args.velocity_tolerance if key == 'qvel' else args.state_tolerance)
                           for key, value in current_errors.items()) and current_point_error <= args.pointcloud_tolerance
            if not matching:
                mismatch_frames.append(frame)
            success = bool(torch.as_tensor(info['success']).item())
            success_once |= success
            scene = rgb_array(env.render())
            sensor_images = env.unwrapped.get_sensor_images()
            sensor = rgb_array(sensor_images['base_camera']['rgb'])
            lines = [f'{ep.name} | frame {frame}/{len(ep.actions)} | seed {ep.metadata.get("seed")}',
                     f'Replayed RGB | saved HDF5 cloud/state panels',
                     f'Observation match: {"PASS" if matching else "MISMATCH"} | success: {success}',
                     f'Cloud max error: {current_point_error:.3g} (policy units)']
            writer.append_data(compose(scene, sensor, dashboard.render(frame), lines))
            if frame % 20 == 0:
                print(f'{ep.name}: frame {frame}/{len(ep.actions)}, match={matching}', flush=True)
    # 实际解码确认帧数、尺寸；不只检查文件存在。
    decoded = 0
    with imageio.get_reader(str(destination)) as reader:
        for image in reader:
            if image.shape != (1200, 1800, 3):
                raise RuntimeError(f'视频尺寸异常：{image.shape}')
            decoded += 1
    if decoded != len(ep.states):
        raise RuntimeError(f'视频帧数异常：{decoded} != {len(ep.states)}')
    return {'episode': ep.name, 'source_episode': source_id, 'video': str(destination.resolve()),
            'frames': decoded, 'fps': fps, 'max_state_errors': state_errors,
            'max_pointcloud_feature_error': point_error, 'matching_observations': not mismatch_frames,
            'mismatch_frames': mismatch_frames, 'replay_success_once': success_once,
            'replay_success_end': success, 'elapsed_seconds': time.monotonic() - started}


def run(args):
    control_mode = getattr(args, 'control_mode', None)
    output = args.output or default_output(args.dataset)
    report_path = require_new(output / ('inspection.json' if args.inspect_only else 'quality.json'))
    report = inspect_dataset(args.dataset, args.near_radius, control_mode)
    if args.inspect_only:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(f'数据检查完成：{report_path}；{report["episode_count"]} 条轨迹')
        return
    names = episode_names(args.dataset, control_mode)
    if args.all:
        selected = names
    elif args.episodes:
        selected = [load_episode(args.dataset, name, control_mode).name for name in args.episodes]
    else:
        indices = [0, 2, 54, 80, 99] if len(names) >= 100 else list(range(min(5, len(names))))
        selected = [names[i] for i in indices]
    if len(set(selected)) != len(selected):
        raise ValueError('episodes 不得重复')
    destinations = {name: require_new(output / (name + '.mp4')) for name in selected}
    with h5py.File(args.dataset, 'r') as f:
        manifest = json.loads(f.attrs['manifest'])
    source = args.raw_source or args.dataset.with_suffix('.raw.h5')
    if not source.exists() and args.raw_source is None:
        source = Path(manifest['source'])
    source_json = source.with_suffix('.json')
    for path, expected in [(source, manifest['source_sha256']), (source_json, manifest['source_json_sha256'])]:
        if sha256(path) != expected:
            raise ValueError(f'原始轨迹来源哈希不匹配：{path}')
    raw_meta = json.loads(source_json.read_text())
    raw_episodes = {e['episode_id']: e for e in raw_meta['episodes']}
    sys.path.insert(0, str(ROOT / 'examples/baselines/flow_dp3'))
    from obs_adapter import config_from_contract, make_env
    contract = load_episode(args.dataset, selected[0], control_mode).contract
    config_from_contract(contract)
    os.environ.setdefault('VK_ICD_FILENAMES', '/usr/share/vulkan/icd.d/lvp_icd.json')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime/matplotlib'))
    # CPU 仿真及软件 Vulkan；策略计算和 W&B 不参与录像。
    env = make_env(max_episode_steps=max(16, report['steps_min_median_max'][2]), render_mode='rgb_array',
                   contract=contract)
    report.update({'raw_source': str(source.resolve()), 'video_kind': 'training_action_replay',
                   'pointcloud_panels': 'saved HDF5 observations',
                   'tolerances': {'state': args.state_tolerance, 'qvel': args.velocity_tolerance,
                                  'pointcloud_policy_units': args.pointcloud_tolerance}, 'replays': []})
    completed = False
    try:
        with h5py.File(source, 'r') as raw:
            for name in selected:
                ep = load_episode(args.dataset, name, control_mode)
                if ep.contract != contract:
                    raise ValueError(f'{name}: 观测契约不同')
                result = replay_episode(ep, env, raw, raw_episodes, args, destinations[name])
                report['replays'].append(result)
                # 每条完成即保存报告，后续失败仍能知道哪些输出已验证。
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
                print(f'{name}: 视频已解码验证；观测一致={result["matching_observations"]}', flush=True)
        completed = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        env.close()
        report['completed'] = completed
        report['all_replays_match'] = completed and all(r['matching_observations'] for r in report['replays'])
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'视频与报告：{output}', flush=True)
    if not report['all_replays_match']:
        raise RuntimeError('回放与保存观测存在偏差；视频已标记 MISMATCH，详情见 quality.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--control-mode', choices=CONTROL_CHOICES, help='选择ee/joint分支；默认ee')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--episodes', nargs='+', help='索引或 episode 完整名称；默认5条代表轨迹')
    selection.add_argument('--all', action='store_true', help='导出训练集全部轨迹，而非全部原始轨迹')
    parser.add_argument('--output', type=Path, help='默认 <数据集名称>-visualization；拒绝覆盖已有输出')
    parser.add_argument('--raw-source', type=Path, help='搬迁后的原始 .raw.h5；仍需匹配来源哈希')
    parser.add_argument('--inspect-only', action='store_true', help='只检查整个 HDF5 并生成 inspection.json；不需要 Open3D/仿真')
    parser.add_argument('--near-radius', type=float, default=1.0)
    parser.add_argument('--fps', type=float, help='默认环境控制频率；修改只影响播放速度')
    parser.add_argument('--state-tolerance', type=float, default=1e-4)
    parser.add_argument('--velocity-tolerance', type=float, default=1e-3)
    parser.add_argument('--pointcloud-tolerance', type=float, default=1e-4, help='策略特征单位；包含点索引顺序')
    args = parser.parse_args()
    for key in ('near_radius', 'fps', 'state_tolerance', 'velocity_tolerance', 'pointcloud_tolerance'):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f'{key} 必须为有限正数')
    run(args)


if __name__ == '__main__':
    main()
