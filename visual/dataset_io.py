"""只读加载 Flow DP3 观测；坐标转换与点选择不依赖图形环境。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / '.runtime/flow_dp3/pickcube-100-camera-v5.h5'


@dataclass
class Episode:
    name: str
    features: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    metadata: dict
    contract: dict

    @property
    def scale(self):
        return float(self.contract['pointcloud']['length_scale'])

    @property
    def tcp(self):
        lo, hi = self.contract['state_fields']['tcp_base_pose_wxyz']
        return self.states[:, lo:hi][:, :3]

    @property
    def goals(self):
        lo, hi = self.contract['state_fields']['goal_base_pos']
        return self.states[:, lo:hi]

    def positions(self, frame, coordinate_frame='base'):
        relative = self.features[frame, :, :3].astype(np.float64) * self.scale
        return relative + self.tcp[frame] if coordinate_frame == 'base' else relative

    def markers(self, frame, coordinate_frame='base'):
        tcp = self.tcp[frame].astype(np.float64)
        goal = self.goals[frame].astype(np.float64)
        return (np.zeros(3), tcp, goal) if coordinate_frame == 'base' else (-tcp, np.zeros(3), goal - tcp)

    def point_info(self, frame, index):
        relative = self.positions(frame, 'relative')[index]
        return {'episode': self.name, 'frame': int(frame), 'point_index': int(index),
                'base_xyz_m': self.positions(frame, 'base')[index].tolist(),
                'relative_xyz_m': relative.tolist(),
                'distance_m': float(self.features[frame, index, 3] * self.scale),
                'policy_features': self.features[frame, index].tolist()}


def episode_names(path):
    with h5py.File(path, 'r') as f:
        return sorted(f.keys())


def view_bounds(episode, coordinate_frame='base', near_radius=None):
    """整条轨迹共享的相机范围，包含点云、基座轴、末端和目标。"""
    xyz = episode.features[..., :3].astype(np.float64) * episode.scale
    if coordinate_frame == 'base':
        xyz += episode.tcp[:, None, :]
        markers = np.vstack((np.zeros((1, 3)), episode.tcp, episode.goals))
    else:
        markers = np.vstack((-episode.tcp, np.zeros((1, 3)), episode.goals - episode.tcp))
    xyz = xyz.reshape(-1, 3)
    if near_radius is not None:
        mask = episode.features[..., 3].ravel() * episode.scale <= near_radius
        xyz = xyz[mask]
    all_xyz = np.vstack((xyz, markers, markers + [.25, .25, .25]))
    low, high = all_xyz.min(axis=0), all_xyz.max(axis=0)
    return ((low + high) / 2).astype(np.float32), max(float((high - low).max()), .5)


def load_episode(path, name='0'):
    with h5py.File(path, 'r') as f:
        names = sorted(f.keys())
        if str(name).isdigit():
            index = int(name)
            if not 0 <= index < len(names):
                raise ValueError(f'episode 索引越界：{index}，共有 {len(names)} 条')
            name = names[index]
        if name not in f:
            raise ValueError(f'不存在 episode：{name}')
        g = f[name]
        episode = Episode(name, g['pointcloud_distance'][:], g['state'][:], g['action'][:],
                          json.loads(g.attrs['metadata']), json.loads(f.attrs['contract']))
    c = episode.contract
    if c['frame'] != 'robot_base' or c['distance_channels'] != '[relative_xyz_m, norm_m] / length_scale':
        raise ValueError('不支持的数据坐标/距离契约')
    t = len(episode.actions)
    if episode.features.shape != (t + 1, c['pointcloud']['num_points'], 4) or episode.states.shape != (t + 1, 28) or episode.actions.shape != (t, 4):
        raise ValueError(f'{name}: T+1 观测 / T 动作形状异常')
    if not np.isfinite(episode.scale) or episode.scale <= 0:
        raise ValueError('length_scale 必须为有限正数')
    if not all(np.isfinite(x).all() for x in (episode.features, episode.states, episode.actions)):
        raise ValueError(f'{name}: 包含 NaN/Inf')
    if np.max(np.abs(np.linalg.norm(episode.features[..., :3], axis=-1) - episode.features[..., 3])) > 1e-5:
        raise ValueError(f'{name}: 距离通道与矢量不一致')
    if np.abs(episode.actions).max(initial=0) > 1.0001:
        raise ValueError(f'{name}: 动作超出 [-1,1]')
    return episode


def project_points(points, view, projection, width, height):
    """OpenGL 相机投影；保留原点索引，过滤相机后方与裁剪范围外的点。"""
    points = np.asarray(points)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    clip = homogeneous @ np.asarray(view).T @ np.asarray(projection).T
    valid = clip[:, 3] > 0
    ndc = np.full((len(points), 3), np.nan)
    ndc[valid] = clip[valid, :3] / clip[valid, 3:4]
    valid &= (np.abs(ndc) <= 1).all(axis=1)
    pixels = np.column_stack(((ndc[:, 0] + 1) * width / 2, (1 - ndc[:, 1]) * height / 2))
    return pixels, (ndc[:, 2] + 1) / 2, valid


def pick_point(points, view, projection, width, height, x, y, radius=8):
    """选择点击附近最靠近相机的实际数据点，不返回插值后的表面坐标。"""
    pixels, depths, valid = project_points(points, view, projection, width, height)
    distances = np.linalg.norm(pixels - [x, y], axis=1)
    candidates = np.flatnonzero(valid & (distances <= radius))
    if not len(candidates):
        return None
    # 相近像素中的前景点优先，避免选到被遮挡的背景点。
    order = np.lexsort((distances[candidates], depths[candidates]))
    return int(candidates[order[0]])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def default_output(dataset):
    p = Path(dataset).resolve()
    return p.parent / (p.stem + '-visualization')


def require_new(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'输出已存在：{path}；请选择新路径')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
