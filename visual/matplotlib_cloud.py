"""CPU/Agg 点云图片渲染，不导入 Open3D 或仿真环境。"""
from __future__ import annotations

import os

import numpy as np

from dataset_io import ROOT, view_bounds


class MatplotlibCloud:
    def __init__(self, episode, args, width=960, height=768):
        os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime/matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import Normalize
        from matplotlib.figure import Figure
        from matplotlib.cm import ScalarMappable

        self.episode, self.args = episode, args
        self.limit = max(args.color_max or args.near_radius or float(episode.features[..., 3].max() * episode.scale), 1e-6)
        self.figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor='#101725')
        self.canvas = FigureCanvasAgg(self.figure)
        self.ax = self.figure.add_axes([.02, .12, .78, .76], projection='3d')
        self.ax.set_facecolor('#101725')
        self.ax.tick_params(colors='white', labelsize=9)
        for direction in ('x', 'y', 'z'):
            axis = getattr(self.ax, direction + 'axis')
            axis.label.set_color('white')
            axis.set_pane_color((.075, .1, .16, 1))
        self.ax.set_xlabel('X (m)'); self.ax.set_ylabel('Y (m)'); self.ax.set_zlabel('Z (m)')
        self.ax.view_init(elev=30, azim=-55)
        self.ax.set_box_aspect([1, 1, 1])
        center, extent = view_bounds(episode, args.coordinate_frame, args.near_radius)
        half = extent * .55
        self.ax.set_xlim(center[0] - half, center[0] + half)
        self.ax.set_ylim(center[1] - half, center[1] + half)
        self.ax.set_zlim(center[2] - half, center[2] + half)
        # 色标与相机范围在整段视频中保持固定。
        self.cmap = matplotlib.colormaps['turbo']
        self.norm = Normalize(vmin=0, vmax=self.limit)
        color_ax = self.figure.add_axes([.84, .27, .024, .48])
        colorbar = self.figure.colorbar(ScalarMappable(norm=self.norm, cmap=self.cmap), cax=color_ax)
        colorbar.set_label('Distance to TCP (m)', color='white', fontsize=11)
        colorbar.ax.tick_params(colors='white', labelsize=9)
        self.title = self.figure.text(.04, .96, '', color='white', fontsize=14, weight='bold')
        self.subtitle = self.figure.text(.04, .925, '', color='#d4deeb', fontsize=10)
        self.figure.text(.04, .045, 'BASE: white +   TCP: cyan   GOAL: green star\nBase axes: X red / Y green / Z blue; vectors start at TCP',
                         color='#d4deeb', fontsize=10)
        self.artists = []
        # 全场景的尺度较大，额外提供方向示意，基座原点仍标在主图中。
        inset = self.figure.add_axes([.015, .14, .16, .16], projection='3d')
        inset.set_facecolor('#101725')
        inset.set_axis_off()
        inset.view_init(elev=30, azim=-55)
        inset.set_box_aspect([1, 1, 1])
        inset.set_xlim(-.1, 1.25); inset.set_ylim(-.1, 1.25); inset.set_zlim(-.1, 1.25)
        for axis, label, color in zip(np.eye(3), 'XYZ', ['#ff5555', '#55ee55', '#5599ff']):
            inset.quiver(0, 0, 0, *axis, color=color, arrow_length_ratio=.15)
            inset.text(*(axis * 1.1), label, color=color, fontsize=10)
        inset.set_title('Base orientation', color='#d4deeb', fontsize=8)

    def render(self, frame):
        for artist in self.artists:
            artist.remove()
        self.artists = []
        ep, args = self.episode, self.args
        xyz = ep.positions(frame, args.coordinate_frame)
        distance = ep.features[frame, :, 3] * ep.scale
        indices = np.arange(len(xyz))
        if args.near_radius is not None:
            indices = indices[distance <= args.near_radius]
        points = xyz[indices]
        self.artists.append(self.ax.scatter(*points.T, c=distance[indices], cmap=self.cmap, norm=self.norm,
                                           s=(args.point_size * .72) ** 2, depthshade=False))
        base, tcp, goal = ep.markers(frame, args.coordinate_frame)
        for position, label, color, marker in [(base, 'BASE', 'white', '+'), (tcp, 'TCP', 'cyan', 'o'), (goal, 'GOAL', '#55ff55', '*')]:
            self.artists.append(self.ax.scatter(*position, c=color, marker=marker, s=55))
            self.artists.append(self.ax.text(*(position + [0, 0, .04]), label, color=color, fontsize=9))
        for axis, color in zip(np.eye(3) * .25, ['#ff5555', '#55ee55', '#5599ff']):
            self.artists.append(self.ax.quiver(*base, *axis, color=color, arrow_length_ratio=.2, linewidth=1.5))
        count = min(args.vector_count, len(indices))
        if count:
            selected = indices[np.linspace(0, len(indices) - 1, count, dtype=int)]
            origins = np.repeat(tcp[None, :], count, axis=0)
            vectors = xyz[selected] - tcp
            rgba = self.cmap(self.norm(distance[selected]))
            rgba[:, 3] = .5
            self.artists.append(self.ax.quiver(*origins.T, *vectors.T, color=rgba, normalize=False,
                                              length=1, arrow_length_ratio=.035, linewidth=.7))
        suffix = '' if args.near_radius is None else f' | within {args.near_radius:g} m of TCP'
        self.title.set_text(f'{ep.name} | frame {frame}/{len(ep.actions)} | {args.coordinate_frame} frame')
        self.subtitle.set_text(f'Points: {len(indices)}/{len(xyz)} | vectors: {count} | color: 0 to {self.limit:.3f} m{suffix}')
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        self.figure.clear()
