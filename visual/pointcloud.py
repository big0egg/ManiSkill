"""距离矢量点云：Open3D 交互与可选选点，Matplotlib/Open3D 导出 PNG/MP4。"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from dataset_io import DEFAULT_DATASET, episode_names, load_episode, pick_point, require_new, view_bounds


def import_open3d(headless=False):
    if headless:
        # 必须在首次 import Open3D 之前设置。
        os.environ.setdefault('EGL_PLATFORM', 'surfaceless')
        os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', 'true')
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError('请先执行 bash visual/setup_env.sh，再使用 .runtime/visual-venv/bin/python') from exc
    return o3d


def colors(distances, limit):
    from matplotlib import colormaps
    return colormaps['turbo'](np.clip(distances / limit, 0, 1))[:, :3]


class CloudView:
    """GUI 与离屏渲染共用几何体和固定相机。"""
    def __init__(self, scene, episode, args, o3d):
        self.scene, self.episode, self.args, self.o3d = scene, episode, args, o3d
        self.geometry_names = []
        self.limit = args.color_max or args.near_radius or float(episode.features[..., 3].max() * episode.scale)
        self.limit = max(self.limit, 1e-6)
        scene.set_background([0.04, 0.055, 0.08, 1])

    def material(self, shader, color=None):
        mat = self.o3d.visualization.rendering.MaterialRecord()
        mat.shader = shader
        mat.point_size = self.args.point_size
        mat.line_width = 1.0
        if color is not None:
            mat.base_color = color
        return mat

    def update(self, frame):
        o3d = self.o3d
        for name in self.geometry_names:
            self.scene.remove_geometry(name)
        self.geometry_names = []
        self.frame = frame
        ep = self.episode
        self.positions = ep.positions(frame, self.args.coordinate_frame)
        self.distances = ep.features[frame, :, 3] * ep.scale
        self.indices = np.arange(len(self.positions))
        if self.args.near_radius:
            self.indices = self.indices[self.distances <= self.args.near_radius]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self.positions[self.indices])
        cloud.colors = o3d.utility.Vector3dVector(colors(self.distances[self.indices], self.limit))
        if len(self.indices):
            self._add('cloud', cloud, self.material('defaultUnlit'))
        base, tcp, goal = ep.markers(frame, self.args.coordinate_frame)
        self.base, self.tcp, self.goal = base, tcp, goal
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=base)
        self._add('base_axes', axes, self.material('defaultUnlit'))
        for name, position, color in [('tcp', tcp, [0.1, 0.95, 0.95, 1]), ('goal', goal, [0.2, 1, 0.2, 1])]:
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.018)
            mesh.translate(position)
            self._add(name, mesh, self.material('defaultUnlit', color))
        count = min(self.args.vector_count, len(self.indices))
        if count:
            selected = self.indices[np.linspace(0, len(self.indices) - 1, count, dtype=int)]
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(np.vstack((tcp, self.positions[selected])))
            lines.lines = o3d.utility.Vector2iVector(np.column_stack((np.zeros(count, dtype=int), np.arange(1, count + 1))))
            lines.colors = o3d.utility.Vector3dVector(colors(self.distances[selected], self.limit) * 0.5)
            self._add('vectors', lines, self.material('unlitLine'))
        return cloud

    def _add(self, name, geometry, material):
        self.scene.add_geometry(name, geometry, material)
        self.geometry_names.append(name)

    def camera_bounds(self):
        return view_bounds(self.episode, self.args.coordinate_frame, self.args.near_radius)


class InteractiveViewer:
    def __init__(self, episode, args, o3d):
        from open3d.visualization import gui, rendering
        self.gui, self.args, self.o3d = gui, args, o3d
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window('Flow DP3 distance point cloud', 1280, 850)
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget)
        self.panel = gui.Vert(8, gui.Margins(10, 10, 10, 10))
        self.window.add_child(self.panel)
        self.panel.add_child(gui.Label('Episode'))
        chooser = gui.Combobox()
        self.names = episode_names(args.dataset)
        for name in self.names:
            chooser.add_item(name)
        chooser.selected_index = self.names.index(episode.name)
        chooser.set_on_selection_changed(self.change_episode)
        self.panel.add_child(chooser)
        self.status = gui.Label('')
        self.panel.add_child(self.status)
        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_on_value_changed(lambda value: self.show_frame(int(value)))
        self.panel.add_child(self.slider)
        buttons = gui.Horiz(5)
        for title, callback in [('Previous', lambda: self.step(-1)), ('Play / Pause', self.toggle), ('Next', lambda: self.step(1))]:
            button = gui.Button(title)
            button.set_on_clicked(callback)
            buttons.add_child(button)
        self.panel.add_child(buttons)
        reset = gui.Button('Reset camera')
        reset.set_on_clicked(self.reset_camera)
        self.panel.add_child(reset)
        self.panel.add_child(gui.Label('Base axes: X red / Y green / Z blue'))
        self.panel.add_child(gui.Label('TCP cyan / Goal green'))
        self.instructions = gui.Label('Click a point: coordinates\nDrag: rotate / wheel: zoom' if args.pick_points else 'Drag: rotate / wheel: zoom\nPoint picking is disabled')
        self.panel.add_child(self.instructions)
        self.info = gui.Label('')
        self.panel.add_child(self.info)
        self.playing = args.play
        self.last_tick = time.monotonic()
        self.mouse_down = None
        self.labels = []
        self.pick_label = None
        self.window.set_on_layout(self.layout)
        self.window.set_on_tick_event(self.tick)
        if args.pick_points:
            self.widget.set_on_mouse(self.on_mouse)
        self.set_episode(episode, args.frame)

    def layout(self, context):
        r = self.window.content_rect
        width = min(350, r.width // 2)
        self.widget.frame = self.gui.Rect(r.x, r.y, r.width - width, r.height)
        self.panel.frame = self.gui.Rect(r.x + r.width - width, r.y, width, r.height)

    def set_episode(self, episode, frame):
        if hasattr(self, 'view'):
            for name in self.view.geometry_names:
                self.widget.scene.remove_geometry(name)
        self.episode = episode
        self.view = CloudView(self.widget.scene, episode, self.args, self.o3d)
        self.slider.set_limits(0, len(episode.states) - 1)
        self.show_frame(frame)
        self.reset_camera()

    def change_episode(self, text, index):
        self.playing = False
        self.set_episode(load_episode(self.args.dataset, text), 0)

    def reset_camera(self):
        center, extent = self.view.camera_bounds()
        bounds = self.o3d.geometry.AxisAlignedBoundingBox(center - extent / 2, center + extent / 2)
        self.widget.setup_camera(55, bounds, center)
        self.widget.look_at(center, center + np.array([0.8, -1.3, 0.9]) * extent, [0, 0, 1])

    def show_frame(self, frame):
        self.frame = max(0, min(frame, len(self.episode.states) - 1))
        self.view.update(self.frame)
        self.slider.int_value = self.frame
        self.status.text = f'Frame {self.frame}/{len(self.episode.actions)}\nPoints: {len(self.view.indices)}/{len(self.view.positions)}\nColor: 0 to {self.view.limit:.3f} m\nView: {self.args.coordinate_frame} (meters)'
        self.info.text = ''
        for label in self.labels:
            self.widget.remove_3d_label(label)
        self.labels = []
        self.pick_label = None
        for position, text in [(self.view.base, 'BASE'), (self.view.tcp, 'TCP'), (self.view.goal, 'GOAL')]:
            self.labels.append(self.widget.add_3d_label(position, text))
        self.widget.force_redraw()

    def toggle(self):
        self.playing = not self.playing
        self.last_tick = time.monotonic()

    def step(self, delta):
        self.playing = False
        self.show_frame(self.frame + delta)

    def tick(self):
        if self.playing and time.monotonic() - self.last_tick >= 1 / self.args.fps:
            self.last_tick = time.monotonic()
            self.show_frame((self.frame + 1) % len(self.episode.states))
            return True
        return False

    def on_mouse(self, event):
        gui = self.gui
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT):
            self.mouse_down = (event.x, event.y)
            self.playing = False
        elif event.type == gui.MouseEvent.Type.BUTTON_UP and self.mouse_down is not None:
            start = self.mouse_down
            self.mouse_down = None
            if math.hypot(event.x - start[0], event.y - start[1]) <= 4:
                r = self.widget.frame
                camera = self.widget.scene.camera
                index = pick_point(self.view.positions[self.view.indices], camera.get_view_matrix(),
                                   camera.get_projection_matrix(), r.width, r.height,
                                   event.x - r.x, event.y - r.y, self.args.point_size / 2 + 4)
                if index is None:
                    self.info.text = 'No point at this position.'
                else:
                    source_index = int(self.view.indices[index])
                    detail = self.episode.point_info(self.frame, source_index)
                    fmt = lambda values: ', '.join(f'{v:.6f}' for v in values)
                    self.info.text = (f'Point #{source_index} / frame {self.frame}\n'
                                      f'Base xyz (m):\n{fmt(detail["base_xyz_m"])}\n'
                                      f'Relative xyz (m):\n{fmt(detail["relative_xyz_m"])}\n'
                                      f'Distance: {detail["distance_m"]:.6f} m\n'
                                      f'Policy [dx,dy,dz,d]:\n{fmt(detail["policy_features"])}')
                    if self.pick_label is not None:
                        self.widget.remove_3d_label(self.pick_label)
                        self.labels.remove(self.pick_label)
                    self.pick_label = self.widget.add_3d_label(self.view.positions[source_index], f'#{source_index}')
                    self.labels.append(self.pick_label)
                    print(json.dumps(detail, ensure_ascii=False), flush=True)
                self.window.set_needs_layout()
        # 让相机控制器同时接收完整的按下/释放事件，拖动继续旋转。
        return gui.Widget.EventCallbackResult.IGNORED

    def run(self):
        self.app.run()


class OffscreenCloud:
    def __init__(self, episode, args, width=512, height=512):
        o3d = import_open3d(headless=True)
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self.view = CloudView(self.renderer.scene, episode, args, o3d)
        self.view.update(0)
        center, extent = self.view.camera_bounds()
        self.renderer.setup_camera(55, center, center + np.array([0.8, -1.3, 0.9]) * extent, [0, 0, 1])

    def render(self, frame):
        self.view.update(frame)
        return np.asarray(self.renderer.render_to_image()).copy()


def export(episode, args):
    # 先检查全部输出，避免失败后覆盖已有文件。
    paths = [require_new(p) for p in (args.png, args.video, args.ply) if p is not None]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('PNG、MP4 和 PLY 必须使用不同输出路径')
    if args.ply:
        o3d = import_open3d(headless=True)
        cloud = o3d.geometry.PointCloud()
        xyz = episode.positions(args.frame, args.coordinate_frame)
        distance = episode.features[args.frame, :, 3] * episode.scale
        mask = np.ones(len(xyz), dtype=bool) if not args.near_radius else distance <= args.near_radius
        cloud.points = o3d.utility.Vector3dVector(xyz[mask])
        limit = args.color_max or args.near_radius or float(episode.features[..., 3].max() * episode.scale)
        cloud.colors = o3d.utility.Vector3dVector(colors(distance[mask], max(limit, 1e-6)))
        if not o3d.io.write_point_cloud(str(args.ply), cloud):
            raise RuntimeError(f'PLY 写入失败：{args.ply}')
    if args.png or args.video:
        from PIL import Image, ImageDraw
        if args.export_backend == 'matplotlib':
            from matplotlib_cloud import MatplotlibCloud
            renderer = MatplotlibCloud(episode, args)
        else:
            renderer = OffscreenCloud(episode, args, 768, 768)
        def image(frame):
            result = Image.fromarray(renderer.render(frame))
            if args.export_backend == 'open3d':
                ImageDraw.Draw(result).text((12, 12), f'{episode.name} | frame {frame} | {args.coordinate_frame}\nBASE: RGB axes | TCP: cyan | GOAL: green\nDistance color: 0 to {renderer.view.limit:.3f} m', fill='white')
            return np.asarray(result)
        try:
            if args.png:
                Image.fromarray(image(args.frame)).save(args.png)
            if args.video:
                import imageio.v2 as imageio
                with imageio.get_writer(str(args.video), fps=args.fps, codec='libx264', quality=8, macro_block_size=1) as writer:
                    for frame in range(args.frame, len(episode.states)):
                        writer.append_data(image(frame))
        finally:
            if args.export_backend == 'matplotlib':
                renderer.close()
        del renderer
    for path in paths:
        print(f'已保存：{path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--episode', default='0', help='episode 索引或完整名称')
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--pick-points', action='store_true', help='点击实际点显示基座坐标、相对位移、距离与原始索引')
    parser.add_argument('--coordinate-frame', choices=['base', 'relative'], default='base')
    parser.add_argument('--near-radius', type=float, help='仅展示距 TCP 小于此距离的点（米）；不修改数据')
    parser.add_argument('--vector-count', type=int, default=24, help='显示从 TCP 到点的矢量线数量；0关闭，512显示全部')
    parser.add_argument('--point-size', type=float, default=6)
    parser.add_argument('--color-max', type=float, help='固定色标上限（米）；默认整条轨迹最大距离')
    parser.add_argument('--fps', type=float, default=20)
    parser.add_argument('--play', action='store_true', help='GUI 启动后自动播放')
    parser.add_argument('--export-backend', choices=['matplotlib', 'open3d'], default='matplotlib',
                        help='PNG/MP4 后端：默认 CPU/Agg，无需 Open3D 或桌面；GUI/PLY 仍使用 Open3D')
    parser.add_argument('--png', type=Path, help='离屏导出指定帧 PNG')
    parser.add_argument('--video', type=Path, help='离屏导出从指定帧开始的点云 MP4')
    parser.add_argument('--ply', type=Path, help='导出指定帧距离着色点云 PLY')
    args = parser.parse_args()
    for key in ('fps', 'point_size', 'near_radius', 'color_max'):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f'{key} 必须为有限正数')
    if args.vector_count < 0:
        parser.error('vector-count 不得为负数')
    ep = load_episode(args.dataset, args.episode)
    if not 0 <= args.frame < len(ep.states):
        parser.error(f'frame 必须在 0 到 {len(ep.states) - 1} 之间')
    headless = any(p is not None for p in (args.png, args.video, args.ply))
    if headless and args.pick_points:
        parser.error('--pick-points 需要交互窗口，请不要同时指定离屏导出参数')
    if headless:
        export(ep, args)
    else:
        if sys.platform.startswith('linux') and not os.getenv('DISPLAY') and not os.getenv('WAYLAND_DISPLAY'):
            parser.error('当前终端没有桌面显示。请在桌面/X11会话运行，或使用 --png / --video / --ply 离屏导出。')
        InteractiveViewer(ep, args, import_open3d()).run()


if __name__ == '__main__':
    main()
