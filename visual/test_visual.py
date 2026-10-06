"""验证缩放、坐标系、实际点选取和只读数据边界。"""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from dataset_io import Episode, load_episode, pick_point, require_new, view_bounds


class CoordinateTests(unittest.TestCase):
    def setUp(self):
        features = np.array([[[1., 0., 0., 1.], [0., 2., 0., 2.]]], dtype=np.float32)
        states = np.zeros((1, 28), dtype=np.float32)
        states[0, 18:21] = [3, 4, 5]
        states[0, 25:28] = [6, 7, 8]
        self.ep = Episode('episode_00000', features, states, np.empty((0, 4)), {},
                          {'pointcloud': {'length_scale': 2},
                           'state_fields': {'tcp_base_pose_wxyz': [18, 25], 'goal_base_pos': [25, 28]}})

    def test_scaled_features_reconstruct_base_and_relative_coordinates(self):
        np.testing.assert_allclose(self.ep.positions(0), [[5, 4, 5], [3, 8, 5]])
        np.testing.assert_allclose(self.ep.positions(0, 'relative'), [[2, 0, 0], [0, 4, 0]])
        info = self.ep.point_info(0, 1)
        self.assertEqual(info['point_index'], 1)
        self.assertEqual(info['base_xyz_m'], [3, 8, 5])
        self.assertEqual(info['relative_xyz_m'], [0, 4, 0])
        self.assertEqual(info['distance_m'], 4.)
        self.assertEqual(info['policy_features'], [0, 2, 0, 2])

    def test_relative_view_keeps_base_origin_at_correct_position(self):
        base, tcp, goal = self.ep.markers(0, 'relative')
        np.testing.assert_equal(base, [-3, -4, -5])
        np.testing.assert_equal(tcp, [0, 0, 0])
        np.testing.assert_equal(goal, [3, 3, 3])

    def test_click_prefers_foreground_and_rejects_background(self):
        points = np.array([[0, 0, .8], [0, 0, -.5], [2, 0, 0]])
        self.assertEqual(pick_point(points, np.eye(4), np.eye(4), 200, 100, 100, 50), 1)
        self.assertIsNone(pick_point(points, np.eye(4), np.eye(4), 200, 100, 20, 10))

    def test_perspective_rejects_point_behind_camera(self):
        projection = np.eye(4)
        projection[3] = [0, 0, -1, 0]
        points = np.array([[0, 0, 1.], [0, 0, -1.]])
        self.assertEqual(pick_point(points, np.eye(4), projection, 200, 100, 100, 50), 1)


class FileTests(unittest.TestCase):
    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'existing.mp4'
            p.write_bytes(b'existing')
            with self.assertRaises(FileExistsError):
                require_new(p)
            self.assertEqual(p.read_bytes(), b'existing')

    def test_misaligned_observation_length_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'bad.h5'
            with h5py.File(p, 'w') as f:
                f.attrs['contract'] = json.dumps({'frame': 'robot_base',
                    'distance_channels': '[relative_xyz_m, norm_m] / length_scale',
                    'pointcloud': {'num_points': 2, 'length_scale': 1}})
                g = f.create_group('episode_00000')
                g.attrs['metadata'] = '{}'
                g.create_dataset('action', data=np.zeros((2, 4)))
                g.create_dataset('state', data=np.zeros((2, 28)))
                g.create_dataset('pointcloud_distance', data=np.zeros((3, 2, 4)))
            with self.assertRaisesRegex(ValueError, '形状异常'):
                load_episode(p)


class ExportTests(unittest.TestCase):
    def setUp(self):
        features = np.tile(np.array([[[.2, 0, 0, .2], [0, .3, 0, .3]]], dtype=np.float32), (3, 1, 1))
        states = np.zeros((3, 28), dtype=np.float32)
        states[:, 18:21] = [[.4, 0, .2], [.5, 0, .2], [.6, 0, .2]]
        states[:, 25:28] = [.7, .1, .3]
        self.ep = Episode('episode_00000', features, states, np.zeros((2, 4)), {},
                          {'pointcloud': {'length_scale': 2},
                           'state_fields': {'tcp_base_pose_wxyz': [18, 25], 'goal_base_pos': [25, 28]}})
        self.args = SimpleNamespace(export_backend='matplotlib', export_view='both', coordinate_frame='base', frame=1,
                                    near_radius=None, color_max=None, vector_count=2, point_size=6,
                                    fps=10, ply=None, png=None, video=None)

    def test_png_mp4_export_without_open3d_or_display_includes_terminal_frame(self):
        import imageio.v2 as imageio
        from PIL import Image
        from pointcloud import export
        with tempfile.TemporaryDirectory() as directory:
            self.args.png = Path(directory) / 'frame.png'
            self.args.video = Path(directory) / 'cloud.mp4'
            with patch('pointcloud.import_open3d', side_effect=AssertionError('服务器导出不得导入 Open3D')), \
                 patch.dict(os.environ, {'DISPLAY': '', 'WAYLAND_DISPLAY': ''}):
                export(self.ep, self.args)
            with Image.open(self.args.png) as im:
                self.assertEqual(im.size, (1920, 768))
                self.assertGreater(np.asarray(im).max() - np.asarray(im).min(), 200)
            with imageio.get_reader(str(self.args.video)) as reader:
                self.assertEqual(reader.get_meta_data()['fps'], 10)
                frames = list(reader.iter_data())
            self.assertEqual(len(frames), 2)  # 起始帧1和终止观测帧2。
            self.assertEqual(frames[0].shape, (768, 1920, 3))

    def test_dual_views_show_same_fps_points_and_only_distance_view_has_vectors(self):
        from matplotlib_cloud import MatplotlibCloud
        renderer = MatplotlibCloud(self.ep, self.args)
        try:
            for radius, expected in [(None, [[.9, 0, .2], [.5, .6, .2]]),
                                     (.5, [[.9, 0, .2]])]:
                self.args.near_radius = radius
                renderer.render(1)
                for kind in ('fps', 'distance'):
                    plotted = np.column_stack(renderer.point_artists[kind]._offsets3d)
                    np.testing.assert_allclose(plotted, expected, atol=1e-7)
                self.assertIsNone(renderer.point_artists['fps'].get_array())
                np.testing.assert_allclose(renderer.point_artists['distance'].get_array(),
                                           [.4, .6] if radius is None else [.4])
                left, right = renderer.axes['fps'], renderer.axes['distance']
                for axis in ('x', 'y', 'z'):
                    self.assertEqual(getattr(left, 'get_' + axis + 'lim')(),
                                     getattr(right, 'get_' + axis + 'lim')())
                np.testing.assert_allclose(left.get_proj(), right.get_proj())
                # 两图均有点、三个标记及三个基座轴；右图另有一个矢量集合。
                self.assertEqual(len(right.collections), len(left.collections) + 1)
        finally:
            renderer.close()

    def test_single_view_options_keep_original_export_size(self):
        from matplotlib_cloud import MatplotlibCloud
        for kind in ('fps', 'distance'):
            with self.subTest(view=kind):
                self.args.export_view = kind
                renderer = MatplotlibCloud(self.ep, self.args)
                try:
                    self.assertEqual(renderer.render(0).shape, (768, 960, 3))
                    self.assertEqual(list(renderer.axes), [kind])
                finally:
                    renderer.close()

    def test_empty_near_filter_and_relative_frame_keep_camera_and_color_range(self):
        from matplotlib_cloud import MatplotlibCloud
        self.args.coordinate_frame = 'relative'
        self.args.near_radius = .1  # 所有点距离均超过范围，仍应显示基座/末端/目标。
        renderer = MatplotlibCloud(self.ep, self.args, width=320, height=320)
        try:
            before = (renderer.ax.get_xlim(), renderer.ax.get_ylim(), renderer.ax.get_zlim())
            for frame in range(len(self.ep.states)):
                image = renderer.render(frame)
                self.assertEqual(image.shape, (320, 320, 3))
                self.assertIn('Points: 0/2', renderer.subtitle.get_text())
                self.assertEqual(before, (renderer.ax.get_xlim(), renderer.ax.get_ylim(), renderer.ax.get_zlim()))
                self.assertEqual(renderer.limit, .1)
        finally:
            renderer.close()
        center, extent = view_bounds(self.ep, 'relative', .1)
        self.assertTrue(np.isfinite(center).all())
        self.assertGreaterEqual(extent, .5)


if __name__ == '__main__':
    unittest.main()
