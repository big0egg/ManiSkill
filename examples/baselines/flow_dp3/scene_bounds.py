"""五个任务的点云工作区边界：世界系转基座系后、减 TCP 前裁剪，单位米。

PickCube 的10条轨迹及其他任务各3条成功轨迹，共2402帧验证：
保留全部原始可见机械臂与任务目标点，移除地面（高度约 -0.920m）。
DrawTriangle 裁去部分外围空白画布，保留可见轮廓及墨迹。
边界不改变长度尺度、编码器邻域半径或 FPS 总点数；默认仍为512点。
这些是工作区预设，并非任意机器人姿态的全身包围盒。
"""

TASK_SCENE_CROP_BOUNDS = {
    "PickCube-v1": ((-0.2, -0.3, -0.05), (1.1, 0.3, 1.0)),
    "PushCube-v1": ((-0.2, -0.3, -0.05), (1.1, 0.3, 1.0)),
    "StackCube-v1": ((-0.2, -0.3, -0.05), (1.1, 0.3, 1.0)),
    "PegInsertionSide-v1": ((-0.2, -0.6, -0.05), (1.1, 0.6, 1.0)),
    "DrawTriangle-v1": ((-0.2, -0.5, -0.05), (1.1, 0.3, 1.0)),
}


def get_scene_crop_bounds(env_id: str) -> tuple[tuple, tuple]:
    """按任务取得基座系边界；未知任务需明确配置，不能套用 PickCube。"""
    try:
        return TASK_SCENE_CROP_BOUNDS[env_id]
    except KeyError:
        raise ValueError(f"未配置任务 {env_id!r} 的 crop；支持 {list(TASK_SCENE_CROP_BOUNDS)}") from None


PICKCUBE_SCENE_CROP_MIN, PICKCUBE_SCENE_CROP_MAX = get_scene_crop_bounds("PickCube-v1")
