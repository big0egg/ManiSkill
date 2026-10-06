"""五个任务的点云工作区边界：世界系转基座系后、减 TCP 前裁剪，单位米。

采用 testpointcloud 实验确定的操作区，优先目标与末端，允许裁掉部分机械臂。
相机与 crop 共用 mani_skill.utils.task_pointcloud 中的正式预设。
边界不改变长度尺度、编码器邻域半径或 FPS 总点数；默认仍为512点。
这些是工作区预设，并非任意机器人姿态的全身包围盒。
"""

from mani_skill.utils.task_pointcloud import TASK_POINTCLOUD_PRESETS

TASK_SCENE_CROP_BOUNDS = {
    name: (preset.crop_min, preset.crop_max) for name, preset in TASK_POINTCLOUD_PRESETS.items()
}


def get_scene_crop_bounds(env_id: str) -> tuple[tuple, tuple]:
    """按任务取得基座系边界；未知任务需明确配置，不能套用 PickCube。"""
    try:
        return TASK_SCENE_CROP_BOUNDS[env_id]
    except KeyError:
        raise ValueError(f"未配置任务 {env_id!r} 的 crop；支持 {list(TASK_SCENE_CROP_BOUNDS)}") from None


PICKCUBE_SCENE_CROP_MIN, PICKCUBE_SCENE_CROP_MAX = get_scene_crop_bounds("PickCube-v1")
