"""Panda task point-cloud presets checked with task demonstrations.

The registry imports no rendering libraries until a camera is requested.
Camera positions are world coordinates; crop bounds are robot-base coordinates,
in metres. These operation regions prioritize objects and the end effector.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TaskPointCloudPreset:
    eye: tuple
    target: tuple
    fov_degrees: float
    crop_min: tuple
    crop_max: tuple
    width: int = 256
    height: int = 256


TASK_POINTCLOUD_PRESETS = {
    "PickCube-v1": TaskPointCloudPreset((.30, -.30, .35), (0, 0, .12), 60,
                                       (.44, -.23, -.03), (.79, .25, .52)),
    "PushCube-v1": TaskPointCloudPreset((.30, -.30, .35), (.15, 0, .08), 75,
                                       (.40, -.23, -.03), (1.06, .25, .52)),
    "StackCube-v1": TaskPointCloudPreset((.30, .35, .28), (0, 0, .07), 65,
                                        (.36, -.36, -.03), (.87, .36, .52)),
    "PegInsertionSide-v1": TaskPointCloudPreset((.30, -.35, .55), (0, .10, .12), 75,
                                               (.34, -.40, -.03), (.90, .62, .52)),
    "DrawTriangle-v1": TaskPointCloudPreset((.25, -.40, .50), (-.10, -.10, .04), 60,
                                           (.28, -.35, -.03), (.80, .18, .52)),
    "LiftPegUpright-v1": TaskPointCloudPreset((.30, -.35, .50), (0, 0, .15), 65,
                                             (.32, -.30, -.03), (.92, .30, .60)),
    "PlaceSphere-v1": TaskPointCloudPreset((.25, -.30, .35), (0, 0, .04), 60,
                                           (.43, -.23, -.03), (.83, .23, .45)),
    "PullCube-v1": TaskPointCloudPreset((.30, .30, .35), (-.12, 0, .06), 65,
                                        (.20, -.25, -.03), (.85, .25, .45)),
}


def get_task_pointcloud_preset(env_id):
    try:
        return TASK_POINTCLOUD_PRESETS[env_id]
    except KeyError:
        raise ValueError(f"未配置任务 {env_id!r} 的点云预设；支持 {list(TASK_POINTCLOUD_PRESETS)}") from None


def make_pointcloud_camera_config(env_id, *, eye=None, target=None):
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils.sapien_utils import look_at
    preset = get_task_pointcloud_preset(env_id)
    pose = look_at(eye=eye if eye is not None else preset.eye,
                   target=target if target is not None else preset.target)
    return CameraConfig("base_camera", pose, preset.width, preset.height,
                        math.radians(preset.fov_degrees), .01, 100)


def pointcloud_sensor_configs(env_id="PickCube-v1", *, legacy_pickcube=False):
    """Explicit, JSON-serializable camera settings for data/checkpoint contracts."""
    if legacy_pickcube:
        from mani_skill.sensors.camera import CameraConfig
        from mani_skill.utils.sapien_utils import look_at
        camera = CameraConfig("base_camera", look_at(eye=(.3, 0, .6), target=(-.1, 0, .1)),
                              128, 128, math.pi / 2, .01, 100)
    else:
        camera = make_pointcloud_camera_config(env_id)
    pose = camera.pose.raw_pose.detach().cpu().reshape(-1).tolist()
    return {"shader_pack": "default", "base_camera": {
        "pose": pose, "width": camera.width, "height": camera.height,
        "fov": camera.fov, "near": camera.near, "far": camera.far}}
