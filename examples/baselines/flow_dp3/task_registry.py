"""First-batch task interfaces; presets alone do not imply tested support."""
from dataclasses import dataclass
import math
from mani_skill.utils.task_pointcloud import pointcloud_sensor_configs


@dataclass(frozen=True)
class TaskSpec:
    robot: str
    control_mode: str
    joints: int
    action_dim: int
    max_steps: int
    solver: str

    @property
    def state_fields(self):
        n = self.joints
        fields = {"qpos": [0, n], "qvel": [n, 2*n], "tcp_base_pose_wxyz": [2*n, 2*n+7]}
        if self.solver == "pick_cube":
            fields["goal_base_pos"] = [2*n+7, 2*n+10]
        return fields

    @property
    def state_dim(self):
        return max(end for _, end in self.state_fields.values())


TASKS = {
    "PickCube-v1": TaskSpec("panda", "pd_ee_delta_pos", 9, 4, 200, "pick_cube"),
    "PushCube-v1": TaskSpec("panda", "pd_ee_delta_pos", 9, 4, 200, "push_cube"),
    "StackCube-v1": TaskSpec("panda", "pd_ee_delta_pose", 9, 7, 400, "stack_cube"),
    "PegInsertionSide-v1": TaskSpec("panda_wristcam", "pd_ee_delta_pose", 9, 7, 500, "peg_insertion_side"),
    "DrawTriangle-v1": TaskSpec("panda_stick", "pd_ee_delta_pos", 7, 3, 300, "draw_triangle"),
}


def get_task(env_id):
    try:
        return TASKS[env_id]
    except KeyError:
        raise ValueError(f"FlowDP3 尚未接入任务 {env_id!r}；可选 {list(TASKS)}") from None


def task_sensors(env_id):
    sensors = pointcloud_sensor_configs(env_id)
    if get_task(env_id).robot == "panda_wristcam":
        sensors["hand_camera"] = {"pose": [0, 0, 0, 1, 0, 0, 0], "width": 128,
                                  "height": 128, "fov": math.pi/2, "near": .01,
                                  "far": 100, "entity_uid": "camera_link"}
    return sensors
