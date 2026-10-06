import mplib
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.base_motionplanner.motionplanner import BaseMotionPlanningSolver


class PandaStickMotionPlanningSolver(BaseMotionPlanningSolver):
    MOVE_GROUP = "panda_hand_tcp"

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        # Stick has no grasp visual; BaseMotionPlanningSolver does not accept it.
        super().__init__(env, debug=debug, vis=vis, base_pose=base_pose,
                         print_env_info=print_env_info, joint_vel_limits=joint_vel_limits,
                         joint_acc_limits=joint_acc_limits)
