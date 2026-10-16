"""OBB grasping expert for the fixed sugar-box FlowDP3 task."""
import numpy as np
import sapien

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb,
)
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver


def object_goal_tcp_pose(tcp_pose, object_pose, goal_position):
    """Move the object origin to the goal while preserving the measured grasp."""
    tcp_to_object = tcp_pose.inv() * object_pose
    object_goal = sapien.Pose(np.asarray(goal_position), object_pose.q)
    return object_goal * tcp_to_object.inv()


def solve(env, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    base = env.unwrapped
    planner = PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis, base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=vis, print_env_info=False,
        joint_vel_limits=0.75, joint_acc_limits=0.75,
    )
    try:
        obb = get_actor_obb(base.obj)
        closing = base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp = compute_grasp_info_by_obb(
            obb, approaching=np.array([0, 0, -1]), target_closing=closing, depth=0.03,
        )
        # Panda's full opening is 8 cm; leave room to approach without squeezing.
        if grasp["extents"][1] > 0.075:
            return -1
        grasp_pose = base.agent.build_grasp_pose([0, 0, -1], grasp["closing"], grasp["center"])
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.08])
        for pose in (reach_pose, grasp_pose):
            result = planner.move_to_pose_with_screw(pose)
            if isinstance(result, int):
                return -1
        planner.close_gripper(t=12)
        if not bool(base.agent.is_grasping(base.obj).item()):
            return -1
        current = base.agent.tcp.pose.sp
        lifted = sapien.Pose(current.p + np.array([0, 0, 0.12]), current.q)
        result = planner.move_to_pose_with_screw(lifted)
        if isinstance(result, int) or not bool(base.agent.is_grasping(base.obj).item()):
            return -1
        target = object_goal_tcp_pose(
            base.agent.tcp.pose.sp, base.obj.pose.sp, base.goal_site.pose.sp.p,
        )
        result = planner.move_to_pose_with_screw(target)
        if isinstance(result, int):
            return -1
        # Real simulation steps settle the arm; the box remains grasped.
        return planner.close_gripper(t=20)
    finally:
        planner.close()
