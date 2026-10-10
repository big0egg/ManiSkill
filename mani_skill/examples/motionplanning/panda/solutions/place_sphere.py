import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import PlaceSphereEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb

def solve(env: PlaceSphereEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    obb = get_actor_obb(env.obj)

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Search a valid pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2) + np.pi / 4
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")
        planner.close()
        return -1

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Stack
    # -------------------------------------------------------------------------- #
    goal_pose = env.bin.pose * sapien.Pose([0, 0, env.radius + env.block_half_size[0] + 0.003])
    offset = (goal_pose.p - env.obj.pose.p).cpu().numpy()[0] # remember that all data in ManiSkill is batched and a torch tensor
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    res = planner.move_to_pose_with_screw(align_pose)
    if res == -1: return res

    res = planner.open_gripper(t=10)
    retreat_pose = sapien.Pose([0, 0, 0.08]) * env.agent.tcp.pose.sp
    res = planner.move_to_pose_with_screw(retreat_pose)
    if res == -1: return res
    res = planner.open_gripper(t=20)
    planner.close()
    return res
