import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import LiftPegUprightEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb

def main():
    env: LiftPegUprightEnv = gym.make(
        "LiftPegUpright-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    for seed in range(100):
        res = solve(env, seed=seed, debug=False, vis=True)
        print(res[-1])
    env.close()

def solve(env: LiftPegUprightEnv, seed=None, debug=False, vis=False):
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
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
    
    env = env.unwrapped
    FINGER_LENGTH = 0.025

    obb = get_actor_obb(env.peg)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    peg_init_pose = env.peg.pose.sp

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

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
    lift_pose = sapien.Pose([0, 0, 0.25]) * grasp_pose
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Place upright
    # -------------------------------------------------------------------------- #
    # Use the measured grasp transform to rotate the peg's long X axis upright.
    # A quaternion with cos(theta)/sin(theta) rotates by 2*theta, not theta.
    peg_to_tcp = (env.peg.pose.inv() * env.agent.tcp.pose).sp
    # Both ends and any yaw are valid. Search a reachable wrist orientation
    # rather than forcing a single 90-degree rotation through a joint limit.
    best = None
    for sign in (1, -1):
        for yaw in (0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2, np.pi):
            upright_rotation = (sapien.Pose(q=euler2quat(0, sign * np.pi / 2, yaw)) *
                                sapien.Pose(q=peg_init_pose.q))
            upright_peg = sapien.Pose([*peg_init_pose.p[:2], env.peg_half_length + 0.10], upright_rotation.q)
            final_pose = upright_peg * peg_to_tcp
            path = planner.move_to_pose_with_screw(final_pose, dry_run=True)
            if path != -1 and (best is None or len(path["position"]) < len(best[0]["position"])):
                best = path, final_pose
    if best is None:
        planner.close()
        return -1
    path, final_pose = best
    res = planner.follow_path(path)

    # -------------------------------------------------------------------------- #
    # Lower
    # -------------------------------------------------------------------------- #
    lower_pose = sapien.Pose([0, 0, -0.099]) * final_pose
    res = planner.move_to_pose_with_screw(lower_pose)
    if res == -1: return res

    res = planner.open_gripper(t=10)
    # Withdraw away from the peg along the gripper's approach axis, then settle.
    retreat_pose = env.agent.tcp.pose.sp * sapien.Pose([0, 0, -0.06])
    res = planner.move_to_pose_with_screw(retreat_pose)
    if res == -1: return res
    res = planner.open_gripper(t=20)
    planner.close()
    return res

if __name__ == "__main__":
    main()
