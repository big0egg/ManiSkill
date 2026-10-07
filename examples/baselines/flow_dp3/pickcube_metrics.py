"""Task truth is recorded for diagnosis only; it never enters policy observations."""
import torch


def pickcube_metrics(env, info=None):
    env = env.unwrapped
    info = env.evaluate() if info is None else info
    qvel = env.agent.robot.get_qvel()[0, :7]
    tcp = env.agent.tcp_pose.p[0]
    cube, goal = env.cube.pose.p[0], env.goal_site.pose.p[0]
    return {"goal_error_m": float(torch.linalg.norm(cube - goal).item()),
            "tcp_goal_error_m": float(torch.linalg.norm(tcp - goal).item()),
            "arm_qvel_maxabs": float(qvel.abs().max().item()),
            **{key: bool(torch.as_tensor(info[key]).item()) for key in
               ("is_obj_placed", "is_robot_static", "is_grasped", "success")}}
