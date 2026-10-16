"""Simulator truth for demonstration checks and reports, never policy inputs."""
import torch

from pickcube_metrics import pickcube_metrics
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix


def task_metrics(env, info=None):
    env = env.unwrapped
    info = env.evaluate() if info is None else info
    env_id = env.spec.id
    if env_id == "PickCube-v1":
        return pickcube_metrics(env, info)
    metrics = {"success": bool(torch.as_tensor(info["success"]).item()),
               "arm_qvel_maxabs": float(env.agent.robot.get_qvel()[0, :7].abs().max().item())}
    if env_id == "LiftPegUpright-v1":
        vertical = quaternion_to_matrix(env.peg.pose.q)[0, 2, 0].abs().clamp(0, 1)
        metrics.update(upright_angle_rad=float(torch.acos(vertical).item()),
                       goal_error_m=float((env.peg.pose.p[0, 2] - env.peg_half_length).abs().item()),
                       is_grasped=bool(env.agent.is_grasping(env.peg).item()),
                       is_obj_static=bool(env.peg.is_static(lin_thresh=1e-2, ang_thresh=0.5).item()))
    elif env_id == "PlaceSphere-v1":
        offset = env.obj.pose.p[0] - env.bin.pose.p[0]
        error = offset.clone()
        error[2] -= env.radius + env.block_half_size[0]
        metrics.update(goal_error_m=float(error.norm().item()),
                       xy_error_m=float(offset[:2].norm().item()),
                       height_error_m=float(error[2].abs().item()),
                       is_grasped=bool(info["is_obj_grasped"].item()),
                       is_obj_static=bool(info["is_obj_static"].item()))
    elif env_id == "PullCube-v1":
        offset = env.obj.pose.p[0, :2] - env.goal_region.pose.p[0, :2]
        metrics.update(goal_error_m=float(offset.norm().item()),
                       is_grasped=bool(env.agent.is_grasping(env.obj).item()))
    elif env_id == "PullCubeTool-v1":
        base_pos = env.agent.robot.get_links()[0].pose.p
        distance = (env.cube.pose.p[0, :2] - base_pos[0, :2]).norm()
        force = env.scene.get_pairwise_contact_forces(env.l_shape_tool, env.cube)[0].norm()
        metrics.update(cube_to_base_xy_m=float(distance.item()),
                       is_tool_grasped=bool(env.agent.is_grasping(env.l_shape_tool).item()),
                       tool_cube_contact_force_n=float(force.item()),
                       tool_cube_contact=bool((force > 1e-4).item()))
    elif env_id == "PickSingleYCB-v1":
        offset = env.obj.pose.p[0] - env.goal_site.pose.p[0]
        metrics.update(goal_error_m=float(offset.norm().item()),
                       xy_error_m=float(offset[:2].norm().item()),
                       height_error_m=float(offset[2].abs().item()),
                       is_grasped=bool(env.agent.is_grasping(env.obj).item()),
                       is_robot_static=bool(env.agent.is_static(0.2).item()))
    return metrics


def demonstration_success(env_id, metrics):
    # Lift's native geometric success can occur while the arm supports the peg.
    # Only released, static upright demonstrations are useful for this dataset.
    if env_id == "LiftPegUpright-v1":
        return metrics["success"] and not metrics["is_grasped"] and metrics["is_obj_static"]
    if env_id == "PickSingleYCB-v1":
        return metrics["success"] and metrics["is_grasped"]
    return metrics["success"]


def demonstration_trace_success(env_id, trace):
    """Additional dataset quality checks; native evaluation success is unchanged."""
    if env_id == "PullCubeTool-v1":
        return any(row["is_tool_grasped"] and row["tool_cube_contact"] for row in trace)
    return True
