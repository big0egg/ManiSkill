"""Shared, stateless point-cloud inputs for offline training and live inference.

XYZ is independently sampled in robot-base axes, in metres. No point-cloud
normalizer is fitted. Only B/T are flattened; e is the current observed TCP.
"""
import math

import torch


INPUT_VERSION = 2
STORAGE_CONTRACT = {
    "version": 2, "xyz_field": "pointcloud_xyz", "frame": "robot_base",
    "units": "m", "source": "sampled_before_tcp_subtraction",
    "alignment": "same_observation_and_same_point_indices",
}
ABLATIONS = {
    "A": ("xyz", "pointnetpp"),
    "B": ("ee_vector_distance", "pointnetpp"),
    "C": ("xyz", "ee_relation_pointnetpp"),
    "D": ("ee_vector_distance", "ee_relation_pointnetpp"),
}


def ablation_group(config):
    for group, pair in ABLATIONS.items():
        if pair == (config.pointcloud_representation, config.pointcloud_encoder_type):
            return group
    raise ValueError("Unsupported point-cloud representation/encoder combination")


def build_pointcloud_inputs(obs, representation, length_scale=1.0):
    """Return geometry [B*T,N,3] and attributes [B*T,N,3 or 4].

    A/C never access ee_position, state, or pointcloud_distance. A cached
    distance field is not used to reconstruct XYZ, including for B/D.
    Input point padding is unsupported: collectors must supply N real points.
    """
    if not math.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length_scale must be finite and positive")
    if "valid_mask" in obs:
        raise ValueError("Padded input points are unsupported; supply N real points")
    if "pointcloud_xyz" not in obs:
        raise ValueError("需要重新录制包含独立 pointcloud_xyz 的数据，不能从距离字段反推 XYZ")
    xyz = obs["pointcloud_xyz"]
    if xyz.ndim != 4 or xyz.shape[-1] != 3 or min(xyz.shape[:3]) < 1:
        raise ValueError("pointcloud_xyz must be [B,T,N,3]")
    if not xyz.is_floating_point() or not torch.isfinite(xyz).all():
        raise ValueError("pointcloud_xyz must contain finite floating-point coordinates")
    if representation == "xyz":
        geometry = xyz / length_scale
        features = geometry
    elif representation == "ee_vector_distance":
        if "ee_position" not in obs:
            raise ValueError("ee_vector_distance requires current base-frame ee_position")
        ee = obs["ee_position"]
        if ee.shape != (*xyz.shape[:2], 3) or not torch.isfinite(ee).all():
            raise ValueError("ee_position must be finite [B,T,3], aligned with XYZ")
        relative = xyz - ee.unsqueeze(-2)
        geometry = relative / length_scale
        # Keep the original D arithmetic order: concatenate, then divide by L.
        features = torch.cat((relative, relative.norm(dim=-1, keepdim=True)), dim=-1) / length_scale
    else:
        raise ValueError(f"Unsupported pointcloud_representation: {representation}")
    return geometry.flatten(0, 1), features.flatten(0, 1)


def validate_encoder_inputs(geometry, features, feature_dim):
    if (geometry.ndim != 3 or geometry.shape[-1] != 3 or
            features.shape != (*geometry.shape[:2], feature_dim) or
            min(geometry.shape[:2]) < 1):
        raise ValueError(f"Expected [BT,N,3] geometry and [BT,N,{feature_dim}] attributes")
    if (not geometry.is_floating_point() or not features.is_floating_point() or
            not torch.isfinite(geometry).all() or not torch.isfinite(features).all()):
        raise ValueError("Encoder inputs must be finite floating-point tensors")
    if not torch.equal(geometry, features[..., :3]):
        raise ValueError("Geometry and attribute XYZ must share coordinates and point indices")
