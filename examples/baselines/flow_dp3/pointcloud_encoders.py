"""Independent representation/encoder selection with one 64-D interface."""
from ee_relation_encoder import EERelationPointNetPPEncoder
from pointnetpp_encoder import PointNetPPEncoder


def build_pointcloud_encoder(config):
    common = dict(radius1=config.radius1_m / config.length_scale,
                  radius2=config.radius2_m / config.length_scale, output_dim=64,
                  num_centers=config.pointcloud_num_centers,
                  num_neighbors=config.pointcloud_num_neighbors)
    if config.pointcloud_encoder_type == "pointnetpp":
        return PointNetPPEncoder(input_dim=3 if config.pointcloud_representation == "xyz" else 4, **common)
    if config.pointcloud_encoder_type == "ee_relation_pointnetpp":
        return EERelationPointNetPPEncoder(
            **common, input_representation=config.pointcloud_representation,
            local_relation_mode=config.local_relation_mode, readout_mode=config.readout_mode,
            geometry_backend="torch")
    raise ValueError(f"Unsupported pointcloud_encoder_type: {config.pointcloud_encoder_type}")
