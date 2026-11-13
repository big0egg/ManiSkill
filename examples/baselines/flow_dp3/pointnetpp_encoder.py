"""Pure-PyTorch PointNet++ SSG task encoder with LayerNorm and 64-D output.

Two sampled local SA stages and a group-all max readout. This adaptation
shares the tested geometry budget with the relation encoder; it does not
include its anchor augmentation, scalar-origin branch, or attention readout.
"""
from torch import nn
import torch

from ee_relation_encoder import (
    _shared_mlp, _positive_int, farthest_point_indices, ball_query_indices,
    gather_points, masked_max,
)
from pointcloud_inputs import validate_encoder_inputs


class PointNetPPEncoder(nn.Module):
    def __init__(self, input_dim, radius1, radius2, output_dim=64,
                 num_centers=(128, 32), num_neighbors=(32, 32)):
        super().__init__()
        if input_dim not in (3, 4):
            raise ValueError("PointNet++ attributes must have 3 or 4 channels")
        if len(num_centers) != 2 or len(num_neighbors) != 2:
            raise ValueError("Exactly two local SA stages are required")
        self.num_centers = tuple(_positive_int(x, "num_centers") for x in num_centers)
        self.num_neighbors = tuple(_positive_int(x, "num_neighbors") for x in num_neighbors)
        if self.num_centers[1] > self.num_centers[0]:
            raise ValueError("SA2 centers must not exceed SA1 centers")
        import math
        if any(not math.isfinite(r) or r <= 0 for r in (radius1, radius2)):
            raise ValueError("SA radii must be finite and positive")
        self.radius1, self.radius2 = radius1, radius2
        self.input_dim = input_dim
        self.output_dim = _positive_int(output_dim, "output_dim")
        self.sa1 = _shared_mlp((3 + input_dim, 64, 64, 128))
        self.sa2 = _shared_mlp((131, 128, 128, 256))
        self.global_mlp = _shared_mlp((259, 256))
        self.output_projection = nn.Linear(256, output_dim)

    def encode(self, geometry_xyz, point_features):
        validate_encoder_inputs(geometry_xyz, point_features, self.input_dim)
        if geometry_xyz.shape[1] < self.num_centers[0]:
            raise ValueError("Not enough real points for SA1 sampling")
        i1 = farthest_point_indices(geometry_xyz, self.num_centers[0])
        xyz1 = gather_points(geometry_xyz, i1)
        j1, mask1 = ball_query_indices(geometry_xyz, xyz1, self.radius1, self.num_neighbors[0])
        # Preserve the supplied original attributes, including B's v and d.
        local1 = torch.cat((gather_points(geometry_xyz, j1) - xyz1.unsqueeze(2),
                            gather_points(point_features, j1)), dim=-1)
        h1 = masked_max(self.sa1(local1), mask1)
        i2 = farthest_point_indices(xyz1, self.num_centers[1])
        xyz2 = gather_points(xyz1, i2)
        j2, mask2 = ball_query_indices(xyz1, xyz2, self.radius2, self.num_neighbors[1])
        local2 = torch.cat((gather_points(xyz1, j2) - xyz2.unsqueeze(2),
                            gather_points(h1, j2)), dim=-1)
        h2 = masked_max(self.sa2(local2), mask2)
        global_features = self.global_mlp(torch.cat((xyz2, h2), dim=-1)).max(dim=1).values
        return self.output_projection(global_features)

    def forward(self, geometry_xyz, point_features):
        return self.encode(geometry_xyz, point_features)
