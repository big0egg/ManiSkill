"""Flow DP3 末端关系保持型 PointNet++ 的独立副本。

来自用户提供的参考算法，保持编码器结构；运行时仅依赖本文件和 PyTorch，
不导入参考项目，不依赖 CUDA/PyTorch3D 扩展。

输入已经是同一固定坐标轴下的 [相对位移/L, 距离/L]，不可重复减末端。
几何索引不求导；gather、MLP、关系评分和聚合保留梯度。
"""
import math

import torch
from torch import nn


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def gather_points(points, indices):
    """[B,N,C] + [B,...] -> [B,...,C]，不切断源特征梯度。"""
    batch = torch.arange(points.shape[0], device=points.device)
    batch = batch.reshape(points.shape[0], *([1] * (indices.ndim - 1)))
    return points[batch, indices]


@torch.no_grad()
def canonical_order(xyz):
    """几何字典序作为并列时的固定规则；重复坐标按原索引稳定排列。"""
    order = torch.arange(xyz.shape[1], device=xyz.device).expand(xyz.shape[0], -1)
    for axis in (2, 1, 0):
        values = gather_points(xyz, order)[..., axis]
        rank = torch.argsort(values, dim=1, stable=True)
        order = order.gather(1, rank)
    return order


@torch.no_grad()
def farthest_point_indices(xyz, count):
    """批次向量化 FPS；从距质心最远的点开始，避免重复选择同一索引。"""
    _positive_int(count, "FPS count")
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or count > xyz.shape[1]:
        raise ValueError(f"FPS needs [B,N,3] and count <= N, got {tuple(xyz.shape)}, {count}")
    order = canonical_order(xyz)
    coords = gather_points(xyz, order).float()
    b, n, _ = coords.shape
    centroid = coords.mean(dim=1, keepdim=True)
    farthest = ((coords - centroid) ** 2).sum(-1).argmax(dim=1)
    distances = torch.full((b, n), float("inf"), device=xyz.device)
    chosen = torch.zeros((b, n), dtype=torch.bool, device=xyz.device)
    indices = torch.empty((b, count), dtype=torch.long, device=xyz.device)
    batch = torch.arange(b, device=xyz.device)
    for step in range(count):
        indices[:, step] = farthest
        chosen[batch, farthest] = True
        center = coords[batch, farthest].unsqueeze(1)
        distances = torch.minimum(distances, ((coords - center) ** 2).sum(-1))
        farthest = distances.masked_fill(chosen, -1).argmax(dim=1)
    return order.gather(1, indices)


@torch.no_grad()
def ball_query_indices(source, centers, radius, neighbors):
    """半径内最近的不同索引；不足 K 的槽位返回安全索引和 False mask。"""
    _positive_int(neighbors, "neighbors")
    if not math.isfinite(float(radius)) or float(radius) <= 0:
        raise ValueError(f"radius must be finite and positive, got {radius!r}")
    if source.ndim != 3 or centers.ndim != 3 or source.shape[-1] != 3 or centers.shape[-1] != 3:
        raise ValueError("ball query needs [B,N,3] source and [B,M,3] centers")
    if source.shape[0] != centers.shape[0] or source.shape[1] == 0:
        raise ValueError("ball query requires matching batches and nonempty source")
    order = canonical_order(source)
    coords = gather_points(source, order).float()
    d2 = ((centers.float().unsqueeze(2) - coords.unsqueeze(1)) ** 2).sum(-1)
    # Stable sort gives nearest-first queries with geometric tie breaking.
    ranks = torch.argsort(d2, dim=-1, stable=True)[..., :min(neighbors, source.shape[1])]
    valid = d2.gather(-1, ranks) <= float(radius) ** 2
    indices = order.unsqueeze(1).expand(-1, centers.shape[1], -1).gather(-1, ranks)
    indices = indices.masked_fill(~valid, 0)
    if neighbors > source.shape[1]:
        padding = neighbors - source.shape[1]
        indices = torch.nn.functional.pad(indices, (0, padding), value=0)
        valid = torch.nn.functional.pad(valid, (0, padding), value=False)
    return indices, valid


class _ChannelLayerNorm(nn.LayerNorm):
    def forward(self, value):
        # PAI torch 2.0 CPU BF16 LayerNorm backward cannot mix BF16 inputs with
        # FP32 affine parameters. Compute normalization in FP32, then restore
        # activation dtype; casts keep both input and parameter gradients.
        work_dtype = torch.float32 if value.dtype in (torch.float16, torch.bfloat16) else value.dtype
        result = torch.nn.functional.layer_norm(
            value.to(work_dtype), self.normalized_shape,
            self.weight.to(work_dtype), self.bias.to(work_dtype), self.eps)
        return result.to(value.dtype)


def _shared_mlp(channels):
    layers = []
    for in_dim, out_dim in zip(channels[:-1], channels[1:]):
        layers.extend((nn.Linear(in_dim, out_dim), _ChannelLayerNorm(out_dim), nn.ReLU()))
    return nn.Sequential(*layers)


def masked_max(features, valid):
    pooled = features.masked_fill(~valid.unsqueeze(-1), float("-inf")).max(dim=-2).values
    return torch.where(valid.any(dim=-1, keepdim=True), pooled, torch.zeros_like(pooled))


class EERelationPointNetPPEncoder(nn.Module):
    def __init__(self, radius1=None, radius2=None, output_dim=64,
                 num_centers=(128, 32), num_neighbors=(32, 32),
                 local_relation_mode="explicit_anchor", readout_mode="dual",
                 geometry_backend="torch", debug_checks=False):
        super().__init__()
        if len(num_centers) != 2 or len(num_neighbors) != 2:
            raise ValueError("num_centers and num_neighbors must each contain two integers")
        self.num_centers = tuple(_positive_int(x, "num_centers") for x in num_centers)
        self.num_neighbors = tuple(_positive_int(x, "num_neighbors") for x in num_neighbors)
        if self.num_centers[1] > self.num_centers[0]:
            raise ValueError("SA2 center count must not exceed SA1 center count")
        for name, value in (("radius1", radius1), ("radius2", radius2)):
            if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be configured as a finite positive radius in input U units; got {value!r}")
        if local_relation_mode not in ("explicit_anchor", "raw_features"):
            raise ValueError(f"Unsupported local_relation_mode: {local_relation_mode}")
        if readout_mode not in ("dual", "global_max"):
            raise ValueError(f"Unsupported readout_mode: {readout_mode}")
        if geometry_backend != "torch":
            raise ValueError("Only the tested pure PyTorch geometry_backend='torch' is implemented")
        self.radius1, self.radius2 = float(radius1), float(radius2)
        self.output_dim = _positive_int(output_dim, "output_dim")
        self.local_relation_mode, self.readout_mode = local_relation_mode, readout_mode
        self.debug_checks = bool(debug_checks)
        self.sa1 = _shared_mlp((8 if local_relation_mode == "explicit_anchor" else 7, 64, 64, 128))
        self.sa2 = _shared_mlp((135 if local_relation_mode == "explicit_anchor" else 131, 128, 128, 256))
        self.region_projection = _shared_mlp((260, 256))
        if readout_mode == "dual":
            self.score_a = nn.Sequential(nn.Linear(259, 64), nn.ReLU(), nn.Linear(64, 1))
            self.score_b = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
        self.output_projection = nn.Linear(512 if readout_mode == "dual" else 256, output_dim)

    def forward(self, x, return_debug=False):
        if x.ndim != 3 or x.shape[-1] != 4 or x.shape[0] == 0 or x.shape[1] < self.num_centers[0]:
            raise ValueError(f"Expected [B,N,4], B>=1 and N>={self.num_centers[0]}, got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise ValueError("Point cloud input must have a floating point dtype")
        if self.debug_checks or return_debug:
            if not torch.isfinite(x).all():
                raise ValueError("Point cloud contains NaN/Inf")
            if (x[..., 3] < 0).any():
                raise ValueError("Point cloud distances must be nonnegative")
            eps = torch.finfo(x.dtype).eps
            if not torch.allclose(x[..., 3].float(), x[..., :3].float().norm(dim=-1),
                                  atol=max(1e-5, 2 * eps), rtol=max(1e-4, 2 * eps)):
                raise ValueError("Distance channel is inconsistent with ||relative displacement||; check input normalization/frame")
        u0 = x[..., :3]
        i1 = farthest_point_indices(u0, self.num_centers[0])
        u1 = gather_points(u0, i1)
        j1, m1 = ball_query_indices(u0, u1, self.radius1, self.num_neighbors[0])
        neighbor_u1 = gather_points(u0, j1)
        delta1 = neighbor_u1 - u1.unsqueeze(2)
        if self.local_relation_mode == "explicit_anchor":
            anchor1 = torch.cat((u1, u1.float().norm(dim=-1, keepdim=True).to(x.dtype)), dim=-1)
            g1 = torch.cat((delta1, gather_points(x[..., 3:4], j1),
                            anchor1.unsqueeze(2).expand(-1, -1, j1.shape[-1], -1)), dim=-1)
        else:
            g1 = torch.cat((delta1, gather_points(x, j1)), dim=-1)
        h1 = masked_max(self.sa1(g1), m1)
        i2 = farthest_point_indices(u1, self.num_centers[1])
        u2 = gather_points(u1, i2)
        j2, m2 = ball_query_indices(u1, u2, self.radius2, self.num_neighbors[1])
        delta2 = gather_points(u1, j2) - u2.unsqueeze(2)
        rho2 = u2.float().norm(dim=-1, keepdim=True).to(x.dtype)
        parts = [delta2, gather_points(h1, j2)]
        if self.local_relation_mode == "explicit_anchor":
            parts.append(torch.cat((u2, rho2), dim=-1).unsqueeze(2).expand(-1, -1, j2.shape[-1], -1))
        g2 = torch.cat(parts, dim=-1)
        h2 = masked_max(self.sa2(g2), m2)
        f = self.region_projection(torch.cat((h2, u2, rho2), dim=-1))
        z_global = f.max(dim=1).values
        alpha = None
        if self.readout_mode == "dual":
            scores = self.score_a(torch.cat((h2, u2), dim=-1)) + self.score_b(rho2)
            alpha = torch.softmax(scores.float(), dim=1)
            z_interaction = (alpha * f.float()).sum(dim=1).to(f.dtype)
            z = torch.cat((z_global, z_interaction), dim=-1)
        else:
            z = z_global
        output = self.output_projection(z)
        if not return_debug:
            return output
        return output, {"U0": u0, "U1": u1, "U2": u2, "H1": h1, "H2": h2,
                        "g1": g1, "g2": g2, "F": f, "rho2": rho2,
                        "center_indices1": i1, "center_indices2": i2,
                        "neighbor_indices1": j1, "neighbor_indices2": j2,
                        "mask1": m1, "mask2": m2, "alpha": alpha}
