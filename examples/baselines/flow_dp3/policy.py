"""独立 FlowDP3：末端关系编码器 + 本体 MLP + FiLM U-Net + 原分段一致性目标。

保留用户参考 flow_dp3.py 的全局条件路径、时间尺度、两时刻损失与采样公式。
使用任务记录的 state/action 维度；不输入 RGB/触觉。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from conditional_unet1d import ConditionalUnet1D
from ee_relation_encoder import EERelationPointNetPPEncoder


@dataclass(frozen=True)
class PolicyConfig:
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8
    state_dim: int = 28
    action_dim: int = 7
    radius1_m: float = 0.05
    radius2_m: float = 0.12
    length_scale: float = 1.0
    down_dims: tuple = (512, 1024, 2048)
    diffusion_step_embed_dim: int = 128
    kernel_size: int = 5
    n_groups: int = 8
    num_inference_steps: int = 10
    fm_eps: float = 0.01
    fm_time_scale: float = 100.0
    num_segments: int = 2
    boundary: float = 1.0
    delta: float = 0.01
    alpha: float = 1e-5
    noise_scale: float = 1.0
    sigma_var: float = 0.0
    solver: str = "consistency"

    def __post_init__(self):
        divisor = 2 ** (len(self.down_dims) - 1)
        if len(self.down_dims) < 2 or self.horizon % divisor or not 1 <= self.n_obs_steps <= self.horizon:
            raise ValueError("horizon 必须可被 U-Net 下采样倍数整除，观测步数必须有效")
        if not 1 <= self.n_action_steps <= self.horizon - self.n_obs_steps + 1:
            raise ValueError("动作块超出 horizon")
        if type(self.state_dim) is not int or type(self.action_dim) is not int or self.state_dim < 1 or self.action_dim < 1:
            raise ValueError("state_dim/action_dim 必须为正整数")
        if self.kernel_size % 2 != 1 or self.kernel_size < 1 or self.n_groups < 1 or any(
            d <= 0 or d % self.n_groups for d in self.down_dims
        ) or self.down_dims[0] % 8:
            raise ValueError("kernel 必须为正奇数，U-Net 通道必须可被 GroupNorm 组数整除")
        if self.diffusion_step_embed_dim < 4 or self.diffusion_step_embed_dim % 2:
            raise ValueError("时间编码维数必须为至少4的偶数")
        if self.solver not in ("consistency", "euler") or self.num_inference_steps < 1 or self.num_segments < 1:
            raise ValueError("solver/采样步数/分段数配置错误")
        if not 0 < self.fm_eps < 1 or not 0 < self.delta < 1 or not 0 <= self.boundary <= 1:
            raise ValueError("时间范围/边界配置错误")
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.radius1_m, self.radius2_m, self.length_scale, self.noise_scale, self.fm_time_scale)):
            raise ValueError("物理半径、长度尺度、噪声与时间尺度必须为有限正数")
        if not math.isfinite(self.alpha) or self.alpha < 0 or not math.isfinite(self.sigma_var) or self.sigma_var < 0:
            raise ValueError("alpha/sigma_var 必须为有限非负数")


class LimitsNormalizer(nn.Module):
    """参考 limits 归一化；近常量通道保留单位尺度，仅减训练集常量。"""
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("scale", torch.ones(dim))
        self.register_buffer("offset", torch.zeros(dim))

    @torch.no_grad()
    def fit(self, data):
        data = torch.as_tensor(data, dtype=torch.float32, device=self.scale.device)
        low, high = data.amin(dim=0), data.amax(dim=0)
        constant = high - low < 1e-4
        scale = 2 / torch.where(constant, torch.full_like(low, 2), high - low)
        offset = torch.where(constant, -low, -1 - scale * low)
        self.scale.copy_(scale)
        self.offset.copy_(offset)

    def normalize(self, x):
        return x * self.scale + self.offset

    def unnormalize(self, x):
        return (x - self.offset) / self.scale


class FlowDP3(nn.Module):
    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        self.distance_encoder = EERelationPointNetPPEncoder(
            radius1=config.radius1_m / config.length_scale,
            radius2=config.radius2_m / config.length_scale, output_dim=64,
            num_centers=(128, 32), num_neighbors=(32, 32),
            local_relation_mode="explicit_anchor", readout_mode="dual", geometry_backend="torch")
        self.state_mlp = nn.Sequential(nn.Linear(config.state_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.model = ConditionalUnet1D(
            input_dim=config.action_dim, global_cond_dim=128 * config.n_obs_steps,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims, kernel_size=config.kernel_size, n_groups=config.n_groups,
            condition_type="film", use_down_condition=True, use_mid_condition=True, use_up_condition=True)
        self.state_normalizer = LimitsNormalizer(config.state_dim)
        self.action_normalizer = LimitsNormalizer(config.action_dim)

    def condition(self, obs):
        config = self.config
        distance = obs["pointcloud_distance"]
        state = obs["state"]
        if distance.ndim != 4 or distance.shape[1] != config.n_obs_steps or distance.shape[-1] != 4:
            raise ValueError("距离点云应为 [B,n_obs_steps,N,4]")
        if state.shape != (*distance.shape[:2], config.state_dim):
            raise ValueError("状态/点云批次与时间维度不匹配")
        # 点云已按物理契约统一除以 L，这里保持原样，不做逐通道归一化。
        point_features = self.distance_encoder(distance.flatten(0, 1))
        state_features = self.state_mlp(self.state_normalizer.normalize(state).flatten(0, 1))
        return torch.cat((point_features, state_features), dim=-1).reshape(distance.shape[0], -1)

    def compute_loss(self, batch):
        config = self.config
        actions = self.action_normalizer.normalize(batch["action"])
        if actions.shape[1:] != (config.horizon, config.action_dim):
            raise ValueError("动作序列形状与策略不符")
        condition = self.condition(batch["obs"])
        a0 = torch.randn_like(actions) * config.noise_scale
        t = torch.rand(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t * (1 - config.fm_eps) + config.fm_eps
        r = (t + config.delta).clamp(max=1)
        te, re = t[:, None, None], r[:, None, None]
        xt = te * actions + (1 - te) * a0
        xr = re * actions + (1 - re) * a0
        segments = torch.linspace(0, 1, config.num_segments + 1, device=actions.device, dtype=actions.dtype)
        ends = segments[torch.searchsorted(segments, t, side="left").clamp(min=1)]
        se = ends[:, None, None]
        xend = se * actions + (1 - se) * a0
        vt = self.model(xt, t * config.fm_time_scale, global_cond=condition)
        vr = self.model(xr, r * config.fm_time_scale, global_cond=condition)
        # 健康数值时与原目标等价；异常时直接报错，避免 nan_to_num 掩盖训练故障。
        if not torch.isfinite(vt).all() or not torch.isfinite(vr).all():
            raise FloatingPointError("FlowDP3 速度场出现 NaN/Inf")
        ft = xt + (se - te) * vt
        fr = torch.where(re < config.boundary, xr + (se - re) * vr, xend)
        loss_f = (ft - fr).square().flatten(1).mean(1)
        velocity_mask = (te < config.boundary) & ((se - te) > 1.01 * config.delta)
        loss_v = ((vt - vr).square() * velocity_mask).flatten(1).mean(1)
        return (loss_f + config.alpha * loss_v).mean()

    @torch.no_grad()
    def predict_action(self, obs):
        config = self.config
        condition = self.condition(obs)
        trajectory = torch.randn(condition.shape[0], config.horizon, config.action_dim,
                                 device=condition.device, dtype=condition.dtype) * config.noise_scale
        h = (1 - config.fm_eps) / config.num_inference_steps
        for i in range(config.num_inference_steps):
            time = config.fm_eps + i * h
            t = torch.full((trajectory.shape[0],), time * config.fm_time_scale,
                           device=trajectory.device, dtype=trajectory.dtype)
            velocity = self.model(trajectory, t, global_cond=condition)
            if config.solver == "euler":
                trajectory = trajectory + h * velocity
            else:
                sigma = (1 - time) * config.sigma_var
                correction = 0.5 * time * (1 - time) * velocity - 0.5 * (2 - time) * trajectory
                pred_sigma = velocity + sigma**2 / (2 * config.noise_scale**2 * (1 - time)**2) * correction
                trajectory = trajectory + h * pred_sigma + sigma * h**0.5 * torch.randn_like(pred_sigma)
        prediction = self.action_normalizer.unnormalize(trajectory)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("预测动作包含 NaN/Inf")
        start = config.n_obs_steps - 1
        return {"action": prediction[:, start:start + config.n_action_steps], "action_pred": prediction}


@torch.no_grad()
def update_ema(ema, model, optimization_step):
    # 与参考 EMAModel 的 inv_gamma=1, power=.75, max=.9999 一致。
    step = max(0, optimization_step - 1)
    decay = min(1 - (1 + step) ** -0.75, 0.9999) if step else 0.0
    for target, current in zip(ema.parameters(), model.parameters()):
        target.lerp_(current.detach(), 1 - decay)
    for target, current in zip(ema.buffers(), model.buffers()):
        target.copy_(current)
    return decay
