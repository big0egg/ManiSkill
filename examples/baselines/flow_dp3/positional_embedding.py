"""用户 FlowDP3 参考实现的独立副本；仅改本地导入，运行时不依赖参考仓库。"""
import math
import torch
import torch.nn as nn

# 定义一个名为 SinusoidalPosEmb 的类，继承自 nn.Module
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):           # 构造函数 __init__ 接受一个参数 dim，表示嵌入的维度
        super().__init__()
        self.dim = dim                 # 将传入的 dim 存储为实例变量 self.dim

    def forward(self, x):              # 定义前向传播方法 forward，接受输入张量 x
        device = x.device              # 获取输入张量 x 所在的设备（CPU 或 GPU），并存储在 device 变量中
        half_dim = self.dim // 2       # 计算 half_dim，即 dim 的一半。这是因为正弦位置编码会生成正弦和余弦两种分量，每种分量占据一半的维度
        emb = math.log(10000) / (half_dim - 1)       # 计算一个常数 emb，用于控制频率范围。具体公式为 log(10000) / (half_dim - 1)。这个常数决定了位置编码的频率变化范围
        # 创建一个从 0 到 half_dim-1 的张量，并将其乘以 -emb 后取指数，得到频率因子
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 将输入张量 x 和频率因子 emb 进行广播相乘
        emb = x[:, None] * emb[None, :]    # emb[None, :] 将 emb 的形状从 (half_dim,) 扩展为 (1, half_dim)
        # 广播相乘后，emb 的形状变为 (B, half_dim)，其中每一行对应一个输入时间步的位置编码
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # 将 emb 的正弦和余弦分别计算出来，并沿最后一个维度拼接在一起,
        # emb.sin() 计算 emb 的正弦值，形状为 (B, half_dim),emb.cos() 计算 emb 的余弦值，形状为 (B, half_dim)
        # torch.cat((emb.sin(), emb.cos()), dim=-1) 将正弦和余弦值拼接起来，最终形状为 (B, dim)
        return emb


    # 与原始 Transformer 的位置编码不同，此实现将正弦和余弦部分分别拼接，而非交替排列
