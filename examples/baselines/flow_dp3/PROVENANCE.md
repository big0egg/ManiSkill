# Flow DP3 与点云编码器来源

本目录在开发时参考用户提供的DP3修改版；运行时不导入或访问参考项目。

- `ee_relation_encoder.py`：用户末端关系保持型PointNet++的独立实现。D保留原两级关系聚合和双路径读出；统一`encode`入口不改变原`forward`。C采用相同网络，从绝对XYZ及到基座原点的范数构造特征。
- `pointnetpp_encoder.py`：新增纯PyTorch PointNet++ SSG任务编码器，参考[PointNet++论文](https://arxiv.org/abs/1706.02413)与[作者SA实现](https://github.com/charlesq34/pointnet2/blob/master/utils/pointnet_util.py)。为公平比较和本机ROCm运行，使用现有确定性几何算子、LayerNorm、128/32中心、32/32邻居及64维输出；不声称逐项复现原分类网络。
- `conditional_unet1d.py`、`conv1d_components.py`、`positional_embedding.py`：参考实现的独立副本，保留原出处与许可证。
- `policy.py`：保留原Flow DP3全局FiLM路径、分段一致性损失、时间尺度和动作索引，统一接入点云接口。
- `obs_adapter.py`、`prepare_demos.py`、`dataset.py`、`train.py`、`evaluate.py`：ManiSkill任务适配。原始专家使用仓库内运动规划器；策略只使用EE动作。点云坐标变换来自本项目的实际机器人基座与TCP，不使用其他环境的变换矩阵。

状态/动作normalizer仍只拟合训练episode；点云只使用固定物理尺度。新四组未更改Flow DP3损失、AdamW、学习率规则、梯度裁剪、EMA或任务成功定义。

参考代码保留[MIT许可](LICENSE-DP3)，ManiSkill框架遵循根目录许可证。示范来源在manifest中区分自生成专家数据与外部数据。完整接口定义见[ABLATION.md](ABLATION.md)。
