# FlowDP3 独立移植说明

本目录的算法参考用户提供的 DP3 修改版。只在开发时读取参考代码，运行时不导入、不搜索、不访问参考项目，也不要求它存在。

- `ee_relation_encoder.py`：用户 `ee_relation_pointnetpp_encoder.py` 的独立副本，保留两级末端关系聚合与双路径读出。
- `conditional_unet1d.py`、`conv1d_components.py`、`positional_embedding.py`：参考实现的独立副本，只调整本目录导入、移除未使用的 termcolor 导入并补充出处说明。
- `policy.py`：按参考 `flow_dp3.py` 改写独立的全局 FiLM 条件路径。保留分段一致性目标、双时刻梯度、limits 归一化、时间尺度、采样公式和动作索引。未移植未启用的 RGB、触觉、其他点云编码器、非全局条件路径。
- `obs_adapter.py`、`prepare_demos.py`、`dataset.py`、`train.py`、`evaluate.py`：新写的 ManiSkill / Panda 接口。状态字段改为关节位置、关节速度、基座系末端位姿和目标位置；距离点云保持物理几何，不做分通道 min-max。

新训练使用按步数随机抽取窗口、训练集 limits 统计、AdamW、warmup/cosine 调度、梯度裁剪和参考 EMA 衰减。数值异常直接失败，而非用 nan_to_num 掩盖。新 checkpoint 含本任务的28维状态与4维动作，不能直接加载旧任务 checkpoint。

`configs/pickcube.yaml` 保留参考主干宽度 `[512,1024,2048]`；烟雾配置缩小为 `[64,128,256]`，关系编码器不变。半径 `.10/.20` 米是初始候选，不是已调优结论。

参考代码保留 [MIT 许可](LICENSE-DP3)，ManiSkill 框架遵循根目录许可证。DP3 的上游出处见根目录 README 的论文与项目链接。生成示范复用当前 ManiSkill 仓库内置 PickCube 运动规划器；生成数据和公开数据集应在实验记录中区分。
