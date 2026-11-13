# 四组点云接口与验收

本次接入只改变点云表示、编码器及其配置/持久化接口。正式策略全部使用EE控制。专家内部关节规划与EE动作转换继续保留。旧实验数据和权重已经停用。

## 输入

`p` 为同一次FPS采样的基座系点，`e` 为当前观测的TCP位置，均以米为单位。传感器有效点标记不作为距离通道。位置由世界系到基座系时旋转并平移，向量沿基座坐标轴表达。

| 组别 | 三维几何 | 点属性 | 编码器 |
| --- | --- | --- | --- |
| A | p/L | p/L | PointNet++ SSG |
| B | (p-e)/L | [p-e,‖p-e‖]/L | PointNet++ SSG |
| C | p/L | p/L | 同构自研XYZ变体 |
| D | (p-e)/L | [p-e,‖p-e‖]/L | 完整末端关系编码器 |

统一入口是 `encode(geometry_xyz, point_features) -> [B*T,64]`。只合并batch与时间维度。FPS和球查询只使用三维几何。A/C构造不读取距离缓存、TCP或状态；状态分支仍统一接收机器人状态。

独立XYZ由`obs_adapter.py`在减TCP之前保存，单位米。`dataset.py`和实时适配器都提供同一时刻的XYZ及TCP给`pointcloud_inputs.py`。旧距离缓存不参与任何组的模型输入构造，保留供审计/可视化核对。新存储显式记录版本、单位、坐标系与来源，缺失或不匹配会被拒绝。

输入点必须全部有效，裁剪后不足N点报错，不支持带填充点的输入。球查询不足K个邻居时的无效槽位通过mask排除。

## 编码器

普通基线采用纯PyTorch、两级SSG和普通group-all/max读出；SA1通道为`[3+C,64,64,128]`，SA2为`[131,128,128,256]`，全局层为`[259,256]`，输出Linear为256→64。SA1拼接局部坐标与原始点属性，使B的相对向量和距离不会因局部中心化全部消失。

两种编码器使用相同的FPS、球查询、LayerNorm、中心/邻居预算和64维输出。此基线是任务适配版，不能称为原论文分类网络的逐项复现。PointNet++原论文及作者代码见[出处说明](PROVENANCE.md)。

C在点云编码器内部将XYZ提升为`[p/L,‖p/L‖]`，范数指向固定基座原点；没有补零，也不接收TCP。C/D保持相同网络层、通道宽度和双路径读出。D的`encode`直接调用原`forward`，保留原有计算顺序和参数名称。

所有组共用原状态MLP和动作预测主干，没有新增RGB、显著图、语义采样或额外融合。

## 配置与checkpoint

固定覆盖项位于`config/pointcloud_ablation/{a,b,c,d}.yaml`，由`train.py --ablation A/B/C/D`合并现有任务配置。冲突显式报错。输出为`<output>/<组别>/seed_<seed>`。

v2 checkpoint记录完整策略配置、独立XYZ输入版本、表示与编码器、坐标/单位/TCP约定、尺度、采样设置、半径及自研内部模式。训练恢复与推理都重新校验契约并严格加载权重。旧格式或缺失语义字段不再兼容；这是本轮“放弃旧数据和产物”的明确选择。

## 验收

单元测试位于`tests/test_pointcloud_ablation.py`和`tests/test_ee_control.py`，覆盖四组实际参数更新、有限梯度和动作、缓存/TCP独立性、D原路径等价、轴与时间维度、平移性质、邻域及输入填充、配置冲突、权重恢复和契约错配。

本机新验收产物位于`.runtime/flow_dp3/ablation-validation/`。其中`workflow-validation.json`记录真实数据审计、四组短训练/恢复/视频和恢复一致性结果；它是接口验收，不是正式消融结果。旧环境验收报告不再作为当前结果引用。

运行命令见根目录[introduce.md](../../../introduce.md)。本次不修改训练/验证划分、状态/动作统计来源、物理、奖励、成功阈值、评估统计、损失、优化器或EMA。其他任务、完整主干长训练、正式多种子成功率仍须另行执行。

## 本次实际验收（2026-11-13）

- 新生成3条PickCube EE示范、321帧512点观测；独立XYZ、当前TCP及距离缓存逐索引对齐。
- RX 7700 XT / ROCm：四组分别通过有限损失、梯度、编码器参数更新及动作推理。
- 四组共用同一数据指纹和episode划分，先训练2步，再恢复到3步；每组完成8步闭环推理，策略MP4均解码为9帧。
- D恢复到3步与连续训练3步的model/EMA数值一致。
- EE专家录像以间隔2验证成功编号抽样，三条中第1/3条录像分别解码为115/91帧；第2条跳过录像。
- 第一条专家示范真实动作回放115帧，独立XYZ、距离特征和全部state字段最大误差均为0。

完整数据审计与四组运行记录在`.runtime/flow_dp3/ablation-validation/workflow-validation.json`；录像和回放验证在同目录`observations-validation.json`。

## 变更文件范围

以下代码路径相对于`examples/baselines/flow_dp3/`：

- 新增：`pointcloud_inputs.py`、`pointnetpp_encoder.py`、`pointcloud_encoders.py`、本说明、`config/pointcloud_ablation/{a,b,c,d}.yaml`、`tests/test_ee_control.py`、`tests/test_pointcloud_ablation.py`。
- 修改：`control_modes.py`、`task_registry.py`、`obs_adapter.py`、`prepare_demos.py`、`dataset.py`、`policy.py`、`ee_relation_encoder.py`、`runtime_utils.py`、`train.py`、`evaluate.py`、`PROVENANCE.md`、`RIGID_TASKS.md`及相关现有测试。
- 删除：`configs/pickcube_joint.yaml`、`configs/pickcube_joint_smoke.yaml`。
- 可视化修改：根目录`visual/`下的`dataset_io.py`、`pointcloud.py`、`export_videos.py`、`test_visual.py`和README。
- 根目录更新：`README.md`、`introduce.md`、`version.md`；更新`testpointcloud/README.md`和EE回归测试。
- 清理：旧`.runtime/flow_dp3/`与`.runtime/reports/`运行产物；`testpointcloud`中的双控制脚本/测试及旧实验报告。安装环境、依赖缓存、机器人资产与仿真框架保留。

回归测试最终结果：Flow DP3 39项（38通过、1跳过），EE诊断9项通过，可视化11项通过，合计58通过、1跳过。跳过项为未安装真实W&B SDK的离线集成测试；本地日志与模拟SDK测试通过。`git diff --check`通过。
