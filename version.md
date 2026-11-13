# 当前版本与验证范围

## 2026-11-13：EE单控制与四组点云接口

按用户要求清理旧实验数据、权重、日志和报告。保留本机Python/ROCm/Mesa环境及依赖缓存。

策略移除joint及双分支录制，只保留各任务原有EE控制。PickCube采用7维末端位姿动作；其他任务保持各自7/4/3维动作。专家内部关节规划、关节状态观测和EE动作转换保留。

新增独立基座XYZ存储、统一表示构造、PointNet++ SSG编码器、C的同构XYZ变体，以及A/B/C/D配置、训练入口和v2 checkpoint契约。旧数据和权重不迁移。

接口与限制见[ABLATION.md](examples/baselines/flow_dp3/ABLATION.md)，运行命令见[introduce.md](introduce.md)。新验收产物位于`.runtime/flow_dp3/ablation-validation/`，仅用于验证接口、训练恢复和视频链路。

当前任务接口为PickCube、PushCube、StackCube、PegInsertionSide、DrawTriangle、LiftPegUpright、PlaceSphere、PullCube、PullCubeTool和固定糖盒PickSingleYCB，各自独立训练。任务已注册不等于正式性能已验证。其他任务的本机完整链路、完整模型长训练、多种子和正式成功率实验尚须执行。

本轮未修改仿真动力学、奖励、成功阈值、评估统计、损失、优化器、EMA、训练/验证划分或状态/动作归一化统计来源。
