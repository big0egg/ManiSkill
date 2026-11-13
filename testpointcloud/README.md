# 点云与EE示范诊断工具

旧实验数据和历史报告已按用户要求清理。当前接口、命令和验证范围以根目录[introduce.md](../introduce.md)、[version.md](../version.md)和[四组接口说明](../examples/baselines/flow_dp3/ABLATION.md)为准。

本目录保留相机、半径、观测坐标和EE保持的诊断源程序。历史脚本的默认输入路径不代表当前存在的数据，运行时应显式提供新文件。相机或邻域组成检查不等于训练后的策略成功率。

策略只使用EE控制。诊断中的`pd_joint_pos`用于原始专家回放，关节位置/速度用于本体观测和动力学诊断。编码器的双路径读出与机器人控制方式无关。

当前回归测试：

```bash
python -B -m unittest discover -s testpointcloud -p 'test_*.py' -v
python -B -m unittest discover -s examples/baselines/flow_dp3/tests -v
```

四组真实验收使用新生成的PickCube示范，位置为`.runtime/flow_dp3/ablation-validation/`。该目录被Git忽略，不包含于新检出。
