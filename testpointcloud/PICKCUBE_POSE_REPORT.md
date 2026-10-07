# PickCube：保留专家旋转动作

日期：2026-10-07。本次只接入 PickCube 的7维位姿动作；其他任务的控制器、观测与配置保持不变。没有修改专家抓取流程，没有加入颜色或物体真值输入。

## 修改与兼容性

原始专家根据方块朝向选择抓取姿态，但旧 `pd_ee_delta_pos` 转换只保留 XYZ 平移和夹爪，策略无法输出旋转。新接口采用 `pd_ee_delta_pose`，动作为 **XYZ 平移增量3维 + 旋转增量3维 + 夹爪1维**，归一化范围为 `[-1,1]`。末端姿态由控制器通过逆运动学分配到各关节，不是直接控制第七关节。

- [task_registry.py](../examples/baselines/flow_dp3/task_registry.py)：新 PickCube 默认7维，按契约版本恢复旧4维接口。
- [obs_adapter.py](../examples/baselines/flow_dp3/obs_adapter.py)：新 PickCube 使用 **v4 契约**；旧 v1/v2 仍使用4维平移动作和保存的相机配置，其他任务继续使用 v3。
- [policy.py](../examples/baselines/flow_dp3/policy.py)、[正式配置](../examples/baselines/flow_dp3/configs/pickcube.yaml)、[烟雾配置](../examples/baselines/flow_dp3/configs/pickcube_smoke.yaml)：默认/显式动作维度为7。新 PickCube 配置拒绝旧4维数据。
- 训练、评估和视频回放已有按契约绑定维度的路径；新旧模型分别构建各自的动作输出。旧 checkpoint 不能直接切换成7维，应以旧配置恢复旧实验。

保持 **state28、FPS512、SA1/SA2半径0.05/0.12m、中心数128/32、邻居上限32/32**。相机仍为 `(0.30,-0.30,0.35)` → `(0,0,0.12)`、256×256、60°；crop仍为基座系 `(0.44,-0.23,-0.03)` → `(0.79,0.25,0.52)`。

## 真实转换

复用现有120条成功原始关节示范：[PickCube-v1-first-v1.raw.h5](../.runtime/flow_dp3/PickCube-v1-first-v1.raw.h5)。仅恢复每条轨迹的初始状态，之后真实执行转换出的7维动作，每步重新渲染点云并采集状态，最终再次判断成功。没有沿用旧4维回放的点云，也没有逐帧强设状态。

小批量记录：[smoke.h5](runs/pickcube-pose-v4/smoke.h5)，尝试10条、成功保存10条，共729个动作。独立核验使用原始关节状态和机器人URDF的正向运动学，对比闭爪前的末端姿态，并先核对其初始TCP与实际录制值一致。

| 小批量指标 | 原始专家 | 新7维转换 |
| --- | ---: | ---: |
| 闭爪前旋转角度中位数 | 11.868° | 11.867° |
| 夹持方向与方块侧面误差中位数 | 0.0347° | 0.0342° |

方块侧面对齐误差取初始方块两条水平轴中更接近夹持方向的一条；这是几何诊断指标，不能代替接触力或真实抓取判定。[小批量完整核验](runs/pickcube-pose-v4/smoke-verification.json)包含逐条数据、源文件指纹、形状与范围检查以及模型前向/反向检查。

完整新训练数据为 [.runtime/flow_dp3/PickCube-v1-pose-v4.h5](../.runtime/flow_dp3/PickCube-v1-pose-v4.h5)：尝试100条、成功保存100条、拒绝0条，共7,762个动作。全部轨迹通过契约、形状、有限值、动作范围、距离通道和初始正向运动学校验，真实数据的模型前向、反向与7维预测也通过。

| 完整100条指标 | 原始专家 | 新7维转换 |
| --- | ---: | ---: |
| 闭爪前旋转角度中位数 | 17.712° | 17.710° |
| 夹持方向与方块侧面误差中位数 | 0.0253° | 0.0239° |
| 夹持方向与方块侧面误差最大值 | 0.455° | 0.461° |

这些结果确认转换后的环境实际执行了专家所需的姿态调整，且全部回放最终成功；它们不等于学习策略的成功率。逐条统计、源文件指纹和完整核验记录于 [full-verification.json](runs/pickcube-pose-v4/full-verification.json)。

## 流程验证

- [5项新旧接口回归测试](test_pickcube_pose_contract.py)通过：版本选择控制器及相机、拒绝篡改契约、其他任务不变、新旧 checkpoint 保存/加载/预测、新训练拒绝旧4维数据。
- 现有测试17项通过；1项真实 W&B 离线SDK测试受沙箱限制失败（本地Unix/TCP socket不可创建），与本次动作修改无关。
- [真实旧数据核验](runs/pickcube-pose-v4/legacy-verification.json)：旧数据仍读出 `(16,4)` 动作，原训练配置仍为4维；新配置为7维。
- [环境接口核验](runs/pickcube-pose-v4/environment-interfaces.json)：旧v2与新v4环境分别接受4/7维动作，reset、step和观测适配通过。
- [运行探测](runs/pickcube-pose-v4/runtime-probe.json)：真实仿真和点云检查通过，新控制器动作空间为7维。
- [20步小模型训练](runs/pickcube-pose-v4/train-smoke/metrics.jsonl)完成，checkpoint保存/加载后在真实环境执行7维动作：[接口评估记录](runs/pickcube-pose-v4/eval-smoke.json)。只有20次更新和16步执行，这不是正式策略训练或成功率对照；该次烟雾执行没有完成任务。
- [专家回放视频](runs/pickcube-pose-v4/expert-video/episode_00000.mp4)与[质量记录](runs/pickcube-pose-v4/expert-video/quality.json)：75帧，逐帧状态/点云与HDF5一致，回放最终成功。

## 新训练与复现

先按 [introduce.md](../introduce.md) 第1.1节初始化终端。新训练使用新数据和独立输出目录：

```bash
python -B examples/baselines/flow_dp3/train.py \
  --config examples/baselines/flow_dp3/configs/pickcube.yaml \
  --env-id PickCube-v1 \
  --data .runtime/flow_dp3/PickCube-v1-pose-v4.h5 \
  --output .runtime/flow_dp3/PickCube-v1-pose-v4/full \
  --device cuda:0 --steps 30000 --batch-size 32 \
  --wandb-mode disabled
```

如需 W&B，沿用 introduce.md 的日志参数。本次没有启动正式模型训练，仍需训练新7维策略后，用相同评估种子比较抓取率与最终成功率。

复现转换时使用尚未存在的新输出文件；源轨迹与旧数据不会覆盖：

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv-ppu/bin/python -B examples/baselines/flow_dp3/prepare_demos.py \
  --env-id PickCube-v1 \
  --source .runtime/flow_dp3/PickCube-v1-first-v1.raw.h5 \
  --count 100 --num-points 512 --length-scale 1.0 \
  --output .runtime/flow_dp3/PickCube-v1-pose-v4-reproduction.h5

.venv-ppu/bin/python -B testpointcloud/verify_pickcube_pose_data.py \
  .runtime/flow_dp3/PickCube-v1-pose-v4-reproduction.h5 \
  --output testpointcloud/runs/pickcube-pose-v4/reproduction-verification.json
```
