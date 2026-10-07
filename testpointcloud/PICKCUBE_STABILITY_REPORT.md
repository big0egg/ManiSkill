# PickCube 到点稳定性：修复、数据与推理对照

日期：2026-10-07。范围仅为 PickCube；其他任务控制器与配置保持不变。

## 1. 结论与验证边界

已经修复保持阶段的录制与训练窗口，并接入8维关节绝对目标对照。两套新数据各100条，所有轨迹在新增40步保持期间持续满足真实任务成功条件。专家使用现有7维末端控制也能稳定保持，说明不能把策略失败直接归因于末端控制器或逆运动学。

现有模型的20场景配对实验中，将每次执行动作从8步改为2步，没有改善最终成功率。因此保留默认8步，提供推理覆盖参数用于后续比较。更优先的处理是用新保持数据重新训练，关节动作作为独立对照。

**尚未完成新7维/8维正式模型的成功率对照。** 本次完成的是实际专家回放、现有正式模型推理、保持状态预测探测，以及两套新接口的小模型训练/加载/环境执行检查。20次更新的小模型结果不能证明策略收敛，也不能用来比较控制器性能。

## 2. 找到的问题

原训练数据为 `.runtime/flow_dp3/PickCube-v1-first-v1.h5`，契约v4、100条、7,762个动作。原正式权重为该实验目录的 `full/best.pt`，EMA、step 27,700；没有覆盖或修改这些文件。

原配置为2帧观测、预测16步、每次执行8步。旧采样器的当前观测最晚只到 `T−8`，末尾7个动作对应状态以及terminal状态不能成为当前训练条件。专家最后仅短暂停稳后结束，采样还重复最后一个非零增量动作填充未来序列。

以“首次闭爪后，TCP距目标≤2.5cm且7个臂关节最大速度≤0.2rad/s”为统一诊断指标，80条训练轨迹中78条没有到点静止的当前状态窗口，5,662个窗口只有10个符合，占0.177%。这里使用TCP近似到点；真实任务判定使用方块中心，二者不可混同。旧数据没有逐步物体诊断，所以旧数据的真实成功窗口数未估算。

原始100条示范最后的平移指令模长中位数为3.128mm，但这本身不能证明一定漂移：末端增量控制在稳态下也可能需要非零补偿。关键是保持过程缺乏足够的真实后续观测和监督。

## 3. 已落地的修改

### 3.1 真实保持示范

[prepare_demos.py](../examples/baselines/flow_dp3/prepare_demos.py) 默认在 PickCube 到点成功后继续40步，即20Hz下2秒：

- v5位姿动作：冻结最后的控制目标位姿，每步根据实际TCP重新计算位置和姿态修正，持续闭爪。没有简单重复末尾增量。
- v6关节动作：持续执行最后一个绝对关节目标和夹爪指令。
- 每步真实仿真、重新渲染点云并采集state。只恢复轨迹初始状态，没有逐帧强设状态。
- 保持阶段每步都成功、最终也成功才保存。`task_metrics` 单独保存方块距离、关节速度、抓取和成功诊断；这些信息不进入模型。

新增参数：`--hold-steps`（Pick默认40，其他任务默认0）、`--control-mode pd_joint_pos`（仅Pick对照）。

### 3.2 覆盖末尾状态，区分动作填充

[dataset.py](../examples/baselines/flow_dp3/dataset.py) 对v5/v6允许从时刻0到真实terminal观测T作为当前条件。末尾不足预测长度时，v5填充零机械臂增量并保留最终夹爪，v6重复最终绝对目标。填充是序列边界约定，不是伪装成真实录制动作；实际保持标签仍是物理执行产生的修正，可能含非零稳态补偿。

历史v1/v2/v3/v4继续使用旧采样规则，避免改变已有训练恢复行为。新Pick配置加入 `training.required_contract_version`，拒绝误用旧7维数据。旧实验恢复使用原训练目录保存的配置。

### 3.3 关节动作与推理

新增 [pickcube_joint.yaml](../examples/baselines/flow_dp3/configs/pickcube_joint.yaml) 与小模型配置。state保持28维；action为7个关节绝对目标角度（弧度）＋1个标准化夹爪指令，共8维。实际手指状态仍为两个关节，所以qpos为9维，动作不是9维。

v5/v6契约记录控制器动作范围和采样规则；v6关节范围从Panda URDF读取。录制、环境校验、推理裁剪和可视化都按契约范围处理，合法的2.2rad等角度不会被错误截为1rad。模型内部仍按训练数据统计归一化，输出反归一化后以真实弧度交给控制器。

[evaluate.py](../examples/baselines/flow_dp3/evaluate.py) 新增 `--n-action-steps`、`--reset-policy-seed-per-episode`、`--save-trace`。保存逐步真实距离、速度、抓取及动作，持续评估到200步，没有在瞬时成功后提前结束。

输入保持不变：state28、FPS512、点云4通道、SA半径0.05/0.12m、原crop与相机。目标位置仍来自 `obs.extra.goal_pos`，变换到基座坐标进入 `state[25:28]`；目标标记在传感器观测中隐藏。没有增加物体真值、成功标志或抓取标志输入。

## 4. 完整新数据核验

使用同一原始关节数据的source episode 0～99，两个版本均尝试100条、保存100条、拒绝0条。训练/验证按episode划分为80/20，划分种子42。

| 指标 | 旧v4数据 | 新v5位姿保持 | 新v6关节保持 |
| --- | ---: | ---: | ---: |
| action维度 | 7 | 7 | 8 |
| 实际动作数 | 7,762 | 11,762 | 11,720 |
| 训练窗口数 | 5,662 | 9,502 | 9,469 |
| TCP近目标且静止的当前训练窗口 | 10 | 3,612 | 3,613 |
| 没有上述窗口的训练轨迹 | 78/80 | 0/80 | 0/80 |
| 真实任务成功的当前训练窗口 | 未记录 | 3,632 | 3,633 |
| 新增保持阶段全部40步成功 | 无此阶段 | 100/100 | 100/100 |

新v5统一诊断窗口占比38.01%，v6为38.16%。这确认保持状态实际进入了训练条件，不仅是HDF5里多保存了帧。

| 保持阶段的真实指标 | v5位姿 | v6关节 |
| --- | ---: | ---: |
| 所有轨迹保持期间的最大方块目标误差 | 11.25mm | 11.18mm |
| 最终方块目标误差中位数 | 5.82mm | 5.75mm |
| 保持期间最大臂关节速度 | 0.0572rad/s | 0.0583rad/s |

两者都低于任务的25mm、0.2rad/s阈值，保持期间也持续抓住方块。微小差异不能证明关节策略优于位姿策略。

完整核验包含源数据指纹、形状、有限值、动作范围、距离通道、目标坐标一致性、全部保持步的真实成功与抓取，以及末尾样本填充：[full-data-audit.json](runs/pickcube-stability/full-data-audit.json)。

新训练数据：

- [pickcube-pose-hold-v5.h5](runs/pickcube-stability/pickcube-pose-hold-v5.h5)，约88MiB。
- [pickcube-joint-hold-v6.h5](runs/pickcube-stability/pickcube-joint-hold-v6.h5)，约79MiB。

## 5. 现有正式模型：8步与2步配对推理

相同EMA权重、相同20个场景种子1000～1019、每局200步、CPU仿真和策略计算、每局重置策略随机种子42。两个动作块会改变预测次数，后续噪声采样序列不会逐步相同。这是完整策略运行方式的比较；单个策略随机种子的20局不足以证明普遍优劣。

| 指标 | 执行8步 | 执行2步 |
| --- | ---: | ---: |
| 曾抓住方块 | 14/20 | 12/20 |
| 最后仍抓住方块 | 7/20 | 8/20 |
| 曾满足成功条件 | 2/20 | 2/20 |
| 最终成功 | 1/20 | 0/20 |
| 最后40步全部成功 | 1/20 | 0/20 |
| 最终仍抓住但偏离目标 | 6/20 | 8/20 |
| 每局策略计算总时间均值 | 10.49s | 40.16s |

8步的seed1016曾成功9帧，方块最小目标误差23.38mm，最终38.60mm；最终关节速度只有0.0433rad/s且还抓着方块。该局失败直接表现为位置误差超限，不能归结为“机械臂一直没停稳”。2步在另外两局曾成功，但最终也离开目标范围。

所以保留8步默认值，先补保持数据；2步覆盖参数保留供新训练后再比较。本次每局重置随机流且使用CPU，与原先整次评估共享随机流的PPU录像不是完全相同协议，不能直接用本次数字宣称原模型成功率变化。

[配对统计与逐场景结果](runs/pickcube-stability/action-block-comparison.json)、[8步报告](runs/pickcube-stability/existing-block8.json)、[2步报告](runs/pickcube-stability/existing-block2.json)。逐步轨迹文件为对应的 `*-trace.json`。

![位置与速度诊断](runs/pickcube-stability/action-block-comparison.png)

## 6. 稳定观测上的预测探测

对原训练划分的20条验证轨迹，取新v5真实保持结束时的两帧观测，输入原正式模型，策略随机种子42。仅测预测指令，没有执行这些预测，不能把指令增量当作实际物体移动。

| 前两步/末两步指令模长的跨轨迹中位数 | 原模型预测前两步 | 专家真实保持末两步 |
| --- | ---: | ---: |
| 平移增量 | 2.101mm | 0.465mm |
| 按控制器尺度换算的旋转向量模长 | 0.0952° | 0.0473° |

探测状态中方块已稳定在目标附近，但原模型仍输出明显更大的平移指令。这支持优先补保持监督的判断，并没有证明它是所有失败的唯一原因。专家保持的非零指令也说明不能把“非零动作”直接等同于错误。[完整探测](runs/pickcube-stability/existing-hold-action-probe.json)。

## 7. 检查与产物

28项相关测试通过，包含旧契约恢复、新旧checkpoint预测、terminal条件、增量/绝对动作填充、真实弧度范围、可视化范围校验、拒绝旧v4数据、位姿修正与本地控制器旋转约定一致性。[测试日志](runs/pickcube-stability/tests-final.log)。另一个真实W&B离线SDK测试受沙箱Unix/TCP socket限制失败，未修改日志模块；最终相关测试不包含该环境受阻项。

两个小模型均完成20次更新、保存/加载checkpoint，并在实际环境执行7/8维动作16步。两次短执行都没有完成任务，仅验证链路：

- [位姿训练日志](runs/pickcube-stability/pose-train-smoke/metrics.jsonl)、[位姿接口执行](runs/pickcube-stability/pose-eval-smoke.json)。
- [关节训练日志](runs/pickcube-stability/joint-train-smoke/metrics.jsonl)、[关节接口执行](runs/pickcube-stability/joint-eval-smoke.json)。

专家视频各115帧、20FPS、1800×1200；逐帧state和点云误差均为0，最终成功：

- [位姿保持视频](runs/pickcube-stability/pose-expert-video/episode_00000.mp4)、[质量记录](runs/pickcube-stability/pose-expert-video/quality.json)。
- [关节保持视频](runs/pickcube-stability/joint-expert-video/episode_00000.mp4)、[质量记录](runs/pickcube-stability/joint-expert-video/quality.json)。

代码修改涉及任务契约、录制、采样、动作范围、推理诊断、训练版本校验、Pick配置、可视化与测试；README、introduce、task、version和visual说明已同步。没有修改参考策略项目、其他任务配置、FPS、相机或crop。

## 8. 新正式训练与复现

本次工具会话初始化PPU SDK后，`torch.cuda.is_available()`仍为False；此前用户DSW训练记录可使用PPU，这不代表用户终端也无法使用。当前磁盘约6GiB可用；现有正式checkpoint约3.9GiB，保存best/last及原子临时文件需约12GiB余量。故未启动新的30,000步正式训练。

先按 introduce 第1.1节初始化可访问PPU的终端，并为新训练输出准备足够空间。以下两组独立训练使用相同主干、batch size和更新数；不会续写旧实验：

```bash
python -B examples/baselines/flow_dp3/train.py \
  --config examples/baselines/flow_dp3/configs/pickcube.yaml \
  --env-id PickCube-v1 \
  --data testpointcloud/runs/pickcube-stability/pickcube-pose-hold-v5.h5 \
  --output testpointcloud/runs/pickcube-stability/pose-full \
  --device cuda:0 --steps 30000 --batch-size 32 --wandb-mode disabled

python -B examples/baselines/flow_dp3/train.py \
  --config examples/baselines/flow_dp3/configs/pickcube_joint.yaml \
  --env-id PickCube-v1 \
  --data testpointcloud/runs/pickcube-stability/pickcube-joint-hold-v6.h5 \
  --output testpointcloud/runs/pickcube-stability/joint-full \
  --device cuda:0 --steps 30000 --batch-size 32 --wandb-mode disabled
```

两个实验都用相同场景/策略种子评估，先每次执行8步，再用新输出名比较2步。不要用旧v4 checkpoint的 `--resume` 来训练新v5数据；恢复校验会拒绝数据/契约改变。

复现录制时使用新输出名：

```bash
python -B examples/baselines/flow_dp3/prepare_demos.py \
  --env-id PickCube-v1 --source .runtime/flow_dp3/PickCube-v1-first-v1.raw.h5 \
  --count 100 --hold-steps 40 \
  --output testpointcloud/runs/pickcube-stability/pose-reproduction.h5

python -B examples/baselines/flow_dp3/prepare_demos.py \
  --env-id PickCube-v1 --source .runtime/flow_dp3/PickCube-v1-first-v1.raw.h5 \
  --control-mode pd_joint_pos --count 100 --hold-steps 40 \
  --output testpointcloud/runs/pickcube-stability/joint-reproduction.h5
```

对现有模型复现动作块比较，分别替换执行步数和报告输出名：

```bash
python -B examples/baselines/flow_dp3/evaluate.py \
  --checkpoint .runtime/flow_dp3/PickCube-v1-first-v1/full/best.pt \
  --device cuda:0 --episodes 20 --start-seed 1000 --policy-seed 42 \
  --n-action-steps 8 --reset-policy-seed-per-episode --save-trace \
  --output testpointcloud/runs/pickcube-stability/reproduction-block8.json \
  --wandb-mode disabled
```

本报告的对照使用 `--device cpu`、OMP线程4；改变为PPU可能引入数值差异。报告中的科学图可由 `summarize_pickcube_action_blocks.py` 复现，数据覆盖统计由 `analyze_pickcube_stability.py` 复现。正式新训练完成后，才能判断保持修复能提升多少成功率，以及关节动作是否值得作为默认接口。
