# 第二批刚体任务

每个任务独立采集、训练和评估。输入仍是512点的 `[dx,dy,dz,distance]`，不添加RGB或隐藏物体位姿。

| 任务 | 配置文件（configs/） | state | 动作 | 默认上限 | 成功后保持 |
| --- | --- | --- | --- | --- | --- |
| LiftPegUpright-v1 | liftpegupright.yaml | 25 | 7：位姿增量＋夹爪 | 400 | 20步 |
| PlaceSphere-v1 | placesphere.yaml | 25 | 7：位姿增量＋夹爪 | 300 | 20步 |
| PullCube-v1 | pullcube.yaml | 28 | 4：位置增量＋夹爪 | 200 | 20步 |

state 的0–8维是qpos，9–17维是qvel，18–24维是基座系TCP位姿xyz＋wxyz。
PullCube 的25–27维额外提供基座系指定目标位置；薄目标圆盘不依赖点云可见性。
PlaceSphere 的容器位置由点云描述，不读取其 `bin_pos`、`is_grasped` 或球的真值位姿作为策略输入。
所有任务诊断中的物体位置、速度、抓取、成功标志只用于验收和日志。

## 成功判定和示范

LiftPeg 原本用XYZ欧拉角的第三项判断直立，与杆的实际长轴不一致。
现在按局部X轴和世界竖直方向的夹角判断：小于0.08rad，杆中心高度距0.12m小于5mm。
允许两端朝上及任意yaw。专家先抓取、抬高，再搜索可达的直立姿态，放下、释放和撤离。
示范额外要求杆已释放且静止，20步保持期间也须持续满足。
该成功判定是本地修正版，不能将结果直接视为采用旧错误判定的官方基准结果。

PlaceSphere 沿用本地原生成功条件：水平偏差和高度偏差均不超过5mm、球静止且不再被抓持。
专家使用球半径和容器底厚计算放置高度，释放后撤离并等待稳定，不收录运动规划失败结果。
PullCube 沿用原生条件：方块与目标中心的XY距离小于0.1m；原生条件没有额外的高度或静止限制。
位置增量控制的保持动作置零机械臂增量，保留夹爪状态，避免重复末动作继续推动。

新任务数据采用契约v7，记录相机、crop、目标字段、动作范围和LiftPeg修正后的成功语义。
窗口覆盖真实末帧；末尾动作填充为零机械臂增量并保持夹爪。
已有PickCube契约及其余第一批任务v3保持兼容。

## 使用命令

先按项目现有环境说明激活 `.venv-ppu`、设置源码路径与软件Vulkan；PPU训练还需要初始化已有SDK。
以下从项目根目录运行。软件Vulkan路径以机器实际存在的文件为准，本机为 `/usr/share/vulkan/icd.d/lvp_icd.json`。
下面以PullCube为例；测试另外两项时，同时替换任务名和对应配置文件。

```bash
export FLOW_TASK=PullCube-v1
export FLOW_CONFIG=examples/baselines/flow_dp3/configs/pullcube.yaml
export FLOW_RUN_ROOT=.runtime/flow_dp3/pullcube-batch2-run01
export FLOW_DATA="$FLOW_RUN_ROOT/demos.h5"
export FLOW_CONTROL=ee

python -B examples/baselines/flow_dp3/prepare_demos.py \
  --env-id "$FLOW_TASK" --generate 120 --count 100 --max-attempts 300 \
  --control-mode "$FLOW_CONTROL" --output "$FLOW_DATA"

python -B examples/baselines/flow_dp3/train.py \
  --config "$FLOW_CONFIG" --env-id "$FLOW_TASK" --data "$FLOW_DATA" \
  --control-mode "$FLOW_CONTROL" --output "$FLOW_RUN_ROOT/full" \
  --device cuda:0 --batch-size 32 --steps 30000 \
  --wandb-mode online --wandb-project manskill \
  --wandb-name "${FLOW_RUN_ROOT##*/}-full" --wandb-log-every 10

python -B examples/baselines/flow_dp3/evaluate.py \
  --env-id "$FLOW_TASK" --checkpoint "$FLOW_RUN_ROOT/full/last.pt" \
  --device cuda:0 --episodes 20 --start-seed 1000 \
  --output "$FLOW_RUN_ROOT/eval.json" --save-video --save-trace \
  --wandb-mode disabled

python -B visual/pointcloud.py \
  --dataset "$FLOW_DATA" --episode 0 --frame 0 \
  --export-backend matplotlib --export-view both \
  --png "$FLOW_RUN_ROOT/cloud.png"
```

LiftPeg 使用 `FLOW_TASK=LiftPegUpright-v1`、`configs/liftpegupright.yaml`；
PlaceSphere 使用 `FLOW_TASK=PlaceSphere-v1`、`configs/placesphere.yaml`，并分别选择新的运行目录。
`ee` 别名自动选择各任务默认控制器；这三个任务当前只录制单分支，不支持 `both` 或 `joint`。
只检查CPU流程时将训练/评估的 `--device` 换成 `cpu`，将W&B设为 `disabled`。
生成120条原生成功轨迹是给控制转换失败预留余量，不保证转换后必定有100条。
数量不足会明确报错并保留已生成数据；不要覆盖旧输出。
默认每10条成功示范录制一条完整场景MP4；`--video-every 1` 录制每条。
新任务原生注册上限为50步，Flow DP3使用上表更长的上限，便于完成专家轨迹和保持验收。

本次只运行小规模流程验证，没有启动上述100条数据采集或30000步正式训练。
具体结果见 [version.md](../../../version.md) 和本地 `.runtime/flow_dp3/rigid-batch2-20261010/`。

每个任务保存3条成功转换且持续保持的示范，已验证小模型训练、断点恢复、闭环推理和可视化。
PullCube最终验证使用 `PullCube-v1/demos-right65.h5`，其余两项使用各自目录下的 `demos.h5`。
新相机改善PullCube接触阶段的观察，但首条88帧仍有2帧方块FPS点数为0；
PlaceSphere抽查帧中球只有4–6个FPS点。当前512点预设可用于开始实验，正式成功率仍待评测。
