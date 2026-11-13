# Flow DP3：EE 控制与四组点云消融

本项目每个任务独立录制、训练和评估。策略只使用末端控制；专家生成和动作转换内部仍使用关节位置控制。环境安装见 [README.md](README.md)，四组实现细节见 [ABLATION.md](examples/baselines/flow_dp3/ABLATION.md)。

旧实验数据、权重和报告已清理。新实验要求独立 XYZ 数据和 v2 checkpoint，不再迁移或加载旧权重。

## 1. 初始化本机环境

在项目根目录执行，路径不同的机器相应替换：

```bash
export MANISKILL_ROOT=/home/bigegg/ManiSkill
cd "$MANISKILL_ROOT"
source /home/bigegg/anaconda3/etc/profile.d/conda.sh
conda activate "$MANISKILL_ROOT/.runtime/env-rocm"
unset LD_LIBRARY_PATH
export PYTHONPATH="$MANISKILL_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PIP_CONSTRAINT="$MANISKILL_ROOT/.runtime/constraints-rocm.txt"
export MS_ASSET_DIR="$MANISKILL_ROOT/.runtime/maniskill"
export MPLCONFIGDIR="$MANISKILL_ROOT/.runtime/matplotlib"
export XDG_CACHE_HOME="$MANISKILL_ROOT/.runtime/cache"
export MIOPEN_USER_DB_PATH="$MANISKILL_ROOT/.runtime/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$MANISKILL_ROOT/.runtime/miopen-cache"
export VK_ICD_FILENAMES="$MANISKILL_ROOT/.runtime/mesa-env/share/vulkan/icd.d/lvp_icd.x86_64.json"
```

物理仿真为 CPU PhysX，渲染为 Mesa 软件 Vulkan，`cuda:0` 在本机表示 ROCm 显卡。显卡或渲染在沙箱内不可用时，以能访问设备的本机终端复核。

## 2. 任务与配置

| 任务 | 机器人 | EE 控制器 | state | action | 最大步数 | `configs/` 下的配置 |
| --- | --- | --- | --- | --- | --- | --- |
| PickCube-v1 | panda | pd_ee_delta_pose | 28 | 7 | 200 | pickcube.yaml |
| PushCube-v1 | panda | pd_ee_delta_pos | 25 | 4 | 200 | pushcube.yaml |
| StackCube-v1 | panda | pd_ee_delta_pose | 25 | 7 | 400 | stackcube.yaml |
| PegInsertionSide-v1 | panda_wristcam | pd_ee_delta_pose | 25 | 7 | 500 | peginsertion.yaml |
| DrawTriangle-v1 | panda_stick | pd_ee_delta_pos | 21 | 3 | 300 | drawtriangle.yaml |
| LiftPegUpright-v1 | panda | pd_ee_delta_pose | 25 | 7 | 400 | liftpegupright.yaml |
| PlaceSphere-v1 | panda | pd_ee_delta_pose | 25 | 7 | 300 | placesphere.yaml |
| PullCube-v1 | panda | pd_ee_delta_pos | 28 | 4 | 200 | pullcube.yaml |
| PullCubeTool-v1 | panda | pd_ee_delta_pose | 25 | 7 | 400 | pullcubetool.yaml |
| PickSingleYCB-v1 | panda | pd_ee_delta_pose | 28 | 7 | 300 | picksingleycb.yaml |

状态包含 qpos、qvel 和基座系 TCP 位姿；PickCube、PullCube、PickSingleYCB 额外包含指定目标位置。Panda 的 TCP 位置位于 state[18:21]，PandaStick 位于 state[14:17]，程序按契约提取，不按任务名称猜测。隐藏物体位姿、成功标签和 segmentation 不输入模型。

PickSingleYCB 固定使用官方糖盒 `004_sugar_box`，不代表完整 YCB 基准。首次运行先执行 `python -B -m mani_skill.utils.download_asset ycb`。LiftPeg 使用本地修正的长轴直立判定；这些环境设置不由本次消融改变。

以下命令以 PickCube 为例：

```bash
export FLOW_TASK=PickCube-v1
export FLOW_CONFIG=examples/baselines/flow_dp3/configs/pickcube.yaml
export FLOW_ROOT=.runtime/flow_dp3/pickcube-ablation-001
export FLOW_DATA="$FLOW_ROOT/demos.h5"
```

其他任务替换任务名和对应配置，并使用新的数据与输出路径。`ee` 别名始终表示任务表中的末端控制器。

## 3. 录制一份四组共用的数据

```bash
python -B examples/baselines/flow_dp3/prepare_demos.py \
    --env-id "$FLOW_TASK" --generate 120 --count 100 --max-attempts 2000 \
    --num-points 512 --length-scale 1.0 --video-every 10 \
    --output "$FLOW_DATA"
```

默认只录制 EE；`--control-mode ee` 可写可省略。原始专家轨迹先生成关节目标，再转换成目标 EE 控制器，真实执行后采集观测、重新判断成功。PickCube 成功后保持40步；其他任务保持规则读取任务配置。

每条 episode 保存：

| 字段 | 形状与单位 |
| --- | --- |
| `pointcloud_xyz` | `[T+1,N,3]`，直接采样的基座系 XYZ，米 |
| `pointcloud_distance` | `[T+1,N,4]`，同索引、同观测时刻的 `[p-e,‖p-e‖]/L` 缓存 |
| `state` | `[T+1,state_dim]`，当前本体状态与规定目标 |
| `action` | `[T,action_dim]`，实际执行的 EE 动作 |
| `task_metrics` | 诊断数据，不输入模型 |

根属性记录任务契约、独立 XYZ 存储版本和来源指纹。四组模型都从 `pointcloud_xyz` 构造输入，不从距离缓存反推坐标。训练数据按 episode 划分；默认100条总示范、验证比例0.2，实际约80条训练、20条验证。

采样顺序为：有效点过滤 → 世界系转基座系 → 任务 crop → 最多4096点预采样 → FPS N点。点不足直接报错，没有补零或实体保底采样。相机、crop 和采样种子保持任务预设。改变点数、crop、相机或尺度时重新录制。

默认每10条成功示范录制1条专家场景视频，第1/11/21……条保存到 `<数据名>-videos-full/ee/`。`--video-every 1` 每条录制，`--no-save-video` 关闭录像。未录视频的示范仍保存完整观测和动作。

已有新原始专家轨迹时可以只转换：

```bash
python -B examples/baselines/flow_dp3/prepare_demos.py \
    --env-id "$FLOW_TASK" --source /实际路径/trajectory.raw.h5 \
    --count 100 --num-points 512 --length-scale 1.0 --output "$FLOW_DATA"
```

原始 HDF5 旁须有同名 JSON；输出文件必须不存在。转换数量不足会保留实际结果并报错。四组共用一份转换数据，不分别录制。

## 4. 检查观测与录像

```bash
python -B visual/export_videos.py --dataset "$FLOW_DATA" \
    --inspect-only --output "$FLOW_ROOT/inspection"
python -B visual/pointcloud.py --dataset "$FLOW_DATA" --episode 0 --frame 0 \
    --export-backend matplotlib --export-view both --png "$FLOW_ROOT/frame0.png"
python -B visual/export_videos.py --dataset "$FLOW_DATA" --episodes 0 \
    --output "$FLOW_ROOT/observations"
```

同步视频真实执行保存动作，逐帧核对状态、独立 XYZ 和距离缓存。回放需要原始 HDF5 及 JSON，并核对来源指纹。更多参数见 [visual/README.md](visual/README.md)。

## 5. 四组训练

| 组别 | `pointcloud_representation` | `pointcloud_encoder_type` |
| --- | --- | --- |
| A | xyz | pointnetpp |
| B | ee_vector_distance | pointnetpp |
| C | xyz | ee_relation_pointnetpp |
| D | ee_vector_distance | ee_relation_pointnetpp |

```bash
for FLOW_GROUP in A B C D; do
    python -B examples/baselines/flow_dp3/train.py \
        --config "$FLOW_CONFIG" --ablation "$FLOW_GROUP" \
        --env-id "$FLOW_TASK" --data "$FLOW_DATA" \
        --output "$FLOW_ROOT/runs" --device cuda:0 \
        --batch-size 2 --steps 30000 --wandb-mode disabled
done
```

四份固定覆盖配置位于 `examples/baselines/flow_dp3/config/pointcloud_ablation/`；它们由 `--ablation` 合入任务 YAML，不作为完整训练 YAML 单独传入。若任务 YAML 已显式设置冲突的表示或编码器，启动时报错。省略 `--ablation` 时读取策略配置，缺省为 D。

新训练输出为 `<output>/<A|B|C|D>/seed_<training.seed>/`，日志名也包含组别和种子。默认 seed=42。修改 seed 使用新的任务 YAML；本轮沿用原有 seed 同时控制划分和训练的规则，未重新设计数据划分。

每个目录保存解析后的 `config.yaml`、`run.json`、`metrics.jsonl`、`best.pt`、`last.pt`。`run.json` 包含实际表示、编码器、输入契约、数据指纹、episode 列表、总参数量和编码器参数量。输出目录非空时拒绝覆盖。

主干仍为 `[512,1024,2048]`，本机先用 FP32、batch size 2。`--batch-size` 会覆盖 YAML 的32；更改 batch size 会改变训练量。30000步是当前候选预算，不是已验证的收敛结论。

短链路验收改用 `configs/pickcube_smoke.yaml`、`--steps 2`，只能检查接口。其他任务可用 `configs/multi_task_smoke.yaml`，仍须各自验收。

### 恢复训练

```bash
export FLOW_GROUP=D
export FLOW_RUN="$FLOW_ROOT/runs/$FLOW_GROUP/seed_42"
python -B examples/baselines/flow_dp3/train.py \
    --config "$FLOW_RUN/config.yaml" --ablation "$FLOW_GROUP" \
    --data "$FLOW_DATA" --resume "$FLOW_RUN/last.pt" \
    --output "$FLOW_ROOT/runs" --device cuda:0 --batch-size 2 \
    --steps 30000 --wandb-mode disabled
```

`--steps` 是累计目标更新数；学习率总长度仍由 YAML `training.steps` 决定。恢复须匹配配置、数据 SHA256、任务与点云契约、批大小及计算设备。只恢复到原运行目录。旧 v1 checkpoint 已停用，不猜测缺失字段，不使用 `strict=False`。

## 6. 闭环评估

```bash
export FLOW_GROUP=D
export FLOW_RUN="$FLOW_ROOT/runs/$FLOW_GROUP/seed_42"
python -B examples/baselines/flow_dp3/evaluate.py \
    --checkpoint "$FLOW_RUN/best.pt" --ablation "$FLOW_GROUP" \
    --env-id "$FLOW_TASK" --device cuda:0 \
    --episodes 100 --start-seed 10000 --policy-seed 42 \
    --reset-policy-seed-per-episode --save-video --save-trace \
    --output "$FLOW_RUN/eval.json" --wandb-mode disabled
```

A/B/C/D 分别使用各自权重和输出路径，保持相同测试种子、步数预算、动作执行长度和 EMA 选择。测试种子须与实际采集种子核对。`--ablation`、`--env-id`、`--control-mode` 只用于一致性校验，不能覆盖 checkpoint。点云相机、crop、尺度、半径和编码器结构从权重恢复。

默认使用 EMA；`--raw-weights` 使用原权重。默认两帧观测、预测16步动作、每次执行8步。`--n-action-steps` 可以显式调整执行长度；四组对比必须保持一致。默认按任务最大步数执行，瞬时成功后继续评估。

主要结果为 `success_end_rate` 和 `success_once_rate`；逐局还有误差、动作裁剪比例、保持成功比例及耗时。它们沿用原评估定义，本轮未修改统计方法。短训练的验收结果不能作为正式成功率。

## 7. 配置与记录约定

- 新点云字段：`pointcloud_representation`、`pointcloud_encoder_type`、`pointcloud_input_version`、`pointcloud_num_centers`、`pointcloud_num_neighbors`、`local_relation_mode`、`readout_mode`。
- 统一输出64维点云特征，状态MLP输出64维；拼接后送入原Flow DP3主干。
- 固定长度尺度作用于 XYZ、向量和距离；球查询半径从米同步换算。不做逐帧中心化、单位球缩放或逐轴 min-max。
- 两级中心默认128/32，邻居32/32；Pick半径0.05/0.12m，其他任务读取各自YAML。
- 状态、动作归一化继续只拟合训练episode；损失函数、优化器、学习率规则、EMA和任务成功条件没有更改。
- 配置其他参数的范围以 `PolicyConfig`、现有任务 YAML 和 `--help` 为准。配置冲突不会静默覆盖。

本地日志默认无需 W&B。安装可选依赖后可将 `--wandb-mode disabled` 改为 `online` 或 `offline`；已有日志导入使用 `import_training_logs.py --run-dir "$FLOW_RUN" --output 新目录`。W&B不自动上传数据或权重。

`.runtime/` 中的数据、权重、报告和视频均不随 Git 上传。环境依赖与实验文件分开备份；旧实验产物已经按用户要求清理，新数据必须包含独立 XYZ。
