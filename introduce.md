# Flow DP3：任务选择、数据录制、训练与推理

本文集中说明已安装环境的阿里云 PPU 实例如何运行 PickCube、PushCube、StackCube、PegInsertionSide、DrawTriangle。
每个任务独立录制数据、训练和评估，共用 Flow DP3 算法。使用 CPU 仿真、CPU 软件 Vulkan 渲染；策略计算可使用初始化后的 PPU 或 CPU。
首次安装和排错见 [README.md](README.md)，验证结果与后续任务分批接入计划见 [version.md](version.md)。

运行顺序：初始化终端 → 选择任务与实验路径 → 录制或转换数据 → 检查观测录像 → 训练 → 闭环推理与录像。
继续已有训练使用第3.2节；完整模型训练前可先用第3.5节做小模型流程检查。
各节参数表列出可填内容、默认值及约束；五个任务使用同一套命令，只需先设置任务变量。

## 1. 初始化终端与路径

### 1.1 初始化环境

每次打开新终端先执行：

```bash
export MANISKILL_ROOT=/mnt/workspace/ManiSkill
cd "$MANISKILL_ROOT"
source /usr/local/PPU_SDK/envsetup.sh
source .venv-ppu/bin/activate

export PYTHONPATH="$MANISKILL_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PIP_CONSTRAINT="$MANISKILL_ROOT/.runtime/constraints-ppu.txt"
export MS_ASSET_DIR="$MANISKILL_ROOT/.runtime/maniskill"
export MPLCONFIGDIR="$MANISKILL_ROOT/.runtime/matplotlib"
export MANISKILL_VULKAN_ICD=/usr/share/vulkan/icd.d/lvp_icd.json
export VK_ICD_FILENAMES="$MANISKILL_VULKAN_ICD"

mkdir -p "$MPLCONFIGDIR" "$MS_ASSET_DIR"
test -f "$VK_ICD_FILENAMES"
```

ICD 文件不存在时，按 README 第4节查找并设置实际路径。仅用 CPU 检查流程时可跳过 PPU SDK 初始化，但仍须激活虚拟环境、设置源码路径与软件 Vulkan，并在训练/评估中填写 `--device cpu`。

### 1.2 选择任务、配置和实验路径

`FLOW_TASK` 只能填下面五个完整任务名之一，大小写一致。配置文件都位于 `examples/baselines/flow_dp3/configs/`。

| `FLOW_TASK` / `--env-id` 可填值 | 机器人 | 策略控制器 | `state_dim` | `action_dim` | 默认最大步数 | 正式配置文件 |
| --- | --- | --- | --- | --- | --- | --- |
| `PickCube-v1` | panda | pd_ee_delta_pose | 28 | 7 | 200 | pickcube.yaml |
| `PushCube-v1` | panda | pd_ee_delta_pos | 25 | 4 | 200 | pushcube.yaml |
| `StackCube-v1` | panda | pd_ee_delta_pose | 25 | 7 | 400 | stackcube.yaml |
| `PegInsertionSide-v1` | panda_wristcam | pd_ee_delta_pose | 25 | 7 | 500 | peginsertion.yaml |
| `DrawTriangle-v1` | panda_stick | pd_ee_delta_pos | 21 | 3 | 300 | drawtriangle.yaml |

以下以 DrawTriangle 为例，运行其他任务只改 `FLOW_TASK`，再执行整个代码块。`FLOW_EXPERIMENT` 是自定实验编号；开始新实验时换编号。

```bash
export FLOW_TASK=DrawTriangle-v1
export FLOW_EXPERIMENT=first-v1
case "$FLOW_TASK" in
    PickCube-v1) FLOW_CONFIG_NAME=pickcube.yaml ;;
    PushCube-v1) FLOW_CONFIG_NAME=pushcube.yaml ;;
    StackCube-v1) FLOW_CONFIG_NAME=stackcube.yaml ;;
    PegInsertionSide-v1) FLOW_CONFIG_NAME=peginsertion.yaml ;;
    DrawTriangle-v1) FLOW_CONFIG_NAME=drawtriangle.yaml ;;
    *) echo "请选择上表中的完整任务名"; exit 1 ;;
esac
export FLOW_CONFIG="$MANISKILL_ROOT/examples/baselines/flow_dp3/configs/$FLOW_CONFIG_NAME"
export FLOW_DATA="$MANISKILL_ROOT/.runtime/flow_dp3/${FLOW_TASK}-${FLOW_EXPERIMENT}.h5"
export FLOW_RUN_ROOT="$MANISKILL_ROOT/.runtime/flow_dp3/${FLOW_TASK}-${FLOW_EXPERIMENT}"
mkdir -p "$FLOW_RUN_ROOT"
```

后续命令沿用这些变量。`FLOW_DATA` 指向真实训练数据，`FLOW_RUN_ROOT` 保存训练、日志和评估输出，数据可以放在实验目录外。
新 PickCube 位姿动作数据已转换为 `.runtime/flow_dp3/PickCube-v1-pose-v4.h5`。复用时先选 `FLOW_TASK=PickCube-v1`、`FLOW_EXPERIMENT=pose-v4` 并执行上面的选择块；其数据路径即为该文件。旧4维数据只用于旧实验，不用于新7维配置。
更换目录或文件名不会改变任务；已有 `drawtriangle-100-camera-v1.h5` 实际是 PickCube 数据，不能用来训练 DrawTriangle。
本文命令显式传 `--env-id "$FLOW_TASK"`，任务不匹配时会报错。

Panda/PandaWristCam 的状态为9维 qpos + 9维 qvel + 7维基座系 TCP 位姿，共25维；PandaStick 为7+7+7，共21维，没有夹爪。
PickCube 保留历史3维目标位置，共28维。四个新增任务不增加隐藏物体/目标真值，也不加入 RGB。
每帧仍为 `[512,4]` TCP 相对 XYZ 与距离，经点云编码器后与 state MLP 输出拼接；点云颜色只是可视化着色。
7维动作含位置、旋转增量和夹爪，DrawTriangle 的3维动作只有位置增量。Peg 保留腕部相机，合并相机点云后总共采样512点。

| 变量 | 可填内容 / 示例 | 作用 |
| --- | --- | --- |
| `MANISKILL_ROOT` / `PYTHONPATH` | 当前仓库的绝对路径，如 `/mnt/workspace/ManiSkill` | 工作目录和源码导入路径 |
| `CUDA_VISIBLE_DEVICES` | 本实例使用 `0` | 暴露设备0；当前模型入口使用 `cuda:0` |
| `MANISKILL_VULKAN_ICD` / `VK_ICD_FILENAMES` | 真实存在的 `.json` 路径，如 `/usr/share/vulkan/icd.d/lvp_icd.json` | 选择 Mesa 软件 Vulkan |
| `PIP_CONSTRAINT` | 已生成的 `constraints-ppu.txt` 文件路径 | 保护框架依赖版本 |
| `MS_ASSET_DIR` / `MPLCONFIGDIR` | 可写目录路径 | 资产与 Matplotlib 缓存位置 |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` | 正整数，本环境沿用 `1` | CPU 数值库线程数 |
| `FLOW_TASK` | 上表五个完整任务名之一，如 `DrawTriangle-v1` | 选择任务，显式校验数据和 checkpoint |
| `FLOW_EXPERIMENT` | 自定名称，如 `first-v1`、`camera-v2` | 区分新实验，避免覆盖输出 |
| `FLOW_CONFIG` | 上表对应 YAML 的完整路径 | 正式模型结构与训练配置；由 case 自动设置 |
| `FLOW_DATA` | 已转换或将生成的训练 `.h5` 文件路径 | 数据采集输出和训练输入 |
| `FLOW_RUN_ROOT` | 可写目录路径，如 `…/DrawTriangle-v1-first-v1` | 本次实验输出根目录；新训练换目录，恢复训练用原目录 |

### 1.3 查看帮助

下面的命令只显示参数帮助，不录制数据、不训练、不启动仿真：

```bash
python -B examples/baselines/flow_dp3/prepare_demos.py --help
python -B examples/baselines/flow_dp3/train.py --help
python -B examples/baselines/flow_dp3/evaluate.py --help
python -B visual/pointcloud.py --help
python -B visual/export_videos.py --help
```

## 2. 数据录制与转换

### 2.1 生成100条成功示范

先完成第1.2节任务选择。缺少数据时，使用该任务的仓库内置专家生成120条成功原始轨迹，再转换、筛选100条最终成功的训练示范。
已有正确任务的数据可跳过生成，但仍须通过第2.4节检查真实任务、数量和观测质量。

```bash
if [ ! -f "$FLOW_DATA" ]; then
    python -B examples/baselines/flow_dp3/prepare_demos.py \
        --env-id "$FLOW_TASK" \
        --generate 120 --count 100 --start-seed 0 \
        --max-attempts 2000 \
        --num-points 512 --length-scale 1.0 --output "$FLOW_DATA"
fi

ls -lh "$FLOW_DATA" "${FLOW_DATA%.h5}.json"
```

未填写 `--max-steps` 时按第1.2节任务表选择上限，不把其他任务也固定为200步。
录制专家使用 `pd_joint_pos`，目标控制器按任务表选择。转换仅恢复初始场景，之后真实执行动作并重新判断最终成功。
PickCube 的新 v4 接口保留专家的旋转示范，动作依次为 XYZ 平移增量、三维旋转增量、夹爪；旧 v1/v2 为4维平移动作。转换时必须重新采集实际执行产生的点云，不能把旧4维数据补成7维。
原始轨迹成功不保证转换后仍成功，120条和2000次尝试只是预留余量，不保证最终获得100条；不足时会报错并保留已有结果。
新录制使用独立文件名，不覆盖旧数据。

| 文件 | 内容 |
| --- | --- |
| `<输出名>.raw.h5` / `.raw.json` | generate 模式生成的原始动作、环境状态和轨迹元信息 |
| `FLOW_DATA` 指定的 `.h5` | 每条轨迹有 `T+1` 帧点云/状态和 `T` 个动作；状态和动作维度见第1.2节 |
| 同名 `.json` | 任务契约、完整相机参数、源文件路径与指纹、成功数量与拒绝回放记录 |

`--generate` 模式在输出文件旁生成同名 `.raw.h5` / `.raw.json`；`--source` 模式复用已有原始文件，实际路径记录于 manifest，可视化回放按此路径查找。

### 2.2 可选：转换已有原始轨迹

已有对应任务的原始 `.h5` 时可单独转换，替代第2.1节生成命令。先把 `FLOW_SOURCE` 替换成实际文件路径，并确保 `FLOW_DATA` 指向尚未存在的输出。

```bash
export FLOW_SOURCE=/实际路径/trajectory.h5
python -B examples/baselines/flow_dp3/prepare_demos.py \
    --env-id "$FLOW_TASK" --source "$FLOW_SOURCE" \
    --count 100 --num-points 512 --length-scale 1.0 \
    --output "$FLOW_DATA"
```

例如复用 PickCube 的现有原始轨迹时，选择 `FLOW_TASK=PickCube-v1`，再设置 `FLOW_SOURCE="$MANISKILL_ROOT/.runtime/flow_dp3/PickCube-v1-first-v1.raw.h5"`，输出使用尚未存在的新文件名。
其他任务须提供各自真实原始数据。`--source` 和 `--generate` 二选一，不应连续写入同一个输出。
原始 `.h5` 旁必须有同名 `.json`，其中环境和机器人须匹配第1.2节任务表。
支持从 `pd_joint_pos` 转换，或直接回放该任务对应的策略控制模式；转换完成的训练 `.h5` 直接用于训练。

### 2.3 数据参数：可以填什么

| 参数 | 默认 / 必填 | 可填内容与示例 | 意义和约束 |
| --- | --- | --- | --- |
| `--env-id` | 生成默认 `PickCube-v1`；转换从来源 JSON 读取 | 第1.2节五个完整任务名之一，如 `DrawTriangle-v1` | 显式填写时核对真实来源任务；只改文件名不会改变任务 |
| `--generate` | 与 `--source` 二选一 | 正整数，如 `120`、`200` | 要生成的成功原始轨迹数；须不小于 `--count` |
| `--source` | 与 `--generate` 二选一 | 已存在的原始 `.h5` 路径 | 旁边须有同名 `.json`，环境、机器人及控制模式须受支持 |
| `--output` | 必填 | 新的 `.h5` 路径，如 `"$FLOW_DATA"` | 转换后的训练数据位置；已有同名数据或清单时拒绝覆盖 |
| `--count` | `5` | 非负整数，如 `100`；`0` 表示全部 | 保留多少条成功回放；用于训练至少需要两条成功 episode |
| `--start-seed` | `0` | 非负整数，如 `0`、`100` | 原始轨迹生成的起始种子；source 模式沿用原始元信息 |
| `--max-attempts` | `100` | 正整数，如 `300`、`2000` | 生成阶段最多尝试次数；应不小于 generate，并预留失败余量 |
| `--max-steps` | 按第1.2节任务表 | 整数且 ≥16，如 `200`、`400`、`500`；Draw不得超过300 | 环境单局步数上限及转换回放上限；专家超限按失败重试 |
| `--num-points` | `512` | 128～4096的整数，如 `512`、`1024` | 当前预采样设置下，每帧 FPS 保留的点数 |
| `--length-scale` | `1.0` | 有限正数，如 `1.0` | 相对 xyz 和距离统一除以尺度 L，须与 policy 配置相同 |
| `--crop-min` | 按任务操作区预设 | 空格分隔的3个有限浮点数，如 `0.44 -0.23 -0.03` | 基座系 xyz 裁剪下界，单位米；逐轴小于 crop-max |
| `--crop-max` | 按任务操作区预设 | 空格分隔的3个有限浮点数，如 `0.79 0.25 0.52` | 基座系 xyz 裁剪上界，单位米；裁剪后须有足够点数 |

CLI 中的裁剪参数写成 `--crop-min 0.44 -0.23 -0.03`，不要写成 YAML 列表格式。
下列预设在世界系转基座系后、减 TCP 前应用，单位米；相机位置和注视点采用世界坐标。
基座系范围优先目标和末端，允许裁去部分机械臂并移除地面。

| 任务 | 固定相机位置 eye | 注视点 target | 分辨率 / FOV | crop-min | crop-max |
| --- | --- | --- | --- | --- | --- |
| PickCube-v1 | `(0.30,-0.30,0.35)` | `(0,0,0.12)` | 256×256 / 60° | `(0.44,-0.23,-0.03)` | `(0.79,0.25,0.52)` |
| PushCube-v1 | `(0.30,-0.30,0.35)` | `(0.15,0,0.08)` | 256×256 / 75° | `(0.40,-0.23,-0.03)` | `(1.06,0.25,0.52)` |
| StackCube-v1 | `(0.30,0.35,0.28)` | `(0,0,0.07)` | 256×256 / 65° | `(0.36,-0.36,-0.03)` | `(0.87,0.36,0.52)` |
| PegInsertionSide-v1 | `(0.30,-0.35,0.55)` | `(0,0.10,0.12)` | 256×256 / 75° | `(0.34,-0.40,-0.03)` | `(0.90,0.62,0.52)` |
| DrawTriangle-v1 | `(0.25,-0.40,0.50)` | `(-0.10,-0.10,0.04)` | 256×256 / 60° | `(0.28,-0.35,-0.03)` | `(0.80,0.18,0.52)` |

Peg 还保留安装在 `camera_link` 上的128×128、90°腕部相机。
后续最多4096点预采样与 FPS 仍会减少点数；512是合并后整个场景的总点数，不是各物体或各相机分别512点。
DrawTriangle 的轮廓/画迹辨识与 FPS 漏点需通过录像和训练实验评估。

修改相机、点数、裁剪或尺度后，重新转换原始轨迹或生成新数据，并从头训练匹配模型。
PickCube 新数据使用 v4 契约（7维位姿动作），四个其他任务继续使用 v3；均保存完整相机参数，评估及视频回放按 checkpoint/数据创建环境。
历史 PickCube v1/v2 读取路径保留，继续使用4维平移动作；v1 模型还恢复旧128×128相机和旧范围。恢复旧实验必须沿用旧数据及训练目录保存的配置，旧权重不能直接切换成7维。

生成或转换数量不足会报错并保留已有结果；检查 `.json` 的 `saved` / `rejected`，用新输出名补充采集。
数据录制本身不保存 MP4；专家数据观测录像按第2.4节导出，训练策略的闭环录像在第4节生成。

### 2.4 检查数据、点云和观测录像

所有任务使用同一套可视化入口，由 HDF5 契约决定任务和维度。服务器无桌面时先导出双图 PNG 和一条同步 MP4：

```bash
python -B visual/export_videos.py \
    --dataset "$FLOW_DATA" --inspect-only \
    --output "$FLOW_RUN_ROOT/data-inspection"

python -B visual/pointcloud.py \
    --dataset "$FLOW_DATA" --episode 0 --frame 0 --export-view both \
    --export-backend matplotlib --png "$FLOW_RUN_ROOT/cloud-frame0.png"

python -B visual/export_videos.py \
    --dataset "$FLOW_DATA" --episodes 0 \
    --output "$FLOW_RUN_ROOT/observations"
```

`--inspect-only` 只检查文件；同步 MP4 则真实执行保存动作、重新渲染 RGB，并逐帧核对点云和状态。
源 `.raw.h5` 与同名 JSON 须仍可从记录位置访问；迁移后可在 `export_videos.py` 命令中加入 `--raw-source /实际路径/trajectory.raw.h5`，仍会核对来源指纹。
`--episodes 0` 是数据轨迹索引，可改为 `--episodes 0 1 2`；它不同于评估命令中表示局数的 `--episodes 20`。
双图显示原 FPS 点云及矢量距离；新增任务没有 GOAL 字段，不显示虚构目标，同步视频展示 TCP XYZ。
输出拒绝覆盖，重复导出换新路径。交互选点仍需要桌面和 Open3D，完整 PNG/MP4/交互参数见 [visual/README.md](visual/README.md)。

## 3. 完整模型训练与恢复

### 3.1 开始新的训练

先完成第1节任务选择和第2节数据准备。以下命令使用任务对应的正式配置、batch size 32、30000次更新；训练和评估统一记录到 `manskill` 项目。
使用 online 日志前按 README 安装 W&B 并登录；不需要 W&B 时跳过登录，将命令中的 `online` 改为 `disabled`。

```bash
wandb login

python -B examples/baselines/flow_dp3/train.py \
    --config "$FLOW_CONFIG" \
    --env-id "$FLOW_TASK" --data "$FLOW_DATA" \
    --output "$FLOW_RUN_ROOT/full" --device cuda:0 \
    --batch-size 32 --steps 30000 \
    --wandb-mode online --wandb-project manskill \
    --wandb-name "${FLOW_RUN_ROOT##*/}-full" --wandb-log-every 10
```

`--output` 必须为空目录或尚未存在。正式配置的完整主干为 `[512,1024,2048]`，约2.55亿参数。
batch size 根据可用显存选择，第一次确定后恢复训练沿用同一值。
五份正式配置是各任务的候选起点，尚未宣称经过性能调优。
state/action 归一化只统计训练 episode；点云保持契约指定的物理尺度，编码器、拼接方式、Flow DP3 损失与采样公式保持一致。

单个完整 checkpoint 约4 GB，`last.pt`、`best.pt` 和临时写入均占空间。
保存前会估算空间，失败时保留上一次成功文件；正式训练多个任务前需准备足够存储。
`best.pt` 仍是完整 checkpoint，尚未改成轻量推理权重。

不接入 W&B 时将 `--wandb-mode online` 改成 `disabled`，可以跳过登录；网络受限时改成 `offline`，保留本地 SDK 日志。

### 3.2 继续已有训练

五个任务都用下面的恢复命令。先把 `FLOW_TASK`、`FLOW_DATA`、`FLOW_RUN_ROOT` 设置为原实验的真实任务、数据和目录；
恢复时不用第1.2节的新实验编号创建另一个目录。
示例假定原训练使用 batch size 32、PPU 和总预算30000步；原实验不同则按原记录填写。
使用训练目录内保存的解析后配置，避免后续修改候选配置影响旧实验。

```bash
python -B examples/baselines/flow_dp3/train.py \
    --config "$FLOW_RUN_ROOT/full/config.yaml" \
    --env-id "$FLOW_TASK" --data "$FLOW_DATA" \
    --output "$FLOW_RUN_ROOT/full" --device cuda:0 \
    --batch-size 32 --steps 30000 \
    --resume "$FLOW_RUN_ROOT/full/last.pt" \
    --wandb-mode online --wandb-log-every 10
```

新训练和恢复训练按实际情况选择执行。`--steps 30000` 表示累计到30000步，从10000步恢复时再更新20000步。目标必须大于 checkpoint 已保存步数，已完成30000步的运行无需执行此命令。

学习率调度总长度由 YAML `training.steps=30000` 决定，CLI 提前停止不改变调度。达到总长度后学习率为0；需要不同总预算时，在新训练开始前确定配置。

恢复会加载模型、EMA、normalizer、优化器、调度器和随机状态，并核对配置、数据 SHA256、输入契约、batch size、设备类型及原输出目录。在线 W&B 恢复自动沿用原 run 身份。

例如恢复历史 `pickcube-001`，先执行下面的变量设置，再执行上面的恢复命令；原训练设置仍以 `run.json` 为准：

```bash
export FLOW_TASK=PickCube-v1
export FLOW_RUN_ROOT="$MANISKILL_ROOT/.runtime/flow_dp3/pickcube-001"
export FLOW_DATA="$MANISKILL_ROOT/.runtime/flow_dp3/pickcube-100-v2.h5"
```

### 3.3 训练参数：可以填什么

| 参数 | 默认 / 必填 | 可填内容与示例 | 意义和约束 |
| --- | --- | --- | --- |
| `--config` | `examples/baselines/flow_dp3/configs/pickcube.yaml` | 已存在且格式正确的 `.yaml` 路径 | 模型结构与训练设置；自定义时复制成新配置文件 |
| `--env-id` | 从 HDF5 契约读取 | 第1.2节五个完整任务名之一 | 可选任务校验；显式指定须与数据匹配，不能强制切换任务 |
| `--data` | 必填 | 已转换的训练 `.h5` 路径，如 `"$FLOW_DATA"` | 文件须真实存在，至少包含两条成功 episode |
| `--output` | 必填 | 可写目录，如 `"$FLOW_RUN_ROOT/full"` | 新训练目录须为空；恢复须是原 checkpoint 所在目录 |
| `--device` | `cuda:0` | 只能填 `cpu` 或 `cuda:0` | 策略计算设备；本 PPU 实例使用 cuda:0，仿真和渲染仍用 CPU |
| `--steps` | YAML `training.steps` | 正整数，如 `10000`、`30000` | 目标累计更新次数；恢复须大于已有步数，通常不超过调度总长度 |
| `--batch-size` | YAML `training.batch_size`，当前32 | 正整数，如 `8`、`16`、`32` | 每次更新的序列窗口数，须适合显存；恢复必须保持原值 |
| `--resume` | 不填，从头训练 | 已存在的本项目 `.pt` 路径 | 如 `"$FLOW_RUN_ROOT/full/last.pt"`；配置、数据与设备须匹配 |
| `--wandb-log-every` | `10` | 正整数，如 `1`、`10`、`100` | W&B 记录间隔；首步、验证和末步仍记录 |

### 3.4 训练结果位置

```bash
tail -n 5 "$FLOW_RUN_ROOT/full/metrics.jsonl"
cat "$FLOW_RUN_ROOT/full/run.json"
```

| 文件 | 内容 |
| --- | --- |
| `config.yaml` | 解析并补齐任务状态/动作维度后的配置；实际 CLI batch size 等在 run.json 中查看 |
| `run.json` | 设备、参数量、数据指纹和训练 / 验证 episode |
| `metrics.jsonl` | 每步 train_loss、grad_norm、lr、ema_decay；验证时增加 val_loss |
| `last.pt` | 最近保存的可恢复 checkpoint；末步一定保存 |
| `best.pt` | 验证 loss 改善时保存的 checkpoint；任务表现通过闭环评估确认 |
| `wandb_run.json` / `wandb/` | SDK run 信息与本地日志 |

### 3.5 可选：先做小模型流程检查

同一烟雾配置可用于五个任务，自动从数据契约绑定状态和动作维度。
运行前须有至少两条成功转换示范，`smoke` 目录须为空或尚未存在；此处不使用正式配置。

```bash
python -B examples/baselines/flow_dp3/train.py \
    --config examples/baselines/flow_dp3/configs/multi_task_smoke.yaml \
    --env-id "$FLOW_TASK" --data "$FLOW_DATA" \
    --output "$FLOW_RUN_ROOT/smoke" --device cpu \
    --steps 20 --wandb-mode disabled

python -B examples/baselines/flow_dp3/evaluate.py \
    --checkpoint "$FLOW_RUN_ROOT/smoke/best.pt" --env-id "$FLOW_TASK" \
    --device cpu --episodes 1 --start-seed 1000 --max-steps 16 \
    --save-video --video-dir "$FLOW_RUN_ROOT/smoke-videos" \
    --output "$FLOW_RUN_ROOT/smoke-eval.json" --wandb-mode disabled
```

该配置使用 `[64,128,256]` 小主干、batch size 2，仅检查训练和推理是否可运行。
20次更新和16步推理不能代表任务成功率；正式训练仍使用第1.2节匹配的配置及完整任务步数。

## 4. 评估、推理与视频

### 4.1 运行评估并保存 MP4

训练后执行，默认使用 `best.pt` 的 EMA 权重：

```bash
python -B examples/baselines/flow_dp3/evaluate.py \
    --checkpoint "$FLOW_RUN_ROOT/full/best.pt" --env-id "$FLOW_TASK" --device cuda:0 \
    --episodes 20 --start-seed 1000 --policy-seed 42 \
    --save-video --video-dir "$FLOW_RUN_ROOT/videos-full" --video-fps 20 \
    --output "$FLOW_RUN_ROOT/eval-full-video.json" \
    --wandb-mode online --wandb-project manskill \
    --wandb-name "${FLOW_RUN_ROOT##*/}-full-eval"

ls -lh "$FLOW_RUN_ROOT/videos-full"/*.mp4
cat "$FLOW_RUN_ROOT/eval-full-video.json"
```

这是闭环推理：输入2帧观测，预测动作块，连续执行8步后重新规划。环境物理和点云渲染使用 CPU，策略使用 PPU。

实际任务、机器人、控制器、相机、crop、状态和动作维度均从 checkpoint 读取，启动日志和报告记录真实任务。
`--env-id` 只核对期望任务，不能把旧 PickCube 权重切换成 DrawTriangle 等其他任务。
未填写 `--max-steps` 时按第1.2节任务表运行；DrawTriangle 不允许超过300步。

上述命令生成 `seed_1000.mp4` 至 `seed_1019.mp4`，JSON 每局记录 `video_path`。
以 PickCube 的200步、20 FPS 为例，完整录像约10秒；其他任务按实际步数变化。这是视频播放时长，实际计算可能更久。
上传视频到 W&B 时在评估命令中加入 `--wandb-upload-videos`。

重复评估需要新的 JSON 文件名和视频目录，以免同名文件冲突。比较模型时保持相同任务、控制模式、种子、policy seed 和步数预算。

### 4.2 评估参数：可以填什么

| 参数 | 默认 / 必填 | 可填内容与示例 | 意义和约束 |
| --- | --- | --- | --- |
| `--checkpoint` | 必填 | 已存在的本项目 `.pt` 文件，如 `full/best.pt` 或 `full/last.pt` | 加载策略、归一化统计和输入契约 |
| `--env-id` | 从 checkpoint 读取 | 第1.2节五个完整任务名之一 | 可选任务校验，显式指定须匹配权重；不能强制覆盖环境 |
| `--device` | `cuda:0` | 只能填 `cpu` 或 `cuda:0` | 策略计算设备；评估可跨设备加载权重 |
| `--episodes` | `10` | 正整数，如 `20`、`50`、`100` | 独立评估局数 |
| `--start-seed` | `1000` | 非负整数，如 `1000`、`2000` | 环境起始种子，后续逐局递增；种子集合应与训练数据分离 |
| `--policy-seed` | `42` | 非负整数，如 `42`、`123` | 策略采样噪声种子，与环境种子分开 |
| `--max-steps` | 按第1.2节任务表 | 正整数，如 `200`、`400`、`500`；Draw不得超过300 | 每局动作步数上限；瞬时成功后仍继续执行 |
| `--raw-weights` | 不写，使用 EMA | 开启写 `--raw-weights`；关闭省略 | 改用训练模型原始权重；后面不接 true/false |
| `--output` | 必填 | 新的 `.json` 文件路径 | 评估报告；已有同名报告时拒绝覆盖 |
| `--save-video` | 不写，不录像 | 开启写 `--save-video`；关闭省略 | 每局保存 MP4；后面不接 true/false |
| `--video-dir` | 报告旁 `<报告名>-videos/` | 可写目录路径 | 需要同时启用 save-video；同名种子视频不能已存在 |
| `--video-fps` | 环境控制频率 | 正整数，如 `20`、`30` | 播放帧率；更改只影响播放速度，不改变控制频率 |
| `--wandb-upload-videos` | 不写，不上传 | 开启写 `--wandb-upload-videos`；关闭省略 | 要求同时启用 save-video 和 online / offline 日志 |

评估的相机、裁剪、点数与尺度从 checkpoint 读取。这些参数不在 evaluate.py 的 CLI 中重新填写。

### 4.3 查看结果

| 位置 / 指标 | 含义 |
| --- | --- |
| 终端与 JSON `episodes` | 每局状态、步数、推理次数、耗时及动作裁剪比例 |
| `success_once_rate` | 至少成功过一次的局数比例，范围0～1 |
| `success_end_rate` | 最后仍成功的局数比例，范围0～1 |
| `videos-full/` | 机器人闭环执行 MP4，可在 DSW 文件浏览器中下载 |
| W&B 评估 run | 每局 eval/* 指标与总体成功率；上传视频后也可查看画面 |

`.runtime/` 是隐藏目录，在 DSW 文件浏览器中开启显示隐藏文件。运行结束与任务成功是两种结果：是否完成任务看成功率字段和视频。

## 5. YAML 参数：可以填什么

`--config` 使用的 YAML 顶层为 `policy` 和 `training`。五份正式配置共享下表网络与训练起点，任务维度及半径按表中任务差异填写；
烟雾配置采用第3.5节的小模型设置。“可填值”是格式及有效范围，不保证所有组合都有相同性能。
改变配置时开始新的训练，恢复原运行保持原配置。

### 5.1 `policy`：观测、网络与采样

| 字段 | 当前值 | 可填值 / 格式 | 意义及联动约束 |
| --- | --- | --- | --- |
| `state_dim` | Pick 28；Push/Stack/Peg 25；Draw 21 | 第1.2节任务对应的整数 | 可省略，由训练数据契约绑定；显式填写须与任务匹配 |
| `action_dim` | 新 Pick/Stack/Peg 7；Push 4；Draw 3 | 第1.2节任务对应的整数 | Pick 正式/烟雾配置显式要求7维，拒绝旧4维数据；其他任务可由契约绑定，旧 Pick 恢复使用原配置 |
| `horizon` | `16` | 正整数，如 `16`、`32` | 动作预测长度；须被 `2^(len(down_dims)-1)` 整除，当前3级结构须为4的倍数 |
| `n_obs_steps` | `2` | 整数，1～horizon | 观测历史帧数；与网络条件维度相关 |
| `n_action_steps` | `8` | 整数，1～`horizon-n_obs_steps+1`；当前1～15 | 每次执行的动作数，执行从预测索引 n_obs_steps-1 开始 |
| `radius1_m` / `radius2_m` | Pick / 策略缺省 `0.05` / `0.12`；其他任务配置 `0.10` / `0.20` | 有限正数，单位米 | 两级点云邻域半径；内部与点云一起除以 length_scale；新半径可复用点云数据，旧 checkpoint 仍读取保存值 |
| `length_scale` | `1.0` | 有限正数 | 必须等于数据采集 length-scale |
| `down_dims` | `[512,1024,2048]` | 至少2项的正整数列表，如 `[512, 1024, 2048]` | U-Net 通道宽度；所有项可被 n_groups 整除，第一项还须为8的倍数 |
| `diffusion_step_embed_dim` | `128` | ≥4的偶数，如 `64`、`128`、`256` | 时间条件编码维度 |
| `kernel_size` | `5` | 正奇数，如 `3`、`5`、`7` | 时序卷积核大小 |
| `n_groups` | `8` | 正整数，如 `4`、`8`、`16`，须整除所有 down_dims | GroupNorm 分组数 |
| `num_inference_steps` | `10` | 正整数，如 `10`、`20` | 每次动作块生成的积分更新次数 |
| `solver` | `consistency` | 只能填 `consistency` 或 `euler` | 推理采样公式 |
| `fm_eps` | `0.01` | 浮点数，严格在0～1之间 | 训练与采样的起始时间下界 |
| `fm_time_scale` | `100.0` | 有限正数 | 网络时间条件缩放 |
| `num_segments` | `2` | 正整数，如 `1`、`2`、`4` | 一致性目标的时间分段数 |
| `boundary` | `1` | 0～1的数，包含端点，如 `0`、`0.5`、`1` | 一致性目标的边界阈值 |
| `delta` | `0.01` | 浮点数，严格在0～1之间 | 两时刻预测的时间间隔 |
| `alpha` | `0.00001` | 有限非负数，如 `0`、`0.00001` | 速度一致性损失权重 |
| `noise_scale` | `1.0` | 有限正数，如 `1.0` | 训练起始与推理初始噪声尺度 |
| `sigma_var` | `0.0` | 有限非负数，如 `0.0`、`0.1` | consistency 公式的额外随机扰动系数 |

四个新增任务的正式配置显式记录维度；`pickcube.yaml` 与 `multi_task_smoke.yaml` 省略维度，由训练数据绑定。
所有配置显式填写的维度都须与契约匹配，冲突会报错。
编码器固定为 ee_relation_pointnetpp，输出64维，state MLP 输出64维，拼接为每帧128维；默认两帧展开为256维策略条件。

### 5.2 `training`：优化与数据划分

| 字段 | 当前值 | 可填值 / 格式 | 意义及限制 |
| --- | --- | --- | --- |
| `seed` | `42` | 非负整数，如 `42`、`123` | 训练随机流与 episode 划分种子 |
| `batch_size` | `32` | 正整数，如 `8`、`16`、`32` | 每次更新的窗口数；CLI 可覆盖，恢复须保持原值 |
| `steps` | `30000` | 正整数，如 `10000`、`30000` | 默认总更新数和学习率调度长度 |
| `lr` | `0.0001` | 有限正数，如 `0.0001`、`0.00005` | AdamW 基础学习率 |
| `betas` | `[0.95,0.999]` | 两项浮点数列表，每项满足0≤值<1，如 `[0.95, 0.999]` | AdamW 一阶 / 二阶矩衰减系数 |
| `weight_decay` | `0.000001` | 有限非负数，如 `0`、`0.000001` | AdamW 权重衰减 |
| `warmup_steps` | `500` | 非负整数，如 `0`、`500`，通常小于 steps | 预热更新次数，随后 cosine 调度 |
| `val_ratio` | `0.2` | 浮点数，严格在0～1之间，如 `0.1`、`0.2` | 按整条 episode 留出的验证比例 |
| `val_every` | `100` | 正整数，如 `100`、`500` | 验证间隔，最后一步也验证 |
| `val_batches` | `4` | 正整数，如 `4`、`8` | 每次验证取平均的 batch 数 |
| `checkpoint_every` | `100` | 正整数，如 `100`、`500` | 定期保存间隔，验证改善及末步也保存 |
| `grad_clip` | `1.0` | 有限正数，如 `0.5`、`1.0` | 梯度范数裁剪阈值 |

YAML 列表用方括号和逗号，例如 `betas: [0.95, 0.999]`。CLI 参数名使用连字符，YAML 字段名使用下划线。training 必须保留表中全部12个字段，不能添加脚本未支持的字段。

## 6. W&B 参数与日志操作

训练和评估默认关闭 W&B，本文命令显式启用 online。使用网页曲线前，按 README 安装固定版本 `wandb 0.22.3` 并执行 `wandb login`；该版本支持新生成的86字符 API key，旧版环境升级后使用 `wandb login --relogin`。

| 参数 | 默认 | 可填内容 / 示例 | 意义及限制 |
| --- | --- | --- | --- |
| `--wandb-mode` | 训练 / 评估 `disabled`；历史导入 `online` | 只能填 `disabled`、`online`、`offline` | online 实时上传；offline 留待同步；历史导入不支持 disabled |
| `--wandb-project` | `flow_dp3`；恢复沿用原项目 | 项目名称字符串，如 `manskill`、`flow_dp3` | 本文统一使用 manskill，恢复原 run 时项目须匹配 |
| `--wandb-entity` | 登录账户默认 workspace，恢复沿用原值 | 账号或团队的真实 workspace 名称 | 从 W&B workspace 地址中取名称；有权限才能写入，通常可省略 |
| `--wandb-name` | 输出目录名 | 显示名称字符串，如 `pickcube-002-full` | 本文根据实验目录自动拼接名称 |
| `--wandb-run-id` | 自动生成，恢复沿用 | 原 run ID 字符串，如 `238148c3` | 通常省略；恢复显式填写时必须与保存的 ID 一致 |

W&B 默认不自动上传示范或权重；MP4 上传通过评估的 `--wandb-upload-videos` 开启。训练图横轴看 `global_step`，评估总体成绩看 `eval/success_once_rate` 和 `eval/success_end_rate`。

### 6.1 导入已有训练记录

需要将已有本地 metrics.jsonl 导入 W&B 时执行：

```bash
python -B examples/baselines/flow_dp3/import_training_logs.py \
    --run-dir "$FLOW_RUN_ROOT/full" \
    --output "$FLOW_RUN_ROOT/history" \
    --wandb-mode online --wandb-project manskill \
    --wandb-name "${FLOW_RUN_ROOT##*/}-history"
```

| 参数 | 默认 / 必填 | 可填内容 | 意义及约束 |
| --- | --- | --- | --- |
| `--run-dir` | 必填 | 已存在的训练目录，如 `"$FLOW_RUN_ROOT/full"` | 必须包含 metrics.jsonl、config.yaml、run.json |
| `--output` | 原训练目录的 `wandb-history/` | 可写的新目录，如 `"$FLOW_RUN_ROOT/history"` | 导入日志位置；目录须为空，重复导入换新目录 |

历史导入创建独立 W&B run，直接导入原更新记录，无需重新训练。

### 6.2 同步离线日志

训练命令将 `--wandb-mode online` 改为 `offline` 时，SDK 日志位于 `full/wandb/`。联网后下面的命令会查找并同步实际存在的离线目录：

```bash
wandb login
if [ -d "$FLOW_RUN_ROOT/full/wandb" ]; then
    find "$FLOW_RUN_ROOT/full/wandb" \
        -maxdepth 1 -type d -name 'offline-run-*' -print0 |
    while IFS= read -r -d '' FLOW_OFFLINE_RUN; do
        wandb sync "$FLOW_OFFLINE_RUN"
    done
fi
```

离线恢复沿用逻辑 run ID，每次执行产生新的离线目录，需逐个同步。历史导入的离线日志位于 `history/wandb/`，同步时相应替换查找目录。

## 7. 保存与迁移

数据、checkpoint、日志和视频都在 `.runtime/`，该目录被 Git 忽略。备份或换实例时单独传输数据和完整训练目录；部署代码使用当前仓库和 README 中的环境步骤。

同步观测录像还依赖原始 HDF5 和同名 JSON，迁移时一并保留并检查 manifest 的来源路径。
本版已验证四个新增任务的小规模 CPU 采集、转换、训练恢复、闭环推理和录像，尚未完成正式100条数据训练的成功率评测。
后续分批接入任务、验收标准和已知验证限制统一记录在 [version.md](version.md)；新任务通过验收后更新本文任务表、配置映射及对应参数说明。

`python -B` 用于禁止生成 `.pyc`。路径包含空格或使用环境变量时保留命令中的双引号。开关参数采用“出现即开启、省略即关闭”的格式，数值参数则在名称后填写数值。
