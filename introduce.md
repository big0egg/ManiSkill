# Flow DP3：数据录制、训练与评估

本文用于已安装环境的阿里云 PPU 实例。首次安装和排错见 [README.md](README.md)。当前任务为 `PickCube-v1` / Panda / `pd_ee_delta_pos`，使用 CPU 仿真、CPU 软件 Vulkan 渲染和 PPU 模型计算。

运行顺序：初始化终端 → 准备100条成功示范 → 完整模型训练 → 评估与录像。需要继续已有训练时，使用第3.2节；参数的填写格式、可选值和约束列在对应表格中。

## 1. 初始化终端与路径

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

# 数据文件位置和训练目录分别设置。
export FLOW_DATA="$MANISKILL_ROOT/.runtime/flow_dp3/pickcube-100-v2.h5"
# pickcube-001 已有训练结果；新训练使用新编号。
export FLOW_RUN_ROOT="$MANISKILL_ROOT/.runtime/flow_dp3/pickcube-002"
mkdir -p "$FLOW_RUN_ROOT" "$MPLCONFIGDIR" "$MS_ASSET_DIR"
test -f "$VK_ICD_FILENAMES"
```

下面的命令沿用这两个变量。`FLOW_DATA` 指向真实数据文件，`FLOW_RUN_ROOT` 只负责保存训练、日志与评估输出，数据可以放在实验目录外。ICD 文件不存在时，按 README 第4节查找并设置实际路径。

| 变量 | 可填内容 / 示例 | 作用 |
| --- | --- | --- |
| `MANISKILL_ROOT` / `PYTHONPATH` | 当前仓库的绝对路径，如 `/mnt/workspace/ManiSkill` | 工作目录和源码导入路径 |
| `CUDA_VISIBLE_DEVICES` | 本实例使用 `0` | 暴露设备0；当前模型入口使用 `cuda:0` |
| `MANISKILL_VULKAN_ICD` / `VK_ICD_FILENAMES` | 真实存在的 `.json` 路径，如 `/usr/share/vulkan/icd.d/lvp_icd.json` | 选择 Mesa 软件 Vulkan |
| `PIP_CONSTRAINT` | 已生成的 `constraints-ppu.txt` 文件路径 | 保护框架依赖版本 |
| `MS_ASSET_DIR` / `MPLCONFIGDIR` | 可写目录路径 | 资产与 Matplotlib 缓存位置 |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` | 正整数，本环境沿用 `1` | CPU 数值库线程数 |
| `FLOW_DATA` | 已转换或将生成的训练 `.h5` 文件路径 | 数据采集输出和训练输入 |
| `FLOW_RUN_ROOT` | 可写目录路径，如 `…/pickcube-002` | 本次实验输出根目录；新训练换编号，恢复训练用原目录 |

## 2. 数据录制与转换

### 2.1 生成100条成功示范

工作区已经有 `pickcube-100-v2.h5` 时，下面的命令直接使用现有文件。新实例缺少数据时，会生成120条成功原始轨迹，再回放筛选100条最终成功的训练示范。

```bash
if [ ! -f "$FLOW_DATA" ]; then
    python -B examples/baselines/flow_dp3/prepare_demos.py \
        --generate 120 --count 100 --start-seed 0 \
        --max-attempts 300 --max-steps 200 \
        --num-points 512 --length-scale 1.0 \
        --crop-min 0.05 -0.8 -0.1 --crop-max 1.2 0.8 1.0 \
        --output "$FLOW_DATA"
fi

ls -lh "$FLOW_DATA" "${FLOW_DATA%.h5}.json"
```

录制使用仓库内置运动规划器。原始关节控制轨迹成功，不保证转换为末端控制后仍成功，因此生成数量预留余量，尝试次数也高于生成目标。需要重新采集时先将 `FLOW_DATA` 改成新文件名。

以 `pickcube-100-v2.h5` 为例：

| 文件 | 内容 |
| --- | --- |
| `pickcube-100-v2.raw.h5` / `.raw.json` | 原始动作、环境状态和轨迹元信息 |
| `pickcube-100-v2.h5` | 训练数据：每条轨迹有 `T+1` 帧点云 / 28维状态和 `T` 个4维动作 |
| `pickcube-100-v2.json` | 数据契约、源文件指纹、成功数量与拒绝回放记录 |

### 2.2 可选：转换已有原始轨迹

已有原始 `.raw.h5` 时可单独回放转换。以下示例使用第2.1节保留的原始轨迹，输出到实验目录，并切换后续训练的数据路径：

```bash
python -B examples/baselines/flow_dp3/prepare_demos.py \
    --source "${FLOW_DATA%.h5}.raw.h5" \
    --count 100 --max-steps 200 \
    --output "$FLOW_RUN_ROOT/demos-converted.h5"

export FLOW_DATA="$FLOW_RUN_ROOT/demos-converted.h5"
```

`--source` 和 `--generate` 二选一。原始 `.h5` 旁必须有同名 `.json`；当前支持 PickCube / Panda 的 `pd_joint_pos` 或 `pd_ee_delta_pos` 轨迹。转换完成的训练 `.h5` 直接用于训练。

### 2.3 数据参数：可以填什么

| 参数 | 默认 / 必填 | 可填内容与示例 | 意义和约束 |
| --- | --- | --- | --- |
| `--generate` | 与 `--source` 二选一 | 正整数，如 `120`、`200` | 要生成的成功原始轨迹数；须不小于 `--count` |
| `--source` | 与 `--generate` 二选一 | 已存在的原始 `.h5` 路径 | 旁边须有同名 `.json`，环境、机器人及控制模式须受支持 |
| `--output` | 必填 | 新的 `.h5` 路径，如 `"$FLOW_DATA"` | 转换后的训练数据位置；已有同名数据或清单时拒绝覆盖 |
| `--count` | `5` | 非负整数，如 `100`；`0` 表示全部 | 保留多少条成功回放；用于训练至少需要两条成功 episode |
| `--start-seed` | `0` | 非负整数，如 `0`、`100` | 原始轨迹生成的起始种子；source 模式沿用原始元信息 |
| `--max-attempts` | `100` | 正整数，如 `300`、`500` | 生成阶段最多尝试次数；应不小于 generate，并预留失败余量 |
| `--max-steps` | `200` | 整数且 ≥16，如 `200`、`300` | 环境单局步数上限及转换回放的长度过滤上限 |
| `--num-points` | `512` | 128～4096的整数，如 `512`、`1024` | 当前预采样设置下，每帧 FPS 保留的点数 |
| `--length-scale` | `1.0` | 有限正数，如 `1.0` | 相对 xyz 和距离统一除以尺度 L，须与 policy 配置相同 |
| `--crop-min` | `0.05 -0.8 -0.1` | 空格分隔的3个有限浮点数 | 基座系 xyz 裁剪下界，单位米；逐轴小于 crop-max |
| `--crop-max` | `1.2 0.8 1.0` | 空格分隔的3个有限浮点数 | 基座系 xyz 裁剪上界，单位米；裁剪后须有足够点数 |

CLI 中的裁剪参数写成 `--crop-min 0.05 -0.8 -0.1`，不要写成 YAML 列表格式 `[0.05, -0.8, -0.1]`。修改点数、裁剪或尺度后生成新数据，并使用匹配的训练配置。

生成或转换数量不足会报错并保留已有结果；检查 `.json` 的 `saved` / `rejected`，用新输出名补充采集。数据录制本身不保存 MP4，策略视频在第4节生成。

## 3. 完整模型训练与恢复

### 3.1 开始新的训练

先完成第1节环境初始化、第2节数据准备，并安装 W&B。以下命令使用完整模型、batch size 32、30000次更新；训练和评估统一记录到 `manskill` 项目。

```bash
wandb login

python -B examples/baselines/flow_dp3/train.py \
    --config examples/baselines/flow_dp3/configs/pickcube.yaml \
    --data "$FLOW_DATA" \
    --output "$FLOW_RUN_ROOT/full" --device cuda:0 \
    --batch-size 32 --steps 30000 \
    --wandb-mode online --wandb-project manskill \
    --wandb-name "${FLOW_RUN_ROOT##*/}-full" --wandb-log-every 10
```

`--output` 必须为空目录或尚未存在。完整主干为 `[512,1024,2048]`，约2.55亿参数。batch size 根据可用显存选择，第一次确定后恢复训练沿用同一值。

不接入 W&B 时将 `--wandb-mode online` 改成 `disabled`，可以跳过登录；网络受限时改成 `offline`，保留本地 SDK 日志。

### 3.2 继续已有训练

下面的命令用于将已有 `pickcube-001` 的10000步训练继续到30000步。它要求继续使用原始数据、配置、batch size 32 和 PPU 设备；如果原训练使用了其他值，按原记录填写。

```bash
export FLOW_RUN_ROOT="$MANISKILL_ROOT/.runtime/flow_dp3/pickcube-001"
export FLOW_DATA="$MANISKILL_ROOT/.runtime/flow_dp3/pickcube-100-v2.h5"

python -B examples/baselines/flow_dp3/train.py \
    --config examples/baselines/flow_dp3/configs/pickcube.yaml \
    --data "$FLOW_DATA" \
    --output "$FLOW_RUN_ROOT/full" --device cuda:0 \
    --batch-size 32 --steps 30000 \
    --resume "$FLOW_RUN_ROOT/full/last.pt" \
    --wandb-mode online --wandb-log-every 10
```

新训练和恢复训练按实际情况选择执行。`--steps 30000` 表示累计到30000步，从10000步恢复时再更新20000步。目标必须大于 checkpoint 已保存步数，已完成30000步的运行无需执行此命令。

学习率调度总长度由 YAML `training.steps=30000` 决定，CLI 提前停止不改变调度。达到总长度后学习率为0；需要不同总预算时，在新训练开始前确定配置。

恢复会加载模型、EMA、normalizer、优化器、调度器和随机状态，并核对配置、数据 SHA256、输入契约、batch size、设备类型及原输出目录。在线 W&B 恢复自动沿用原 run 身份。

### 3.3 训练参数：可以填什么

| 参数 | 默认 / 必填 | 可填内容与示例 | 意义和约束 |
| --- | --- | --- | --- |
| `--config` | `examples/baselines/flow_dp3/configs/pickcube.yaml` | 已存在且格式正确的 `.yaml` 路径 | 模型结构与训练设置；自定义时复制成新配置文件 |
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
| `config.yaml` | 使用的原始配置文件；实际 CLI batch size 等在 run.json 中查看 |
| `run.json` | 设备、参数量、数据指纹和训练 / 验证 episode |
| `metrics.jsonl` | 每步 train_loss、grad_norm、lr、ema_decay；验证时增加 val_loss |
| `last.pt` | 最近保存的可恢复 checkpoint；末步一定保存 |
| `best.pt` | 验证 loss 改善时保存的 checkpoint；任务表现通过闭环评估确认 |
| `wandb_run.json` / `wandb/` | SDK run 信息与本地日志 |

## 4. 评估、推理与视频

### 4.1 运行评估并保存 MP4

训练后执行，默认使用 `best.pt` 的 EMA 权重：

```bash
python -B examples/baselines/flow_dp3/evaluate.py \
    --checkpoint "$FLOW_RUN_ROOT/full/best.pt" --device cuda:0 \
    --episodes 20 --start-seed 1000 --policy-seed 42 --max-steps 200 \
    --save-video --video-dir "$FLOW_RUN_ROOT/videos-full" --video-fps 20 \
    --output "$FLOW_RUN_ROOT/eval-full-video.json" \
    --wandb-mode online --wandb-project manskill \
    --wandb-name "${FLOW_RUN_ROOT##*/}-full-eval"

ls -lh "$FLOW_RUN_ROOT/videos-full"/*.mp4
cat "$FLOW_RUN_ROOT/eval-full-video.json"
```

这是闭环推理：输入2帧观测，预测动作块，连续执行8步后重新规划。环境物理和点云渲染使用 CPU，策略使用 PPU。

默认生成 `seed_1000.mp4` 至 `seed_1019.mp4`，JSON 每局记录 `video_path`。200步、20 FPS 的完整录像约10秒，这个时长是视频播放时长，实际计算可能更久。上传视频到 W&B 时在评估命令中加入 `--wandb-upload-videos`。

重复评估需要新的 JSON 文件名和视频目录，以免同名文件冲突。比较模型时保持相同任务、控制模式、种子、policy seed 和步数预算。

### 4.2 评估参数：可以填什么

| 参数 | 默认 / 必填 | 可填内容与示例 | 意义和约束 |
| --- | --- | --- | --- |
| `--checkpoint` | 必填 | 已存在的本项目 `.pt` 文件，如 `full/best.pt` 或 `full/last.pt` | 加载策略、归一化统计和输入契约 |
| `--device` | `cuda:0` | 只能填 `cpu` 或 `cuda:0` | 策略计算设备；评估可跨设备加载权重 |
| `--episodes` | `10` | 正整数，如 `20`、`50`、`100` | 独立评估局数 |
| `--start-seed` | `1000` | 非负整数，如 `1000`、`2000` | 环境起始种子，后续逐局递增；种子集合应与训练数据分离 |
| `--policy-seed` | `42` | 非负整数，如 `42`、`123` | 策略采样噪声种子，与环境种子分开 |
| `--max-steps` | `200` | 正整数，如 `200`、`300` | 每局动作步数上限；瞬时成功后仍继续执行 |
| `--raw-weights` | 不写，使用 EMA | 开启写 `--raw-weights`；关闭省略 | 改用训练模型原始权重；后面不接 true/false |
| `--output` | 必填 | 新的 `.json` 文件路径 | 评估报告；已有同名报告时拒绝覆盖 |
| `--save-video` | 不写，不录像 | 开启写 `--save-video`；关闭省略 | 每局保存 MP4；后面不接 true/false |
| `--video-dir` | 报告旁 `<报告名>-videos/` | 可写目录路径 | 需要同时启用 save-video；同名种子视频不能已存在 |
| `--video-fps` | 控制频率，本任务20 | 正整数，如 `20`、`30` | 播放帧率；更改只影响播放速度，不改变控制频率 |
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

`--config` 使用的 YAML 顶层为 `policy` 和 `training`。下表列出的值是完整配置当前值；“可填值”是格式及有效范围，不保证所有组合都有相同性能。改变配置时开始新的训练，恢复原运行保持原配置。

### 5.1 `policy`：观测、网络与采样

| 字段 | 当前值 | 可填值 / 格式 | 意义及联动约束 |
| --- | --- | --- | --- |
| `horizon` | `16` | 正整数，如 `16`、`32` | 动作预测长度；须被 `2^(len(down_dims)-1)` 整除，当前3级结构须为4的倍数 |
| `n_obs_steps` | `2` | 整数，1～horizon | 观测历史帧数；与网络条件维度相关 |
| `n_action_steps` | `8` | 整数，1～`horizon-n_obs_steps+1`；当前1～15 | 每次执行的动作数，执行从预测索引 n_obs_steps-1 开始 |
| `radius1_m` / `radius2_m` | `0.10` / `0.20` | 有限正数，单位米 | 两级点云邻域半径；内部与点云一起除以 length_scale |
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

当前任务的 `state_dim` 固定为28，`action_dim` 固定为4，通常省略并使用默认值；显式填写时也只能用这两个值。编码器固定为 ee_relation_pointnetpp，输出64维。

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

训练和评估默认关闭 W&B，本文命令显式启用 online。使用网页曲线前，按 README 安装固定版本并执行 `wandb login`。

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

`python -B` 用于禁止生成 `.pyc`。路径包含空格或使用环境变量时保留命令中的双引号。开关参数采用“出现即开启、省略即关闭”的格式，数值参数则在名称后填写数值。
