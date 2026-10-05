# Flow DP3 实验操作与参数说明

本文用于已安装环境后的日常实验。首次安装、依赖版本和报错处理见 [README.md](README.md)。所有命令从 `/mnt/workspace/ManiSkill` 运行。

当前支持 `PickCube-v1` / Panda / `pd_ee_delta_pos`，使用 CPU 仿真、CPU 软件 Vulkan 点云渲染与 PPU 策略计算。数据“录制”指使用内置运动规划器生成并回放专家轨迹。

操作顺序：初始化终端 → 录制成功示范 → 小模型训练 → 评估和录像 → 扩大数据与完整模型训练。训练曲线通过 W&B 查看，闭环执行结果通过 JSON 和 MP4 查看。

第一次验证依次执行第1、2.1、3.1、4.1节；通过后，再按第2.2、3.2、4.2节开展完整模型实验。第3.3节用于继续同一次完整模型训练，其余参数表按需查阅。

## 1. 初始化终端与实验目录

每次打开新终端，执行以下命令。W&B 已安装时，首次在线运行前执行 `wandb login`。

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

# 每次新实验修改编号；恢复训练继续使用原编号。
export FLOW_RUN_ROOT="$MANISKILL_ROOT/.runtime/flow_dp3/pickcube-001"
mkdir -p "$FLOW_RUN_ROOT" "$MPLCONFIGDIR" "$MS_ASSET_DIR"
test -f "$VK_ICD_FILENAMES"
```

ICD 路径应以实例实际文件为准；如果文件不存在，按 README 第4节查找。下文所有输出集中到 `FLOW_RUN_ROOT`，不会覆盖已有的 `ppu-smoke` 结果。

| 环境变量 | 作用 |
| --- | --- |
| `MANISKILL_ROOT` / `PYTHONPATH` | 仓库根目录与本仓库的 Python 导入路径 |
| `CUDA_VISIBLE_DEVICES=0` | 选择设备0；在本镜像中，`cuda:0` 对应 PPU 兼容接口 |
| `VK_ICD_FILENAMES` | 显式选择 Mesa 软件 Vulkan，用于点云与视频渲染 |
| `PIP_CONSTRAINT` | 后续安装时保护原有 PyTorch、NumPy 等版本 |
| `MS_ASSET_DIR` / `MPLCONFIGDIR` | 资产与 Matplotlib 缓存保存在本仓库运行目录 |
| `OMP_NUM_THREADS` 等 | 限制 CPU 数值库线程，沿用已验证设置 |
| `FLOW_RUN_ROOT` | 本次实验的输出根目录；新实验换编号，恢复训练用原目录 |

## 2. 数据录制与转换

### 2.1 先录制5条成功示范

```bash
python -B examples/baselines/flow_dp3/prepare_demos.py \
    --generate 6 --count 5 --start-seed 0 \
    --max-attempts 100 --max-steps 200 --num-points 512 \
    --length-scale 1.0 \
    --output "$FLOW_RUN_ROOT/demos-smoke.h5"
```

程序分两步：先在无渲染 CPU 环境生成6条成功的关节控制轨迹，再用 `pd_ee_delta_pos` 真实回放并采集点云，保留5条最终成功的回放。原始成功轨迹不保证动作转换后仍成功，因此预留一条余量；不够时增加 `--generate`。

| 生成文件 | 内容 |
| --- | --- |
| `demos-smoke.raw.h5` / `demos-smoke.raw.json` | 原始动作、环境状态与轨迹元信息 |
| `demos-smoke.h5` | 可直接用于 FlowDP3 训练的数据 |
| `demos-smoke.json` | 数据契约、源文件指纹、成功数量及拒绝回放记录 |

数据录制默认不生成 MP4。查看策略执行视频按第4节评估录制。

### 2.2 准备20条示范用于完整模型

小模型链路通过后，再运行：

```bash
python -B examples/baselines/flow_dp3/prepare_demos.py \
    --generate 30 --count 20 --start-seed 0 \
    --max-attempts 100 --max-steps 200 \
    --output "$FLOW_RUN_ROOT/demos-20.h5"
```

当前工作区已有 `.runtime/flow_dp3/pickcube-smoke.h5` 和 `.runtime/flow_dp3/pickcube-20.h5`，可将后续训练的 `--data` 换为这两个文件并跳过重新采集。它们没有上传 GitHub，新实例不保证存在。

### 2.3 回放已有原始轨迹

`--source` 与 `--generate` 二选一。例如重新转换前面生成的原始轨迹：

```bash
python -B examples/baselines/flow_dp3/prepare_demos.py \
    --source "$FLOW_RUN_ROOT/demos-smoke.raw.h5" \
    --count 5 --max-steps 200 \
    --output "$FLOW_RUN_ROOT/demos-from-raw.h5"
```

原始 `.h5` 旁必须有同名 `.json`。目前只接受 PickCube / Panda 的 `pd_joint_pos` 或 `pd_ee_delta_pos` 轨迹。已经转换的 `demos-smoke.h5` 直接传给训练，不能当作原始 `--source` 再转换。

### 2.4 数据参数

| 参数 | 默认值 / 必填 | 含义 |
| --- | --- | --- |
| `--generate` | 与 `--source` 必须二选一 | 要生成的成功原始轨迹数；应大于等于 `--count`，可增加余量应对回放失败 |
| `--source` | 与 `--generate` 必须二选一 | 已有原始 `.h5` 路径，同目录须有同名 `.json` |
| `--output` | 必填 | 转换后的训练 `.h5` 路径；同名清单、原始生成文件也放在旁边 |
| `--count` | `5` | 要保留的最终成功回放数；`0` 表示处理全部原始轨迹 |
| `--start-seed` | `0` | 生成原始轨迹的起始种子，尝试时逐次递增；`--source` 模式使用原始元信息中的种子 |
| `--max-attempts` | `100` | 原始轨迹生成的最多尝试次数；不控制回放过滤数量 |
| `--max-steps` | `200` | 环境单局步数上限，同时过滤超过此长度的转换回放；至少16 |
| `--num-points` | `512` | 每帧 FPS 后的点数；当前预采样设置下范围为128～4096 |
| `--length-scale` | `1.0` | 固定长度尺度 `L`；相对 xyz 与距离通道统一除以它，必须与策略配置一致 |
| `--crop-min` | `0.05 -0.8 -0.1` | 基座坐标系下裁剪框的 xyz 下界，单位米 |
| `--crop-max` | `1.2 0.8 1.0` | 基座坐标系下裁剪框的 xyz 上界，单位米 |

点不足时先检查相机和裁剪范围。改变点数、裁剪或尺度时生成新数据，并用相应配置开始新训练。评估会从 checkpoint 读取采集契约。

成功数量不足时程序会报错，并保留已有结果；查看 `.json` 中的 `saved` / `rejected`。已有输出拒绝覆盖，换文件名重做。训练集至少需要两条成功 episode；默认按整条 episode 留出20%做验证。

训练数据每条轨迹含 `T+1` 帧点云和28维状态，以及 `T` 个4维动作。详细坐标和字段见 README 第6节。

## 3. 训练与恢复

### 3.1 小模型：检查完整训练流程

使用第2.1节的5条示范。以下命令启用 W&B，先完成安装与登录：

```bash
wandb login

python -B examples/baselines/flow_dp3/train.py \
    --config examples/baselines/flow_dp3/configs/pickcube_smoke.yaml \
    --data "$FLOW_RUN_ROOT/demos-smoke.h5" \
    --output "$FLOW_RUN_ROOT/smoke" --device cuda:0 \
    --wandb-mode online --wandb-project flow_dp3 \
    --wandb-name pickcube-001-smoke --wandb-log-every 1
```

小模型配置使用 U-Net `[64,128,256]`、batch size 2、20次更新，关系点云编码器保持一致。这一步检查数据读取、网络前向/反向、优化器、验证和 checkpoint 保存。

不使用 W&B 时将 `--wandb-mode online` 改为 `disabled`，即可跳过 SDK 和登录；网络受限时改为 `offline`。需要 CPU 检查时改为 `--device cpu`，使用新的输出目录。

### 3.2 完整模型：先训练1000次更新

准备好第2.2节的20条示范后运行：

```bash
python -B examples/baselines/flow_dp3/train.py \
    --config examples/baselines/flow_dp3/configs/pickcube.yaml \
    --data "$FLOW_RUN_ROOT/demos-20.h5" \
    --output "$FLOW_RUN_ROOT/full" --device cuda:0 \
    --batch-size 4 --steps 1000 \
    --wandb-mode online --wandb-project flow_dp3 \
    --wandb-name pickcube-001-full --wandb-log-every 10
```

完整配置为 U-Net `[512,1024,2048]`、约2.55亿参数，默认 batch size 32；这里显式使用4，先验证 PPU 上的完整模型训练。显存不足时可以减小 batch size，开始新实验。1000步是第一轮预算，不代表收敛。

### 3.3 从 checkpoint 继续到30000次更新

```bash
python -B examples/baselines/flow_dp3/train.py \
    --config examples/baselines/flow_dp3/configs/pickcube.yaml \
    --data "$FLOW_RUN_ROOT/demos-20.h5" \
    --output "$FLOW_RUN_ROOT/full" --device cuda:0 \
    --batch-size 4 --steps 30000 \
    --resume "$FLOW_RUN_ROOT/full/last.pt" \
    --wandb-mode online --wandb-log-every 10
```

`--steps 30000` 表示累计训练到第30000次更新，从第1000步恢复时再执行29000次。它不表示额外30000次，也不表示 epoch 数。

学习率调度总长度来自 YAML 的 `training.steps`。完整配置为30000步，先用 CLI 停在1000步不会改变预定调度。小模型配置总长度为20步，学习率在20步时到达调度终点，不适合直接继续很多步；需要更长训练时先创建新的实验配置，再从头训练。

恢复要求配置内容、数据 SHA256、输入契约、batch size、计算设备类型保持一致，输出写回原 checkpoint 所在目录。恢复时保存的模型、EMA、normalizer、优化器、调度器及随机状态都会加载；仅评估允许将权重加载到另一个设备。

### 3.4 训练参数与输出

| 参数 | 默认值 / 必填 | 含义 |
| --- | --- | --- |
| `--config` | `examples/baselines/flow_dp3/configs/pickcube.yaml` | 完整配置文件；小模型必须显式指定 `pickcube_smoke.yaml` |
| `--data` | 必填 | 转换后的训练 `.h5`；不是原始 `.raw.h5` |
| `--output` | 必填 | 训练输出目录；新训练必须为空，恢复必须是原目录 |
| `--device` | `cuda:0` | 模型计算设备，可选 `cpu` / `cuda:0`；不会切换物理或渲染后端 |
| `--steps` | 使用 YAML `training.steps` | 目标累计优化器更新次数，可用于提前停止，不改学习率调度长度 |
| `--batch-size` | 使用 YAML `training.batch_size` | 每次随机采样的序列窗口数；恢复必须与原训练相同 |
| `--resume` | 未设置 | 恢复 checkpoint 路径；不设置时从头训练 |
| `--wandb-log-every` | `10` | W&B 每隔多少次更新记录指标；首步、验证和末步始终记录 |

其余 W&B 参数见第6节。无论是否启用 W&B，训练目录均保存：

| 文件 | 查看内容 |
| --- | --- |
| `config.yaml` | 本次使用的原始 YAML；实际 CLI batch size 等在 `run.json` 查看 |
| `run.json` | 实际设备、batch size、参数量、数据指纹和训练 / 验证 episode |
| `metrics.jsonl` | 每次更新的 `train_loss`、`grad_norm`、`lr`、`ema_decay`；验证时增加 `val_loss` |
| `last.pt` | 最近保存的可恢复 checkpoint；最后一步一定保存 |
| `best.pt` | 验证损失改善时保存的 checkpoint；不代表任务成功率最高 |
| `wandb_run.json` / `wandb/` | 启用 SDK 后保存的 run 信息和本地日志 |

```bash
tail -n 5 "$FLOW_RUN_ROOT/smoke/metrics.jsonl"
cat "$FLOW_RUN_ROOT/smoke/run.json"
```

终端在首步、每10步和末步打印训练记录。训练的验证 loss 来自离线示范数据；闭环任务成功率需要单独运行评估。

## 4. 评估、推理与 MP4

### 4.1 评估小模型并录制两段视频

```bash
python -B examples/baselines/flow_dp3/evaluate.py \
    --checkpoint "$FLOW_RUN_ROOT/smoke/last.pt" --device cuda:0 \
    --episodes 2 --start-seed 1000 --policy-seed 42 --max-steps 200 \
    --save-video --video-dir "$FLOW_RUN_ROOT/videos-smoke" \
    --output "$FLOW_RUN_ROOT/eval-smoke.json"

ls -lh "$FLOW_RUN_ROOT/videos-smoke"/*.mp4
cat "$FLOW_RUN_ROOT/eval-smoke.json"
```

这是闭环推理：每局 reset 后使用观测历史预测动作块，执行并更新观测，然后再次预测。默认加载 EMA 权重，策略在 PPU 计算，物理和点云渲染仍在 CPU。

输出 `seed_1000.mp4` 与 `seed_1001.mp4`，JSON 的每局结果包含 `video_path`。默认20 FPS，200步执行约10秒；一局录像通常包含初始帧，因此完整录像为201帧。展示相机与观测相机分开，录制展示画面不会改变点云输入契约。录像会增加渲染和编码耗时。

### 4.2 评估完整模型

训练后单独运行，例如使用按验证 loss 选择的 checkpoint：

```bash
python -B examples/baselines/flow_dp3/evaluate.py \
    --checkpoint "$FLOW_RUN_ROOT/full/best.pt" --device cuda:0 \
    --episodes 20 --start-seed 1000 --policy-seed 42 --max-steps 200 \
    --output "$FLOW_RUN_ROOT/eval-full.json" \
    --wandb-mode online --wandb-project flow_dp3 \
    --wandb-name pickcube-001-full-eval
```

此命令只记录评估指标，没有开启录像。需要视频时加 `--save-video`；需要上传视频到 W&B 时，再加 `--wandb-upload-videos`。每次新的评估使用新的 JSON 文件名或实验编号，同名报告和视频拒绝覆盖。

### 4.3 评估参数

| 参数 | 默认值 / 必填 | 含义 |
| --- | --- | --- |
| `--checkpoint` | 必填 | 训练生成的 `last.pt` / `best.pt` |
| `--device` | `cuda:0` | 策略计算设备，可选 `cpu` / `cuda:0`；checkpoint 可跨设备评估 |
| `--episodes` | `10` | 独立评估局数 |
| `--start-seed` | `1000` | 环境起始种子，依次评估到 `start_seed + episodes - 1`；应与训练数据种子分离 |
| `--policy-seed` | `42` | 策略采样噪声的随机种子，与环境种子分开 |
| `--max-steps` | `200` | 每局环境动作步数上限；瞬时成功不会提前终止评估 |
| `--raw-weights` | 关闭 | 开启后使用训练模型原始权重；默认使用 EMA |
| `--output` | 必填 | JSON 评估报告路径，不能覆盖现有报告 |
| `--save-video` | 关闭 | 开启后每局生成 MP4；不依赖 W&B |
| `--video-dir` | 报告旁的 `<报告名>-videos/` | MP4 保存目录，仅在 `--save-video` 时使用 |
| `--video-fps` | 环境控制频率，本任务20 | MP4 播放帧率；改动会影响播放速度，不改变动作控制频率 |
| `--wandb-upload-videos` | 关闭 | 将录制视频记录到 W&B；要求同时启用 `--save-video` 和 online / offline 日志 |

相机、裁剪、采样点数和长度尺度从 checkpoint 的契约恢复，评估 CLI 不提供重新设置这些参数的入口。

### 4.4 结果在哪里看

| 位置 / 指标 | 含义 |
| --- | --- |
| 终端 | 每局种子、成功状态、耗时和末尾总体成功率 |
| JSON `success_once_rate` | 一局内至少成功过一次的比例 |
| JSON `success_end_rate` | 最后一帧仍成功的局数比例 |
| JSON `episodes` | 每局步数、推理次数、平均推理时间、预处理、仿真 / 渲染时间和动作裁剪比例 |
| 本地 MP4 | 机器人实际闭环执行画面；在阿里云 DSW 文件浏览器中打开或下载 |
| W&B 评估 run | 每局 `eval/*` 指标与 summary 成功率；显式上传后才有视频 |

录制目录在隐藏的 `.runtime/` 下，DSW 文件浏览器需要允许显示隐藏文件。现有预览位于 `.runtime/flow_dp3/videos/ppu-smoke-preview/seed_1000.mp4` 和 `seed_1001.mp4`，它们来自20步小模型，结果为失败演示。

当前200步是本适配的评估预算。比较不同模型时应使用相同任务、控制模式、种子集合、policy seed 和步数上限。

## 5. YAML 配置参数

命令行负责选择数据、设备、运行目录和训练预算；模型结构与训练设置在 YAML 中。`pickcube.yaml` 是完整模型，`pickcube_smoke.yaml` 是流程检查模型。新实验修改配置时复制成新文件，不改正在恢复训练的配置。

### 5.1 观测、网络与采样：`policy`

以下默认值以完整配置为准。

| 字段 | 默认值 | 意义 |
| --- | --- | --- |
| `horizon` | `16` | 每次预测的动作序列长度；须符合 U-Net 下采样整除条件 |
| `n_obs_steps` | `2` | 输入的历史观测帧数 |
| `n_action_steps` | `8` | 每次预测后连续执行的动作数；执行从预测索引 `n_obs_steps-1` 开始 |
| `radius1_m` / `radius2_m` | `0.10` / `0.20` | 两级点云邻域的物理半径，单位米；内部除以 `length_scale` |
| `length_scale` | `1.0` | 点云统一长度尺度，须与数据采集一致 |
| `down_dims` | `[512,1024,2048]` | U-Net 各级通道宽度；小模型为 `[64,128,256]` |
| `diffusion_step_embed_dim` | `128` | 网络时间条件的编码维度 |
| `kernel_size` | `5` | 时序卷积核大小，要求为正奇数 |
| `n_groups` | `8` | GroupNorm 分组数，通道宽度须可被它整除 |
| `num_inference_steps` | `10` | 每次动作块预测的速度场积分更新次数；不是环境步数 |
| `solver` | `consistency` | 推理采样公式，支持 `consistency` / `euler` |
| `fm_eps` | `0.01` | 训练和采样的起始时间下界，避开0 |
| `fm_time_scale` | `100.0` | 输入网络的时间条件缩放，不是仿真时间或控制频率 |
| `num_segments` | `2` | 训练一致性目标中的时间分段数 |
| `boundary` | `1` | 构造一致性目标时切换边界目标的时间阈值 |
| `delta` | `0.01` | 一致性损失中两次网络预测的时间间隔，上界截到1 |
| `alpha` | `0.00001` | 速度一致性项的损失权重 |
| `noise_scale` | `1.0` | 训练起始噪声和推理初始动作噪声的尺度 |
| `sigma_var` | `0.0` | consistency 采样公式的额外随机扰动系数；默认不加入额外扰动 |

当前状态维度固定28、动作维度固定4，由 `PolicyConfig` 默认值确定。编码器固定使用纯 PyTorch 实现，输出64维关系点云特征。

### 5.2 优化与数据划分：`training`

| 字段 | 完整配置值 | 意义 |
| --- | --- | --- |
| `seed` | `42` | 训练随机流和 episode 训练 / 验证划分种子 |
| `batch_size` | `32` | 每次更新的序列窗口数；CLI `--batch-size` 可覆盖，小模型为2 |
| `steps` | `30000` | 默认累计更新数及学习率调度总长度；小模型为20 |
| `lr` | `0.0001` | AdamW 基础学习率，实际值受 warmup / cosine 调度影响 |
| `betas` | `[0.95,0.999]` | AdamW 一阶 / 二阶矩衰减系数 |
| `weight_decay` | `0.000001` | AdamW 权重衰减 |
| `warmup_steps` | `500` | 学习率预热步数，随后使用 cosine；小模型为0 |
| `val_ratio` | `0.2` | 按整条 episode 留出的验证比例；至少留一条，统计归一化只用训练集 |
| `val_every` | `100` | 每隔多少次更新计算验证 loss；小模型为10，最后一步也验证 |
| `val_batches` | `4` | 每次验证随机采样多少个 batch 后取平均；小模型为2 |
| `checkpoint_every` | `100` | 定期保存 checkpoint 的更新间隔；小模型为10，验证改善和末步也保存 |
| `grad_clip` | `1.0` | 梯度范数裁剪阈值；日志中的 `grad_norm` 为裁剪前范数 |

## 6. W&B 使用与公共参数

训练和评估默认 `--wandb-mode disabled`；历史日志导入默认 `online`。启用在线模式时，终端会打印页面链接。

| 参数 | 默认值 | 意义 |
| --- | --- | --- |
| `--wandb-mode` | 训练 / 评估 `disabled` | `disabled` 只保存普通本地结果；`online` 实时上传；`offline` 保存 SDK 日志等待同步 |
| `--wandb-project` | `flow_dp3` | W&B 项目；恢复训练未指定时沿用原项目 |
| `--wandb-entity` | 登录账户的默认 workspace | 个人或团队 workspace；已有恢复信息时沿用 |
| `--wandb-name` | 当前输出目录名 | 页面中的显示名称，不作为恢复身份 |
| `--wandb-run-id` | 自动生成，恢复时沿用 | run 的唯一标识；恢复原训练必须与已保存 ID 一致 |

在线恢复训练自动续接同一 run；评估默认创建独立 run，并在 checkpoint 有训练日志信息时关联其训练 ID。W&B 不自动上传示范或 `.pt` 权重，视频上传需要显式开启。PPU 环境默认关闭 SDK 自动硬件监控。

### 6.1 导入已有训练曲线

例如将已有 PPU 小模型的20条记录导入，无需重新训练：

```bash
python -B examples/baselines/flow_dp3/import_training_logs.py \
    --run-dir .runtime/flow_dp3/ppu-smoke \
    --output "$FLOW_RUN_ROOT/history" \
    --wandb-mode online --wandb-project flow_dp3 \
    --wandb-name ppu-smoke-history
```

| 参数 | 含义 |
| --- | --- |
| `--run-dir` | 必填，已有训练目录，必须包含 `metrics.jsonl`、`config.yaml`、`run.json` |
| `--output` | SDK 导入输出目录，默认原训练目录的 `wandb-history/`；目录必须为空 |

网页训练曲线包括 `train/loss`、`val/loss`、`train/lr`、`train/grad_norm`、`train/ema_decay`，横轴为实际 `global_step`。历史导入创建独立日志 run；重复导入换新的输出目录。

### 6.2 离线记录与同步

训练命令中的 `--wandb-mode online` 改为 `offline`，即可在没有登录或外网时记录；仍须安装 SDK。离线期间网页不会实时更新。联网后执行：

```bash
find "$FLOW_RUN_ROOT/full/wandb" \
    -maxdepth 1 -type d -name 'offline-run-*' -print

wandb login
# 将“实际目录名”换成上面输出的真实目录名。
wandb sync "$FLOW_RUN_ROOT/full/wandb/offline-run-实际目录名"
```

小模型离线目录在 `smoke/wandb/`；历史导入在 `history/wandb/`。离线恢复保持逻辑 run ID，但每次执行产生新的离线目录，需要逐个同步。在线认证和网络故障按终端实际报错处理。

## 7. 命令速查与文件保存

```bash
python -B examples/baselines/flow_dp3/prepare_demos.py --help
python -B examples/baselines/flow_dp3/train.py --help
python -B examples/baselines/flow_dp3/evaluate.py --help
python -B examples/baselines/flow_dp3/import_training_logs.py --help
```

`python -B` 禁止生成 `.pyc`，不改变训练算法。数据、权重、日志、视频都保存在 `.runtime/` 并被 Git 忽略；需要备份或换实例时单独传输。代码部署只需要本仓库和对应运行环境。
