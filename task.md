# ManiSkill Diffusion Policy 任务与实验配置：Flow Matching 参考方案

整理日期：2026-10-05。本地源码版本：`87ea3bffc5548a670bf8e803e9411fd6cc773d01`。

本文依据本地 `examples/baselines/diffusion_policy/`、示范生成脚本及官方仓库核对，整理 DP 使用的任务、录制、回放、训练和推理设置，并给出当前 FlowDP3 的推荐起点。官方链接指向可变的 `main` 分支；复现实验时还应记录使用的代码版本和依赖版本。

**“DP 清单值”来自官方实验命令，“代码默认值”来自未被命令覆盖的参数，“FM 推荐值”是待验证的实验建议。推荐不等于官方已调优结果。本文只新增文档，没有修改程序、YAML、数据或权重，也没有执行下面的实验命令。**

## 1. 先确定参考哪些任务

官方 [DP 实验清单](https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/diffusion_policy/baselines.sh) 对应本地 [baselines.sh](examples/baselines/diffusion_policy/baselines.sh)，共 **6 个不同任务、9 组实验**：5 组 state、4 组 RGB。清单没有列出 pointcloud DP，也没有列出完整的 RGB+Depth 实验。

| 任务 | 任务内容 | 主要机器人 | state DP | RGB DP | 示范来源 | FM 接入顺序 |
| --- | --- | --- | --- | --- | --- | --- |
| `PickCube-v1` | 抓取方块并移动到目标位置 | Panda | 有 | 有 | motionplanning | 第 1 个；当前已接入 |
| `PushCube-v1` | 将方块推入目标区域 | Panda | 有 | 有 | motionplanning | 第 2 个；需要任务适配 |
| `StackCube-v1` | 将一个方块堆到另一个方块上 | Panda | 有 | 有 | motionplanning | 第 3 个；需要任务适配 |
| `PegInsertionSide-v1` | 抓取插销并插入侧面孔洞 | Panda | 有 | 未列出 | motionplanning | 后续；需要旋转控制 |
| `PushT-v1` | 将 T 形物体推到目标位姿 | Panda Stick | 有 | 未列出 | RL/PPO | 后续；机器人、动作和仿真后端都不同 |
| `DrawTriangle-v1` | 在桌面画出三角形 | Panda Stick | 未列出 | 有 | motionplanning | 后续；需要绘画观测和目标适配 |

“未列出”只表示这份清单没有对应实验，不代表环境无法提供该观测。RGB 入口虽然名为 `train_rgbd.py`，但清单明确指定 `--obs-mode rgb`。视觉 DP 还输入本体和任务信息，并非只输入图像；具体字段见 [观测提取函数](examples/baselines/diffusion_policy/diffusion_policy/utils.py)。

当前 FlowDP3 的 [环境适配器](examples/baselines/flow_dp3/obs_adapter.py)、[数据转换器](examples/baselines/flow_dp3/prepare_demos.py) 和 [策略维度检查](examples/baselines/flow_dp3/policy.py) 限定 `PickCube-v1 / panda / 28 维状态 / 4 维动作`。其他任务目前不能只改环境名称就运行。

## 2. 参数中的 step 分别是什么意思

| 名称 | DP 参数 | FlowDP3 对应参数 | 含义 |
| --- | --- | --- | --- |
| 训练更新次数 | `total_iters` | `training.steps` / `--steps` | 一个 batch 前向、反向并更新参数一次 |
| 观测历史长度 | `obs_horizon` | `n_obs_steps` | 一次策略查询使用多少帧观测 |
| 动作预测长度 | `pred_horizon` | `horizon` | 一次生成多少个动作 |
| 动作执行长度 | `act_horizon` | `n_action_steps` | 执行多少个动作后重新查询模型 |
| 推理内部迭代次数 | `num_diffusion_iters` | `num_inference_steps` | 生成一段动作时，去噪或积分多少次 |
| 每局环境上限 | `max_episode_steps` | 评估 `--max-steps` | 一局最多执行多少个环境动作 |
| 示范数量 | `num_demos` | 数据准备 `--count`，再按 episode 划分 | 示范轨迹数量，不是帧数或 batch 数 |

例如“训练 30,000 次更新、每次推理积分 10 步、每次执行 8 个动作、每局评估 100 步”描述的是四个独立数量。这里的离线更新次数也不能与 PPO 的 `total_timesteps` 环境交互次数直接比较。

## 3. DP 的原始示范录制设置

流程是：**下载或生成原始轨迹 → 回放生成所需观测并转换控制模式 → 加载指定数量的示范 → 离线训练 → 闭环评估。**

原始轨迹和 DP 训练数据是不同阶段的文件。官方原始下载通常省略完整观测，以减小体积；不能把原始 `trajectory.h5` 当作已经带有 RGB 或点云的训练文件。[官方数据准备说明](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html)

### 3.1 运动规划示范

来源：[官方生成清单](https://github.com/mani-skill/ManiSkill/blob/main/scripts/data_generation/motionplanning.sh)、本地 [motionplanning.sh](scripts/data_generation/motionplanning.sh) 和 [Panda 录制入口](mani_skill/examples/motionplanning/panda/run.py)。下表只整理 DP 用到的运动规划任务。

| 参数 | PickCube / PushCube / StackCube / PegInsertionSide | DrawTriangle | 说明 |
| --- | --- | --- | --- |
| 录制入口 | `mani_skill.examples.motionplanning.panda.run` | 同左 | 使用内置任务规划器 |
| 正式原始示范数量 | `-n 1000` | `-n 1000` | 后续 DP 只取 100 条训练，不是全部 1000 条 |
| 成功筛选 | `--only-count-success` | 同左 | 只保存成功轨迹，持续尝试直到满足数量 |
| 原始控制模式 | `pd_joint_pos` | `pd_joint_pos` | 在录制入口中设置；不是 DP 的最终动作空间 |
| 原始观测 | `--obs-mode none`，代码默认 | 同左 | 主要保存动作、环境状态、元数据；观测后续回放生成 |
| 仿真后端参数 | `auto`，代码默认 | 同左 | 单环境默认路径为 CPU；实际后端以轨迹 JSON 为准 |
| 并行进程 | `--num-procs 1`，代码默认 | 清单显式 `--num-procs 10` | CPU 多进程，不是训练 batch size |
| 轨迹名称 | `--traj-name trajectory` | 同左 | 输出 `.h5` 和同名 `.json` |
| 默认输出位置 | `demos/<env_id>/motionplanning/` | 同左 | `--record-dir` 默认 `demos` |
| 正式批量录制视频 | 未开启 | 未开启 | 清单先单独生成一条示例视频，再批量生成数据 |
| 示例视频 | `-n 1 --save-video --shader rt` | 同左 | 仅用于展示；ray tracing 不是训练数据的必要设置 |
| 视频帧率 | 录制器设置 30 FPS | 同左 | 视频输出参数，不等于策略查询频率 |
| 录制时长 | 入口使用任务默认设置，求解器实际轨迹可更长 | 同左 | 不能把后面的 IL 评估上限当作这里的录制参数 |

本地生成数据和使用官方发布数据，应在实验名称与报告中分别标注。生成 1000 条原始成功轨迹，也不保证转换控制模式后仍有 1000 条成功轨迹。

### 3.2 PushT 的 RL 示范

DP 的 PushT 实验使用 RL 示范，不是运动规划示范。来源：[官方 RL 生成脚本](https://github.com/mani-skill/ManiSkill/blob/main/scripts/data_generation/rl.sh)、本地 [rl.sh](scripts/data_generation/rl.sh)。

| 阶段 | 参数 | PushT 设置 |
| --- | --- | --- |
| 专家训练 | 算法入口 | PPO `ppo_fast.py` |
| 专家训练 | 控制模式 | 生成清单有多种模式；DP 选择 `pd_ee_delta_pose` |
| 专家训练 | `total_timesteps` | 25,000,000 次累计环境交互 |
| 专家训练 | `num_envs` / `num_steps` | 4096 / 16 |
| 专家训练 | `update_epochs` / `num_minibatches` | 8 / 32 |
| 专家训练 | `gamma` | 0.99 |
| 专家录制 | 入口参数 | `--evaluate --save-trajectory --no-capture-video` |
| 专家录制 | `num_eval_envs` / `num_eval_steps` | 1024 / 100 |
| 成功筛选 | 后处理 | 删除从未成功的轨迹，保留到最后一次成功对应的时刻 |
| 后处理入口 | 文件 | `scripts/data_generation/process_rl_trajectories.py` |
| DP 训练 | 最终使用示范数 | 100 |

1024 是专家录制的并行环境数，不是筛选后保证留下的成功示范数。[后处理源码](scripts/data_generation/process_rl_trajectories.py) 默认 `dry_run=True`；要实际生成筛选后的文件需要显式关闭 dry run。

PushT 使用 `panda_stick`，没有 Panda 夹爪。其末端位姿增量动作是 6 维；不能套用当前 FlowDP3 的“xyz + 夹爪”4 维动作。[Panda Stick 控制器](mani_skill/agents/robots/panda/panda_stick.py)

## 4. DP 的回放与训练数据转换设置

来源：[官方回放清单](https://github.com/mani-skill/ManiSkill/blob/main/scripts/data_generation/replay_for_il_baselines.sh)、本地 [replay_for_il_baselines.sh](scripts/data_generation/replay_for_il_baselines.sh)。

| DP 使用的任务 | 回放观测 `-o` | 目标控制器 `-c` | 仿真后端 `-b` | 并行参数 `--num-envs` | 状态恢复方式 |
| --- | --- | --- | --- | --- | --- |
| PickCube | `state` 或 `rgb` | `pd_ee_delta_pos` | `physx_cpu` | 10 | `--use-first-env-state` |
| PushCube | `state` 或 `rgb` | `pd_ee_delta_pos` | `physx_cpu` | 10 | `--use-first-env-state` |
| StackCube | `state` 或 `rgb` | `pd_ee_delta_pos` | `physx_cpu` | 10 | `--use-first-env-state` |
| PegInsertionSide | `state` | `pd_ee_delta_pose` | `physx_cpu` | 10 | `--use-first-env-state` |
| PushT | `state` | `pd_ee_delta_pose` | `physx_cuda` | 1024 | `--use-env-states` |
| DrawTriangle | `rgb` | `pd_ee_delta_pos` | `physx_cpu` | DP 对应回放命令未在该脚本中列出 | 清单存在缺项，需要单独确认 |

所有列出的回放命令使用 `--save-traj`。CPU 回放的 `--num-envs` 在实现中用于多进程回放，不能理解为同一个 CPU 场景并行模拟十个环境；GPU 路径才在一个场景内批量模拟。[回放入口](mani_skill/trajectory/replay_trajectory.py)

| 文件类型 | 典型路径 | 用途 |
| --- | --- | --- |
| 原始运动规划轨迹 | `<env_id>/motionplanning/trajectory.h5` 与 `.json` | 保存原始动作和环境状态 |
| DP state 数据 | `<env_id>/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5` | 已转换为 state 和末端增量控制 |
| DP RGB 数据 | `<env_id>/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5` | 已转换为 RGB 和末端增量控制 |
| PegInsertionSide 数据 | `PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5` | 使用含旋转的控制模式 |
| PushT 原始 RL 数据 | `PushT-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5` | 筛选后的专家动作和环境状态 |
| PushT DP 数据 | `PushT-v1/rl/trajectory.state.pd_ee_delta_pose.physx_cuda.h5` | 回放生成 state 观测 |

`--use-first-env-state` 恢复初始状态后执行动作；`--use-env-states` 是按保存的状态恢复回放画面。后者可用于复现数据，但不属于模型闭环评估。当前 FM 的动作转换应继续使用实际控制回放和成功检查。

原始脚本部分 `${DEMO_PATH}<env_id>` 路径缺少 `/`，DrawTriangle 又存在回放清单缺项。本文保留这些差异，不把推测补充的命令标成官方已验证设置。

## 5. DP 的九组正式训练配置

以下是实验清单的实际覆盖值。`total_iters` 均为参数更新次数，`max_episode_steps` 是训练过程中评估环境的每局上限。

| 任务 | 观测 | 示范数 | 更新次数 | batch size | 控制模式 | 评估仿真后端 | 每局上限 | 评估并行数 | 执行动作数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PickCube | state | 100 | 30,000 | 1024 | `pd_ee_delta_pos` | `physx_cpu` | 100 | 10 | 8 |
| PushCube | state | 100 | 30,000 | 1024 | `pd_ee_delta_pos` | `physx_cpu` | 100 | 10 | 8 |
| StackCube | state | 100 | 30,000 | 1024 | `pd_ee_delta_pos` | `physx_cpu` | 200 | 10 | 8 |
| PegInsertionSide | state | 100 | 100,000 | 1024 | `pd_ee_delta_pose` | `physx_cpu` | 300 | 10 | 8 |
| PushT | state | 100 | 50,000 | 1024 | `pd_ee_delta_pose` | `physx_cuda` | 150 | 100 | **1** |
| PickCube | RGB | 100 | 30,000 | 256 | `pd_ee_delta_pos` | `physx_cpu` | 100 | 10 | 8 |
| PushCube | RGB | 100 | 30,000 | 256 | `pd_ee_delta_pos` | `physx_cpu` | 100 | 10 | 8 |
| StackCube | RGB | 100 | 100,000 | 256 | `pd_ee_delta_pos` | `physx_cpu` | 200 | 10 | 8 |
| DrawTriangle | RGB | 100 | 100,000 | **128** | `pd_ee_delta_pos` | `physx_cpu` | 300 | 10 | 8 |

未显式覆盖的 batch、评估并行数和动作长度来自对应训练入口默认值；只有 PushT 和 DrawTriangle 对相关项做了表中加粗的特殊覆盖。来源：[本地实验清单](examples/baselines/diffusion_policy/baselines.sh)、[state 入口](examples/baselines/diffusion_policy/train.py)、[RGB 入口](examples/baselines/diffusion_policy/train_rgbd.py)。

### 5.1 通用训练超参数

| 参数 | DP state | DP RGB | 参数来源与含义 |
| --- | --- | --- | --- |
| `total_iters` 代码默认 | 1,000,000 | 1,000,000 | 正式实验被上表覆盖，不应直接照用默认值 |
| `seed` | 1 | 1 | 清单变量为 1，代码默认也为 1；清单没有多种子循环 |
| `lr` | `1e-4` | `1e-4` | 学习率 |
| 优化器 | AdamW | AdamW | 训练实现 |
| `betas` | `(0.95, 0.999)` | 同左 | AdamW 参数 |
| `weight_decay` | `1e-6` | 同左 | 权重衰减 |
| 学习率调度 | cosine | cosine | 总更新数由实际 `total_iters` 决定 |
| warmup 更新数 | 500 | 500 | 不是 500 个 epoch |
| EMA | `power=0.75` | 同左 | 用于评估和 checkpoint 中的 EMA 权重 |
| `obs_horizon` | 2 | 2 | 两帧观测 |
| `pred_horizon` | 16 | 16 | 预测 16 个动作 |
| `act_horizon` | 8，PushT 为 1 | 8 | 执行一段动作后重新推理 |
| `unet_dims` | `[64,128,256]` | 同左 | 官方 DP 默认主干宽度 |
| `diffusion_step_embed_dim` | 64 | 64 | 时间嵌入维数 |
| `kernel_size` / `n_groups` | 5 / 8 | 同左 | U-Net 卷积核及 GroupNorm |
| `log_freq` | 1000 | 1000 | 每多少次更新记录训练指标 |
| `eval_freq` | 5000 | 5000 | 定期调用闭环评估；实现还会在起始索引及训练结束处评估 |
| `num_eval_episodes` | 100 | 100 | 每次评估总局数 |
| `save_freq` | `None` | `None` | 默认没有固定周期 checkpoint；根据最佳成功率保存 |
| 梯度裁剪 | 入口未显式调用 | 同左 | 不应把 FM 的 `grad_clip=1.0` 当成 DP 配置 |
| 验证集划分 | 无独立离线验证集划分 | 同左 | `num_demos=100` 是加载 100 条用于训练 |
| `num_dataload_workers` | 0 | 0 | 默认数据加载进程数 |
| 视频 | 默认开启；PushT 清单关闭 | 默认开启 | CPU 评估只录第一个并行环境 |
| W&B | 清单传 `--track` | 同左 | 代码默认关闭；实验清单开启 |

来源：[state DP 源码](https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/diffusion_policy/train.py)、[RGB DP 源码](https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/diffusion_policy/train_rgbd.py)、本地 [U-Net](examples/baselines/diffusion_policy/diffusion_policy/conditional_unet1d.py)。

### 5.2 数据窗口与模型输入

| 项目 | DP 设置 | FM 当前行为 / 需要核对的差异 |
| --- | --- | --- |
| 轨迹长度关系 | `T+1` 帧观测，`T` 个动作 | 同样要求 |
| 历史填充 | 轨迹开头重复第一帧 | 当前同样使用边缘重复 |
| 窗口尾部范围 | DP 使用 `pred_horizon-obs_horizon` | FM 使用 `n_action_steps-1`；窗口覆盖不同 |
| 增量位置动作尾部 | 机械臂补零，保持最后夹爪状态 | FM 重复最后动作；非零末端增量可能影响任务完成后的稳定性，需实测 |
| state DP 输入 | 环境完整 state | 包含物体等真值信息的口径与视觉策略不同 |
| RGB DP 输入 | 图像 + 本体 / 任务附加字段 | FM 使用距离点云 + 28 维状态，模态不同 |
| 图像分辨率 | README 示例假定 128×128 | FM 使用同一相机生成点云，再采样 512 点 |
| state / action 归一化 | DP 入口没有 FM 的训练集 limits 统计 | FM 当前对 state / action 使用训练集 limits；点云保持物理尺度 |

来源：[DP Dataset](examples/baselines/diffusion_policy/train.py)、[DP README](examples/baselines/diffusion_policy/README.md)、[FM Dataset](examples/baselines/flow_dp3/dataset.py)。窗口和归一化差异应写入对比报告；本文不据此修改你的算法。

## 6. DP 的推理与评估配置

| 参数 / 行为 | DP 官方实现 | 对 FM 的参考方式 |
| --- | --- | --- |
| 推理权重 | EMA | 保留 EMA |
| 生成过程 | DDPM 去噪 | 使用当前 FM 速度场 / 一致性采样器 |
| `num_train_timesteps` | 100 个离散噪声时刻 | FM 使用连续时间采样，无需复制该字段 |
| 推理迭代次数 | 默认 100 次 DDPM 去噪 | FM 先保持 10 次积分，再测 5 / 10 / 20 |
| `beta_schedule` | `squaredcos_cap_v2` | DDPM 专属，不直接迁移 |
| `prediction_type` | `epsilon`，预测噪声 | 不用于替代 FM 的速度 / 一致性目标 |
| `clip_sample` | `True` | FM 当前在执行动作前裁剪到 `[-1,1]`；位置和语义不同 |
| 初始动作序列 | 标准高斯噪声 | 保留当前 `noise_scale=1.0` |
| 使用历史 | 两帧 | 保留 `n_obs_steps=2` |
| 动作预测 / 执行 | 16 / 8，PushT 执行 1 | 普通任务先 16 / 8；接触阶段可另测执行 4 或 1 |
| 动作切片起点 | `obs_horizon-1`，默认为索引 1 | 当前 FM 同样从索引 1 取执行动作 |
| 重新查询策略 | 当前动作块执行完，使用更新后的观测历史 | 保持相同逻辑 |
| 成功后是否立即结束 | 不结束，运行到固定评估时长 | 当前 FM 已按这一原则执行 |
| 部分环境提前 reset | 关闭 | 单环境也不要在瞬时成功时立即 reset |
| `reconfiguration_freq` | 1 | 每局重配，包含几何随机化的任务尤其重要 |
| 指标 | `success_once` / `success_at_end` | FM 为 `success_once_rate` / `success_end_rate`，含义对应 |
| 最佳模型选择 | 分别按两种闭环成功率保存 | FM 当前只按离线 val loss；应另做固定场景闭环选择 |
| 仿真后端 | 与示范数据一致 | 当前 CPU 数据先坚持 CPU 评估 |

来源：[DP 推理](examples/baselines/diffusion_policy/train.py)、[DP 评估函数](examples/baselines/diffusion_policy/diffusion_policy/evaluate.py)、[评估环境包装](examples/baselines/diffusion_policy/diffusion_policy/make_env.py) 和 [官方评估规范](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html#evaluation)。

DP 的 `diffusion_policy/evaluate.py` 是训练入口调用的辅助函数，当前 DP 入口没有 PPO 那样的 `--evaluate --checkpoint` 独立命令。不能直接照抄 PPO 推理命令。你的 FM 已有独立 [evaluate.py](examples/baselines/flow_dp3/evaluate.py)，可以加载自己的 checkpoint。

## 7. 推荐给当前 Flow Matching 的参数

### 7.1 每个任务的实验起点

下面是 **FM 推荐预算，不是已经运行过的结果**。点云属于视觉输入，但与 RGB DP 的编码器和信息量不同，因此参考预算后仍需用成功率曲线验证。每个任务先保证 100 条实际参与训练的成功示范；验证数据另行留出。

| 任务 | 推荐原始数据来源 | FM 首轮更新数 | 后续预算候选 | 每局评估上限 | 预测 / 执行长度 | 推荐采样次数 | 现在能否直接运行 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PickCube | motionplanning | 30,000 | 曲线仍提升时另做 50,000 | 主结果 100；补充 200 | 16 / 8；另测执行 4 | 10 | 可以；现有数据实际为 80 条训练 + 20 条验证 |
| PushCube | motionplanning | 30,000 | 必要时 50,000 | 100 | 16 / 8 | 10 | 不可以；需要适配环境与状态字段 |
| StackCube | motionplanning | 30,000 做阶段检查 | 100,000，参考 RGB DP | 200 | 16 / 8；另测执行 4 | 10 | 不可以；需要观测和目标适配 |
| PegInsertionSide | motionplanning | 100,000 | 依据曲线决定 | 300 | 16 / 8；另测执行 4 或 1 | 10，另测 20 | 不可以；Panda 改为 7 维位姿增量 + 夹爪动作 |
| PushT | 官方 RL 示范优先 | 50,000 | 必要时 100,000 | 150 | 16 / **1** | 10 | 不可以；Panda Stick、6 维动作、GPU 示范后端待适配 |
| DrawTriangle | motionplanning | 100,000 | 依据曲线决定 | 300 | 16 / 8；另测执行 4 或 1 | 10 | 不可以；Panda Stick、3 维位置动作和绘画目标待适配 |

后续预算候选需要从新实验开始正确设置总步数 / cosine 调度；当前恢复检查要求配置一致，不支持随意改变原实验 YAML 后继续恢复。

### 7.2 录制与数据准备推荐

| 参数 | 当前 FM | 推荐起点 | 原因 / 实施条件 |
| --- | --- | --- | --- |
| 首轮数据 | 已有 100 条，80 训练 / 20 验证 | 先完成现有实验 | 避免同时改变数据和模型 |
| 正式对比数据 | 总数 100 | 125 条成功回放，按 0.2 划分得到 100 训练 / 25 验证 | 让训练数量与 DP 的 100 条一致 |
| `--generate` | 显式传入 | 新录制候选 150 | 为控制转换失败预留余量，不是保证成功率 |
| `--count` | 默认 5 | 正式新数据 125 | 默认 5 仅适合流程检查 |
| `--max-attempts` | 默认 100 | 新录制候选 400 | 原始生成的最大尝试次数 |
| 录制 / 转换 `--max-steps` | 200 | PickCube 保留 200 | 是数据准备上限；正式评估仍单独指定 100 |
| `--start-seed` | 0 | 保留 0 | 实际成功使用的 seed 写入数据清单 |
| `num_points` | 512 | 保留 512 | 先固定输入规模 |
| `pre_sample_points` | 4096 | 保留 4096 | 点云预采样规模 |
| `length_scale` | 1.0 | 保留 1.0 | 训练和推理统一物理长度尺度 |
| 裁剪下界 / 上界 | `[0.05,-0.8,-0.1]` / `[1.2,0.8,1.0]` | PickCube 保留 | 新任务需检查相机可见性与裁剪后有效点数 |
| 数据过滤 | 成功转换后保留 | 保持最终成功过滤 | 原始成功不保证转换后成功 |
| 视频 | 可独立录制 | 数据采集少量检查，评估另录样例 | 不用 ray tracing 批量采集训练数据 |

新数据会改变数据指纹，不能用于恢复旧 checkpoint 的同一次训练。若严格复现官方数据分布，应从官方下载数据回放，并让 DP 与 FM 使用相同的训练 episode；本地生成与官方发布数据应分开报告。[FM 数据准备](examples/baselines/flow_dp3/prepare_demos.py)

### 7.3 训练推荐

| 参数 | DP 参考 | 当前 FM | FM 推荐起点 |
| --- | --- | --- | --- |
| `steps` | PickCube 30,000 | 30,000 | 保持 30,000；按任务使用 7.1 表 |
| `batch_size` | state 1024，RGB 256 | 32 | 当前大模型先保持 32；新实验可测 64，先验证内存与吞吐量 |
| `lr` | `1e-4` | `1e-4` | 保持 |
| `betas` | `[0.95,0.999]` | 相同 | 保持 |
| `weight_decay` | `1e-6` | 相同 | 保持 |
| warmup / scheduler | 500 / cosine | 相同 | 保持，调度总步数与实验预算一致 |
| `grad_clip` | 无显式裁剪 | 1.0 | 保持 1.0，记录梯度是否频繁被裁剪 |
| EMA | power 0.75 | 参考衰减实现 | 保留 EMA，评估使用 EMA 权重 |
| `seed` | 清单 1 | 42 | 先固定 42；正式实验另做 43、44，作为三个独立训练种子 |
| `val_ratio` | 无独立划分 | 0.2 | 保持；正式对比确保划分后训练集有 100 条 |
| `val_every` | 无离线验证周期 | 100 | 新实验候选 500；旧实验恢复保持原配置 |
| `val_batches` | 不适用 | 4 | 新实验候选 16；更完整覆盖可另做离线评估 |
| `checkpoint_every` | 按成功率保存 | 100 | 新实验候选 1000，减少大模型 I/O |
| 闭环评估频率 | 5000 次更新 | 训练入口未集成 | 每 5000 次更新手动评估并登记结果 |
| 模型选择 | 最佳闭环成功率 | 最低离线 val loss | 分开保留离线最佳与闭环最佳；最终用独立测试场景报告 |
| U-Net 宽度 | `[64,128,256]` | `[512,1024,2048]` | 先完成当前模型；另开新实验测 `[128,256,512]`，不是直接替换当前模型 |
| 时间嵌入 / kernel / groups | 64 / 5 / 8 | 128 / 5 / 8 | 当前保留 128 / 5 / 8 |

当前完整模型实验记录约 **2.55 亿参数**，不是 DP 默认宽度的计算规模。不能照搬 DP 的 batch 1024，也不能把相同的 30,000 次更新视为相同训练成本。更新数 × batch size 表示采样窗口数量口径，还应报告实际耗时、参数量和成功率。[已有运行元数据](.runtime/flow_dp3/pickcube-001/full/run.json)

`checkpoint_every=1000` 在当前实现中只改变定期写入 `last.pt` 的频率；发现更低 val loss 时仍会保存 `best.pt` 并更新 `last.pt`。当前入口只保留这两个文件，不保留每个 5000 步节点：做节点评估时需在该节点保留待评估 checkpoint，避免继续训练覆盖。[FM 训练实现](examples/baselines/flow_dp3/train.py)

### 7.4 FM 专有参数：先保留参考实现

| FM 参数 | 当前值 | 推荐值 | 说明 |
| --- | --- | --- | --- |
| `num_inference_steps` | 10 | 10，另测 5 / 20 | 积分采样次数；不复制 DDPM 的 100 |
| `solver` | `consistency` | 保持 | 当前实现名称；切换应作为独立对照 |
| `fm_eps` | 0.01 | 保持 | 连续时间起点 |
| `fm_time_scale` | 100.0 | 保持 | 输入网络的时间尺度，不是 100 次积分 |
| `num_segments` | 2 | 保持 | 当前分段一致性目标参数 |
| `boundary` | 1 | 保持 | 当前目标边界参数 |
| `delta` | 0.01 | 保持 | 双时刻训练间隔 |
| `alpha` | `1e-5` | 保持 | 当前附加损失权重，不可从 DP 噪声 MSE 推导 |
| `noise_scale` | 1.0 | 保持 | 初始高斯噪声尺度 |
| `sigma_var` | 0.0 | 保持 | 关闭积分中的附加随机扰动；初始动作噪声仍随机 |
| `radius1_m` / `radius2_m` | PickCube 0.05 / 0.12 米 | 新训练采用当前值 | 关系点云编码器参数；依据 [半径实验](testpointcloud/SA_RADIUS_REPORT.md) 调整，成功率待验证；其他任务配置仍为 0.10 / 0.20 米 |

来源：[当前 FM 配置](examples/baselines/flow_dp3/configs/pickcube.yaml) 和 [策略实现](examples/baselines/flow_dp3/policy.py)。本项目训练目标包含分段一致性及双时刻项，不能把它等同于一个只预测噪声的 DDPM，也不应在这份文档中改成另一种 FM 损失。

采样步数和执行长度目前从 checkpoint 配置读取，评估 CLI 没有相应覆盖参数；上述 5 / 20 步或执行 4 / 1 的对照不是可以直接传入的现成选项，需要在获得修改同意后提供适配，或通过独立配置实验实施。

### 7.5 推理与评估推荐

| 参数 | 当前默认 | 推荐 |
| --- | --- | --- |
| 权重 | EMA | 保持 EMA；raw weights 仅用于独立对照 |
| 调试局数 | 10 | 10～20 局检查行为 |
| 最终测试局数 | 10 | 每个训练种子至少 100 局 |
| 训练期间验证场景 | 未定义独立列表 | 手动评估可固定环境 seed 1000～1019 |
| 最终测试场景 | 默认从 1000 开始 | 若 1000～1019 用于选模型，最终另用 2000～2099 |
| `policy_seed` | 42 | 初次固定 42；另用 43、44检查采样随机性 |
| PickCube `max_steps` | 200 | 主表使用 100；200 另列补充结果 |
| `num_envs` | 1 | 当前继续单环境 CPU，不照搬 10 / 100 个并行评估环境 |
| 视频 | 默认关闭 | 最终统计先不录视频；另录少量成功与失败样例 |
| 动作范围 | 执行前裁剪 `[-1,1]` | 保持，记录预测与实际执行裁剪比例 |
| 指标 | once / end 成功率、推理时间、裁剪比例 | 全部报告；说明设备、后端和每局上限 |

训练种子控制模型初始化和训练采样，环境种子控制测试场景，`policy_seed` 控制推理动作噪声，三者应分别记录。同一个 seed 数值在 CPU 与 PPU 上不保证相同的随机动作序列。

## 8. 可审阅的运行命令

下面仅展示已有入口能够接受的命令，没有执行。先按 [README](README.md) 初始化当前 PPU SDK、虚拟环境和软件 Vulkan；不要仅为了运行官方示例就替换已经可用的 PPU PyTorch。

### 8.1 官方 PickCube DP 的下载、回放、训练示例

这是 DP 的 CPU 仿真示例；模型计算由 DP 入口选择可用设备。需要准备 DP 自己的依赖环境。示例明确拼接路径，避免原回放脚本中部分路径缺少 `/`。

```bash
cd /mnt/workspace/ManiSkill

python -m mani_skill.utils.download_demo PickCube-v1

# 默认下载目录；如果设置了 MS_ASSET_DIR，使用对应的实际 demos 目录。
DP_DEMO_ROOT="$HOME/.maniskill/demos"

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "$DP_DEMO_ROOT/PickCube-v1/motionplanning/trajectory.h5" \
  --use-first-env-state \
  -c pd_ee_delta_pos -o state -b physx_cpu \
  --save-traj --num-envs 10

cd examples/baselines/diffusion_policy

python train.py \
  --env-id PickCube-v1 \
  --demo-path "$DP_DEMO_ROOT/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5" \
  --control-mode pd_ee_delta_pos --sim-backend physx_cpu \
  --num-demos 100 --max-episode-steps 100 \
  --total-iters 30000 --batch-size 1024 --lr 0.0001 \
  --obs-horizon 2 --pred-horizon 16 --act-horizon 8 \
  --eval-freq 5000 --num-eval-episodes 100 --num-eval-envs 10 \
  --seed 1 --exp-name dp-pickcube-state-reference
```

RGB 示例把回放观测改为 `-o rgb`，数据路径改为 `trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5`，入口改为 `train_rgbd.py --obs-mode rgb`，batch 改为 256。W&B 是可选项，需要时显式增加 `--track`。

### 8.2 当前 FM 使用现有数据的新训练

保留当前模型和配置，先完成一套 PickCube 实验。下面输出目录需为空或尚不存在；不要与已有训练混用。

```bash
cd /mnt/workspace/ManiSkill

python -B examples/baselines/flow_dp3/train.py \
  --config examples/baselines/flow_dp3/configs/pickcube.yaml \
  --env-id PickCube-v1 --data testpointcloud/runs/pickcube-stability/pickcube-pose-hold-v5.h5 \
  --output .runtime/flow_dp3/PickCube-v1-hold-v5/full \
  --device cuda:0 --steps 30000 --batch-size 32
```

此命令按现有 100 条数据划分为 80 条训练 / 20 条验证，不能在结果表写成“100 条训练示范”。完整训练入口会写出配置、数据指纹、运行元数据、逐步指标和 `best.pt / last.pt`。

准备下一轮“100 条训练 + 25 条验证”的本地生成数据时，可单独审阅以下命令；它与上面的现有数据实验是两个数据版本：

```bash
python -B examples/baselines/flow_dp3/prepare_demos.py \
  --generate 150 --count 125 --start-seed 0 \
  --max-attempts 400 --max-steps 200 \
  --num-points 512 --length-scale 1.0 \
  --crop-min 0.05 -0.8 -0.1 --crop-max 1.2 0.8 1.0 \
  --output .runtime/flow_dp3/pickcube-125-task-reference.h5
```

转换可能不足 125 条，入口会保存实际成功数据并报错，需先检查数据清单。只有确实得到 125 条后，`val_ratio=0.2` 才对应 100 / 25。后续用这个文件开始新训练，不用于恢复旧训练。

### 8.3 恢复已有 FM 实验

```bash
python -B examples/baselines/flow_dp3/train.py \
  --config .runtime/flow_dp3/pickcube-001/full/config.yaml \
  --data .runtime/flow_dp3/pickcube-100-v2.h5 \
  --output .runtime/flow_dp3/pickcube-001/full \
  --resume .runtime/flow_dp3/pickcube-001/full/last.pt \
  --device cuda:0 --steps 30000
```

`--steps 30000` 是恢复后达到总计 30,000 次更新，不是再增加 30,000。恢复时配置、数据指纹、batch size 和设备类型需要匹配，输出目录必须是原训练目录。如果 checkpoint 已达到 30,000，入口会拒绝这个目标。
历史 PickCube v1/v2/v4 恢复必须使用原训练目录保存的配置；当前 `pickcube.yaml` 要求 v5 保持示范，不能用于恢复旧实验。

### 8.4 FM 独立测试示例

以下示例使用 2000～2099 的测试场景、100 步上限和 EMA。`best.pt` 目前仍指离线验证损失最佳；若另行选择了闭环最佳节点，应传入该节点保留的 checkpoint。

```bash
python -B examples/baselines/flow_dp3/evaluate.py \
  --checkpoint .runtime/flow_dp3/pickcube-001/full/best.pt \
  --device cuda:0 \
  --episodes 100 --start-seed 2000 --policy-seed 42 \
  --max-steps 100 \
  --output .runtime/flow_dp3/pickcube-001/eval-task-test-100steps.json
```

200 步补充评估需改 `--max-steps 200` 并使用不同输出文件。录像时额外指定 `--save-video`，建议单独运行少量局数。报告文件不得已存在。

## 9. 对比结果必须记录的内容

| 记录项 | 应填写的内容 |
| --- | --- |
| 任务与代码 | 环境 ID、机器人、Git commit、依赖版本 |
| 数据 | 官方发布 / 本地生成、来源类型、数据 SHA256、训练 / 验证 episode 数及列表 |
| 观测 | state / RGB / pointcloud，图像分辨率或点数，使用的本体及目标字段 |
| 控制 | 控制模式、动作维数、坐标系、归一化范围 |
| 训练 | 更新数、batch、学习率、调度、参数量、训练种子、实际耗时 |
| 模型选择 | 离线最低 loss / 闭环最佳成功率，checkpoint 对应更新数 |
| 推理 | EMA / raw、采样器、采样次数、观测长度、预测长度、执行长度 |
| 评估 | CPU / GPU / PPU 计算设备、物理后端、环境种子、策略种子、局数、每局上限 |
| 结果 | `success_once`、`success_end`、各训练种子结果及均值 / 波动、推理延迟、动作裁剪比例 |

DP state 使用的真值信息与视觉策略不同；点云 FM 与 RGB DP 也有模态差异。对比表应显式列出观测，避免把不同输入条件归结为生成算法的优劣。

需要新增多任务适配、闭环评估集成、checkpoint 节点保留或推理参数覆盖时，应先取得程序修改同意。本文中的推荐值尚未写入任何程序或配置。

## 10. 整个项目的任务、本体、观测与维度

本节核对日期：**2026-10-07**，以当前本地源码和实际环境接口为准。第 1～9 节保留此前的实验参考内容；本节的相机与点云尺寸反映后来已落地的配置。下面区分三个范围：ManiSkill 当前注册的 **74 个环境 ID**、DP 基准使用的 **6 个任务**、FlowDP3 已完成训练与评估接口适配的 **PickCube-v1 单任务**。任务已注册不代表其资源已经下载，也不代表已经接入 FlowDP3。

### 10.1 ManiSkill 全部已注册任务及默认本体

清单来自导入 `mani_skill.envs` 后的 `REGISTERED_ENVS`，包含任务、场景环境和空环境。等级变体分别计数；场景内的对象组合与随机种子不另计环境 ID。表中的本体是环境默认值，部分任务可以通过 `robot_uids` 切换本体。

| 类别 / 任务内容 | 环境 ID | 默认本体 |
| --- | --- | --- |
| 桌面：拾取方块 | `PickCube-v1` | `panda` |
| 桌面：SO100 拾取方块 | `PickCubeSO100-v1` | `so100` |
| 桌面：WidowXAI 拾取方块 | `PickCubeWidowXAI-v1` | `widowxai` |
| 桌面：推方块 | `PushCube-v1` | `panda` |
| 桌面：叠方块、搭金字塔 | `StackCube-v1`、`StackPyramid-v1` | `panda_wristcam` |
| 桌面：侧向插销 | `PegInsertionSide-v1` | `panda_wristcam` |
| 桌面：杂乱物体拾取 | `PickClutterYCB-v1` | `panda` |
| 桌面：单个 YCB 物体拾取 | `PickSingleYCB-v1` | `panda_wristcam` |
| 桌面：竖起销杆 | `LiftPegUpright-v1` | `panda` |
| 桌面：放置球体、戳方块 | `PlaceSphere-v1`、`PokeCube-v1` | `panda` |
| 桌面：拉方块、使用工具拉方块 | `PullCube-v1`、`PullCubeTool-v1` | `panda` |
| 桌面：滚动球体 | `RollBall-v1` | `panda` |
| 桌面：推 T 形物体 | `PushT-v1` | `panda_stick` |
| 桌面：装配零件 | `AssemblingKits-v1` | `panda_wristcam` |
| 桌面：FMB 装配 | `FMBAssembly1Easy-v1` | `panda` |
| 桌面：插充电器、转水龙头 | `PlugCharger-v1`、`TurnFaucet-v1` | `panda_wristcam` |
| 双臂：协作拾取、叠方块 | `TwoRobotPickCube-v1`、`TwoRobotStackCube-v1` | 两台 `panda_wristcam` |
| 绘画：三角形、SVG、自由绘制 | `DrawTriangle-v1`、`DrawSVG-v1`、`TableTopFreeDraw-v1` | `panda_stick` |
| 灵巧操作：三指转方块，等级 0～4 | `TriFingerRotateCubeLevel0-v1`、`TriFingerRotateCubeLevel1-v1`、`TriFingerRotateCubeLevel2-v1`、`TriFingerRotateCubeLevel3-v1`、`TriFingerRotateCubeLevel4-v1` | `trifingerpro` |
| 灵巧操作：转阀门，等级 0～4 | `RotateValveLevel0-v1`、`RotateValveLevel1-v1`、`RotateValveLevel2-v1`、`RotateValveLevel3-v1`、`RotateValveLevel4-v1` | `dclaw` |
| 灵巧操作：手内转物体，等级 0～3 | `RotateSingleObjectInHandLevel0-v1`、`RotateSingleObjectInHandLevel1-v1`、`RotateSingleObjectInHandLevel2-v1`、`RotateSingleObjectInHandLevel3-v1` | `allegro_hand_right_touch` |
| 灵巧操作：插花 | `InsertFlower-v1` | `floating_ability_hand_right` |
| 控制：倒立摆平衡、摆起 | `MS-CartpoleBalance-v1`、`MS-CartpoleSwingUp-v1` | 项目内 `CartPoleRobot` |
| 控制：单腿站立、跳跃 | `MS-HopperStand-v1`、`MS-HopperHop-v1` | 项目内 `HopperRobot` |
| 控制：Ant 行走、跑步 | `MS-AntWalk-v1`、`MS-AntRun-v1` | 项目内 `AntRobot` |
| 控制：人形站立、行走、跑步 | `MS-HumanoidStand-v1`、`MS-HumanoidWalk-v1`、`MS-HumanoidRun-v1` | `humanoid` |
| 四足：到达目标、原地旋转 | `AnymalC-Reach-v1`、`AnymalC-Spin-v1` | `anymal_c` |
| 四足：到达目标 | `UnitreeGo2-Reach-v1` | `unitree_go2_simplified_locomotion` |
| 人形：H1 站立 | `UnitreeH1Stand-v1` | `unitree_h1_simplified` |
| 人形：G1 站立 | `UnitreeG1Stand-v1` | `unitree_g1_simplified_legs` |
| 人形：放苹果、搬箱子 | `UnitreeG1PlaceAppleInBowl-v1`、`UnitreeG1TransportBox-v1` | `unitree_g1_simplified_upper_body_with_head_camera` |
| 移动操作：开柜门、抽屉 | `OpenCabinetDoor-v1`、`OpenCabinetDrawer-v1` | `fetch` |
| 移动操作：厨房场景 | `RoboCasaKitchen-v1` | `fetch` |
| 数字孪生：SO100 抓方块 | `SO100GraspCube-v1` | `so100` |
| Bridge 数字孪生：放胡萝卜、放勺子、叠彩色方块 | `PutCarrotOnPlateInScene-v1`、`PutSpoonOnTableClothInScene-v1`、`StackGreenCubeOnYellowCubeBakedTexInScene-v1` | `WidowX250SBridgeDatasetFlatTable` |
| Bridge 数字孪生：放茄子 | `PutEggplantInBasketScene-v1` | `WidowX250SBridgeDatasetSink` |
| 场景操作：通用、ArchitecTHOR、ReplicaCAD | `SceneManipulation-v1`、`ArchitecTHOR_SceneManipulation-v1`、`ReplicaCAD_SceneManipulation-v1` | `fetch` |
| 场景操作：整理房屋，训练 / 验证场景 | `ReplicaCADTidyHouseTrain_SceneManipulation-v1`、`ReplicaCADTidyHouseVal_SceneManipulation-v1` | `fetch` |
| 场景操作：摆餐桌，训练 / 验证场景 | `ReplicaCADSetTableTrain_SceneManipulation-v1`、`ReplicaCADSetTableVal_SceneManipulation-v1` | `fetch` |
| 场景操作：准备杂货，训练 / 验证场景 | `ReplicaCADPrepareGroceriesTrain_SceneManipulation-v1`、`ReplicaCADPrepareGroceriesVal_SceneManipulation-v1` | `fetch` |
| 空环境 / 开发场景 | `Empty-v1` | `panda` |

当前关注的任务中，`PickCube-v1` 支持 `panda / fetch / xarm6_robotiq / so100 / widowxai`；`PushCube-v1` 支持 `panda / fetch`；`StackCube-v1` 支持 `panda_wristcam / panda / fetch`；`PegInsertionSide-v1` 当前声明支持 `panda_wristcam`；`PushT-v1` 使用 `panda_stick`。下面的具体维度按所注明的本体计算，换本体后需要重新核对。

### 10.2 当前 6 个 DP 基准任务：本体与动作维度

本表记录原 DP 基准控制方式；FlowDP3 的 PickCube 使用7维位姿动作或8维关节目标，见第10.6节。这里的 DP 基准设置保持不变。

`panda` 是 7 轴机械臂加平行夹爪，仿真关节状态包含两个手指关节；`panda_wristcam` 在相同本体上增加腕部相机；`panda_stick` 是末端装有固定杆的 7 轴机械臂，没有可控夹爪。

| 任务 | 任务目标 | DP 基准本体 | `qpos` / `qvel` 各自维度 | 回放 / 策略控制模式 | 单步动作维度 |
| --- | --- | --- | --- | --- | --- |
| `PickCube-v1` | 抓起红色方块并移动到目标位置 | `panda` | 9 / 9 | `pd_ee_delta_pos` | 4：末端平移 3 + 夹爪 1 |
| `PushCube-v1` | 将红色方块推到目标区域 | `panda` | 9 / 9 | `pd_ee_delta_pos` | 4：末端平移 3 + 夹爪 1 |
| `StackCube-v1` | 将红色方块叠到绿色方块上 | `panda`；环境默认是 `panda_wristcam` | 9 / 9 | `pd_ee_delta_pos` | 4：末端平移 3 + 夹爪 1 |
| `PegInsertionSide-v1` | 将销杆插入侧面的孔 | `panda_wristcam` | 9 / 9 | `pd_ee_delta_pose` | 7：末端平移 3 + 旋转 3 + 夹爪 1 |
| `PushT-v1` | 将 T 形物体推到目标姿态 | `panda_stick` | 7 / 7 | `pd_ee_delta_pose` | 6：末端平移 3 + 旋转 3 |
| `DrawTriangle-v1` | 用末端杆沿目标三角形绘制 | `panda_stick` | 7 / 7 | `pd_ee_delta_pos` | 3：末端平移 3 |

这是策略使用的控制接口，和运动规划原始示范的 `pd_joint_pos` 不同。普通 Panda / PandaWristCam 的 `pd_joint_pos` 动作是 **8 维**（机械臂 7 + 联动夹爪 1），PandaStick 是 **7 维**；不能把 `qpos=9` 当成 Panda 的动作维度。使用其他控制模式时，以 `single_action_space.shape` 为准。

### 10.3 环境支持的观测类型与通用形状

以下形状省略并行环境维度；实际张量通常在最前面有 `B=num_envs`。设第 `i` 台相机尺寸为 `H_i×W_i`，所有观测相机像素数之和为 `N=Σ(H_i×W_i)`。

| 观测模式 / 字段 | 内容 | 单环境形状 / 维度 |
| --- | --- | --- |
| `none` | 不返回观测内容 | 空字典 |
| `state_dict` | `agent` 本体状态 + `extra` 任务状态，保留字段 | 各字段维度随本体、控制器和任务变化 |
| `state` | 将对应的 `state_dict` 展平成向量；可以包含物体真值 | `(D_state,)`，6 个基准的具体数值见 10.4 |
| `rgb` | 各相机 RGB 图像，以及 `agent / extra / sensor_param` | 每相机 `(H_i,W_i,3)`；RGB 为 `uint8` |
| `depth` | 相机深度，以及本体与任务字段 | 每相机 `(H_i,W_i,1)`；当前 `default` shader 的转换结果以毫米存储 |
| `rgb+depth` / `rgbd` | 同时获取 RGB 和深度 | 每相机 RGB 3 通道 + 深度 1 通道 |
| `segmentation` | Actor / 机器人 Link 的实例 ID | 每相机 `(H_i,W_i,1)` |
| `position` | 相机坐标中的三维位置纹理 | 每相机 `(H_i,W_i,3)`；转换前后单位需要区分 |
| `normal` / `albedo` | 表面法向 / 反照率 | 每相机 `(H_i,W_i,3)`，是否可用取决于 shader |
| `pointcloud.xyzw` | 多相机合并后的世界坐标点云；第 4 列是有效点标志 | `(N,4)`；xyz 单位为米，w 为 0 / 1 |
| `pointcloud.rgb` | 与点逐一对应的颜色 | `(N,3)` |
| `pointcloud.segmentation` | 与点逐一对应的 Actor / Link ID | `(N,1)` |
| `sensor_param` | 相机内外参 | 每相机 `intrinsic_cv: (3,3)`、`extrinsic_cv: (3,4)`、`cam2world_gl: (4,4)` |
| `sensor_data` | 未应用标准纹理转换的 shader 原始输出 | 纹理名称、通道与类型由 shader 决定，不固定为 RGB-D 格式 |

多数环境继承 `BaseEnv` 的通用观测接口，支持上表的模式以及 `rgb+segmentation`、`state+rgb` 等组合。**例外**：4 个 Bridge 数字孪生环境只声明支持 `rgb+segmentation`；`SO100GraspCube-v1` 声明支持 `none / state / state_dict / rgb+segmentation`。这些环境不能直接按通用点云接口使用。

项目没有对全部 74 个环境规定同一个状态维度。对于使用基础本体观测的单机器人，若关节数为 `J`、控制器状态维度为 `C`、任务额外字段总维度为 `E`，则 `D_state=2J+C+E`。灵巧手、本体自定义观测和双机器人还要加入相应附加字段；例如 TriFinger 额外返回指尖位姿 21 维与速度 9 维，DClaw 额外返回三指指尖位姿 21 维，AllegroTouch 额外返回触觉读数。其他任务未在本节逐一实例化，其状态总维度应按对应配置的 `single_observation_space` 或实际观测核对，不能套用下面六项的数字。

### 10.4 六个基准任务的原生状态与视觉附带状态

以下数值通过 `physx_cpu`、单环境 `reset(seed=0)` 实测，并与各任务的 `_get_obs_extra` 定义核对；控制模式和本体对应 10.2。这些控制模式没有附加的控制器状态。普通位姿 `pose` 是 **7 维：位置 xyz 3 + 四元数 wxyz 4**，位置 / 差向量是 3 维，标志 / 半径是 1 维。

| 任务 | 原生 `state / state_dict` 的任务字段 `extra` | `agent` 维度 | `extra` 维度 | 原生 `state` 总维度 | 纯视觉模式保留的 `extra` | 视觉附带状态总维度 |
| --- | --- | --- | --- | --- | --- | --- |
| `PickCube-v1` | `is_grasped(1)`、`tcp_pose(7)`、`goal_pos(3)`、`obj_pose(7)`、`tcp_to_obj_pos(3)`、`obj_to_goal_pos(3)` | 18 | 24 | **42** | `is_grasped(1)`、`tcp_pose(7)`、`goal_pos(3)` | **29** |
| `PushCube-v1` | `tcp_pose(7)`、`goal_pos(3)`、`obj_pose(7)` | 18 | 17 | **35** | `tcp_pose(7)` | **25** |
| `StackCube-v1` | `tcp_pose(7)`、`cubeA_pose(7)`、`cubeB_pose(7)`、`tcp_to_cubeA_pos(3)`、`tcp_to_cubeB_pos(3)`、`cubeA_to_cubeB_pos(3)` | 18 | 30 | **48** | `tcp_pose(7)` | **25** |
| `PegInsertionSide-v1` | `tcp_pose(7)`、`peg_pose(7)`、`peg_half_size(3)`、`box_hole_pose(7)`、`box_hole_radius(1)` | 18 | 25 | **43** | `tcp_pose(7)` | **25** |
| `PushT-v1` | `tcp_pose(7)`、`goal_pos(3)`、`obj_pose(7)` | 14 | 17 | **31** | `tcp_pose(7)` | **21** |
| `DrawTriangle-v1` | `tcp_pose(7)`、`goal_pose(7)`、`tcp_to_verts_pos(9)`、`goal_pos(3)`、`vertices(9)` | 14 | 35 | **49** | `tcp_pose(7)` | **21** |

这里的“纯视觉模式”指 `rgb / rgb+depth / pointcloud` 等未请求 `state` 的模式；它们仍保留本体与任务允许的附带字段。视觉附带状态维度是把这些 `agent / extra` 字段拼接后的维度，也是本地 DP 视觉入口默认状态提取器使用的字段集合。比如 PickCube 视觉状态是 29 维，包含 `is_grasped`；FlowDP3 自行去掉它并转换坐标，得到 10.6 中的 28 维。`state+rgb` 则会请求更完整的任务真值，不能按纯视觉维度计算。

DrawTriangle 的两个 9 维字段分别是三角形 **3 个三维顶点**与 TCP 到这些顶点的差向量。绘制痕迹通过图像 / 点云体现，没有作为完整绘制历史向量加入上述 49 维状态。

### 10.5 当前相机、原始点云数量与 crop

以下是五个任务已落地的相机 / crop 预设；PushT 沿用其原有相机。相机位置与朝向不改变 `state` 维度，但会改变图像内容和物体可见点数。`human_render` / `render_camera` 用于展示，其分辨率不能当成策略观测分辨率。

| 任务 / 本体配置 | 观测相机及分辨率 | 单帧原始 `pointcloud.xyzw` 形状，省略 B |
| --- | --- | --- |
| PickCube / Panda | `base_camera: 256×256` | `(65536,4)` |
| PushCube / Panda | `base_camera: 256×256` | `(65536,4)` |
| StackCube / DP 的 Panda | `base_camera: 256×256` | `(65536,4)` |
| StackCube / 默认 PandaWristCam | `base_camera: 256×256` + `hand_camera: 128×128` | `(81920,4)` |
| PegInsertionSide / PandaWristCam | `base_camera: 256×256` + `hand_camera: 128×128` | `(81920,4)` |
| PushT / PandaStick | `base_camera: 128×128` | `(16384,4)` |
| DrawTriangle / PandaStick | `base_camera: 256×256` | `(65536,4)` |

这里的 N 是所有像素对应的点槽位数，包含 w=0 的无效点；它不是 crop 后的点数，也不是机械臂或物体的可见点数。`pointcloud.rgb`、`pointcloud.segmentation` 使用相同的 N，最后一维分别为 3 和 1。

| 任务 | base 相机 eye，世界坐标 / m | 相机 target，世界坐标 / m | FOV | crop_min，本体基座坐标 / m | crop_max，本体基座坐标 / m |
| --- | --- | --- | --- | --- | --- |
| PickCube | `(0.30,-0.30,0.35)` | `(0,0,0.12)` | 60° | `(0.44,-0.23,-0.03)` | `(0.79,0.25,0.52)` |
| PushCube | `(0.30,-0.30,0.35)` | `(0.15,0,0.08)` | 75° | `(0.40,-0.23,-0.03)` | `(1.06,0.25,0.52)` |
| StackCube | `(0.30,0.35,0.28)` | `(0,0,0.07)` | 65° | `(0.36,-0.36,-0.03)` | `(0.87,0.36,0.52)` |
| PegInsertionSide | `(0.30,-0.35,0.55)` | `(0,0.10,0.12)` | 75° | `(0.34,-0.40,-0.03)` | `(0.90,0.62,0.52)` |
| DrawTriangle | `(0.25,-0.40,0.50)` | `(-0.10,-0.10,0.04)` | 60° | `(0.28,-0.35,-0.03)` | `(0.80,0.18,0.52)` |

crop 优先覆盖操作物体与末端活动区域，不要求保留机械臂所有 Link；仍可能保留桌面点。确认某类点被采集时，用 `segmentation` 中的 Actor / Link ID 与场景对象、机器人 Link 的 `per_scene_id` 对应，在有效点、crop、预采样和 FPS 四个阶段分别计数。这个标记用于实验统计，没有作为模型的输入通道。

### 10.6 FlowDP3 实际输入、序列与适配边界

本节描述 **PickCube + Panda** 的实际输入；新 v5 使用 **`pd_ee_delta_pose`、7维动作＋40步真实保持示范**，v6 对照使用 **`pd_joint_pos`、8维关节绝对目标**。历史 v4 保留7维位姿动作，v1/v2 保留4维平移动作。环境先产生原始点云，再过滤有效点、变换到机器人基座坐标、crop、最多预采样 4096 点，最后 **FPS 下采样到 512 点**。点数不足 512 时适配器报错。512 是机器人、目标物体、桌面等全部保留类别合计的点数，没有固定的类别配额。

| 模型 / 数据字段 | 内容 | 单帧形状，省略 B |
| --- | --- | --- |
| `state[0:9]` | Panda 关节位置 | `(9,)` |
| `state[9:18]` | Panda 关节速度 | `(9,)` |
| `state[18:25]` | 基座坐标中的 TCP 位姿，xyz + wxyz | `(7,)` |
| `state[25:28]` | 基座坐标中的目标位置 | `(3,)` |
| `state` 合计 | 本体、TCP 与任务目标；不包含物体真值、抓取标志或成功标志 | **`(28,)`** |
| `pointcloud_distance` | 每点 `[dx,dy,dz,‖d‖] / length_scale`，d 是点相对 TCP 的基座坐标差 | **`(512,4)`** |
| `action` | v4/v5：末端平移3＋旋转3＋夹爪1；v6：7个关节目标角度＋夹爪1；v1/v2：平移3＋夹爪1 | **v4/v5 `(7,)`；v6 `(8,)`；v1/v2 `(4,)`** |

`pointcloud_distance` 的第 4 维是到 TCP 的距离，和原始 `xyzw` 的有效标志不同。当前 `length_scale=1.0`，xyz 差与距离保持米制数值；RGB、分割 ID、相机参数和原始有效标志均未拼入这 4 个通道。

当前策略配置 `n_obs_steps=2`、`horizon=16`、`n_action_steps=8`：批大小为 B 时，条件输入是 `state: (B,2,28)` 与 `pointcloud_distance: (B,2,512,4)`；动作维度D为7或8，预测 `(B,16,D)`，执行 `(B,8,D)`。一条轨迹保存 `action: (T,D)`、`state: (T+1,28)`、`pointcloud_distance: (T+1,512,4)`。v5/v6允许包括terminal在内的每个真实状态成为当前观测，动作尾部按控制类型填充；额外 `task_metrics` 只用于核验，不输入模型。目标位置来自环境 `extra.goal_pos`，转到基座坐标进入state最后3维；目标标记在传感器观测中隐藏。稳定保持修复见 [稳定保持报告](testpointcloud/PICKCUBE_STABILITY_REPORT.md)。

PickCube 最新录制默认在同一文件内保存 `ee` / `joint` 两个分支，分别对应v5/v6契约和各自的实际观测序列；训练通过 `--control-mode ee/joint` 选择，推理默认读取checkpoint中的控制器。Joint动作仍使用实际TCP计算矢量距离，保留28维state和512×4点云。每条收录轨迹同步生成场景MP4。数据结构、命令及验证见 [双控制录制报告](testpointcloud/DUAL_CONTROL_REPORT.md)。

五个相机预设任务可复用点云处理接口，**不能据此认为五个任务都已获得可训练的 28 维状态接口**。例如 PushCube 的纯视觉 `extra` 不提供 `goal_pos`，PandaStick 本体是 7 维关节位置 / 速度，PegInsertionSide 的策略动作是 7 维；它们需要各自的任务状态与动作适配。PushT 当前没有上述五任务的 crop 预设。

核对依据：[环境注册](mani_skill/utils/registration.py)、[原生观测接口](mani_skill/envs/sapien_env.py)、[本体观测](mani_skill/agents/base_agent.py)、[观测模式解析](mani_skill/envs/utils/observations/__init__.py)、[点云转换](mani_skill/envs/utils/observations/observations.py)、[DP 视觉状态提取](examples/baselines/diffusion_policy/diffusion_policy/utils.py)、[相机与 crop 配置](mani_skill/utils/task_pointcloud.py)、[FlowDP3 观测适配器](examples/baselines/flow_dp3/obs_adapter.py)、[策略配置](examples/baselines/flow_dp3/configs/pickcube.yaml)。六任务状态维度和观测相机尺寸已在当前代码下实际创建环境并核对；其余任务的清单与默认本体按注册及源码核对。
