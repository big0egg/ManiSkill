# PickCube双控制数据与专家场景录像

本次按授权修改录制、数据读取、训练、推理和可视化入口。PickCube默认一次生成原始专家轨迹，再分别真实回放末端控制与Joint控制，在同一个HDF5中保存两个分支。每个分支独立保存实际观测、动作和40步稳定保持结果；最新默认按成功保存编号每10条抽1条完整场景MP4。

## 数据结构与选择参数

```text
pickcube-dual.h5                    # layout=dual_control_v1
  ee/                              # 契约v5，pd_ee_delta_pose
    episode_00000/
      state                        # (T_ee+1,28)
      pointcloud_distance          # (T_ee+1,512,4)
      action                       # (T_ee,7)
      task_metrics/
  joint/                           # 契约v6，pd_joint_pos
    episode_00000/
      state                        # (T_joint+1,28)
      pointcloud_distance          # (T_joint+1,512,4)
      action                       # (T_joint,8)
      task_metrics/
pickcube-dual.json
pickcube-dual-videos-full/
  ee/episode_00000_seed_0.mp4
  joint/episode_00000_seed_0.mp4
```

两种控制可能产生不同步数，不能把两份动作挂到同一条观测上。回放仅恢复原始初始状态，随后执行真实动作；源episode在两个分支均最终成功、且整个保持阶段成功时才收录。失败时两份视频缓冲都丢弃。分支内保存各自契约与manifest，根manifest记录配对源ID和拒绝原因。同名episode对应同一个源ID，因此同一划分比例和seed下训练/验证划分一致，归一化只统计所选分支的训练episode。

录制 `--control-mode` 默认对PickCube使用 `both`，也可填 `ee` / `joint` 单独录制。其他四任务保留原控制器，使用同一录像抽样规则。完整控制器名称继续支持。`--save-video` 默认开启抽样录像，`--video-every` 默认10：成功保存编号0/10/20……（第1/11/21条）录制，两个分支对应相同源轨迹。失败不占名额，选中但失败的录像缓冲被丢弃；未选中轨迹通过RecordEpisode触发器直接跳过capture_image，无录像渲染或编码，所有训练观测与动作仍逐步保存。每条元信息增加 `video_recorded`；只有选中轨迹带视频路径/帧数/FPS。manifest记录 `video_every` 和抽样规则。`--video-every 1` 恢复每条录像，`--no-save-video` 完全关闭；`--video-dir` 指定新目录，`--video-fps` 默认为控制频率20Hz，仅影响播放速度。抽样视频包含真实初始状态和每一步执行结果，共T+1帧，使用与策略 `videos-full` 相同的场景录像方式，传感器相机和crop不变。

训练 `--control-mode ee/joint` 选择分支；两个命令可以使用同一个 `pickcube.yaml` 或小模型配置。显式控制参数自动匹配v5/v6的7/8维动作接口，模型宽度、SA半径和优化器设置沿用配置。默认双分支训练选择ee。单分支历史数据、旧checkpoint继续按原契约加载；旧数据不会因此获得保持示范。推理默认自动跟随checkpoint，显式控制参数仅校验一致性；7维与8维仍需分别训练，错误搭配会在创建环境前拒绝。

Joint分支也继续读取实际TCP，计算基座坐标差 `point_base - tcp_base` 及其模长。state仍为关节位置9、关节速度9、TCP位姿7、目标位置3，共28维；点云仍为512×4。目标坐标来源、相机、crop、FPS和SA半径均沿用现有接口。

## 此前全量录像的小批量验证

全部实验产物位于 [runs/dual-control](runs/dual-control)。本次使用已有原始专家文件 `.runtime/flow_dp3/PickCube-v1-first-v1.raw.h5`，真实回放并收录源ID 0～4共5条配对示范。

| 检查 | 结果 |
| --- | --- |
| 末端/Joint分支 | 各5条，各560个动作、565帧观测；动作7/8维，state28维，点云512×4 |
| 专家质量门槛 | 两个分支每条最后40步均成功，最终成功；未把诊断输入策略 |
| 专家MP4 | 10份均完整解码，512×512、20FPS；每份帧数与对应T+1观测一致，共1130帧 |
| 与此前不录像数据比较 | 每条动作、state和点云特征逐元素最大误差均为0；同源两个分支初始观测一致 |
| 两种参数选择训练 | 使用同一个pickcube_smoke.yaml，分别20步CPU训练，自动得到7/8维输出 |
| 两种控制恢复训练 | 各自从last.pt继续到21步，所选分支、配置、数据指纹与动作接口一致；新checkpoint记录data_group |
| 训练/验证划分 | 两种控制均为4条训练、1条验证，episode集合一致 |
| checkpoint真实推理 | 各16步闭环执行，并生成17帧MP4和逐步诊断；控制器符合checkpoint |
| 控制方式误配 | 对Joint checkpoint指定ee被拒绝，未创建推理报告 |
| 相关回归测试 | 45项通过，覆盖旧契约、动作时序、距离几何、分支隔离、配对拒绝和可视化 |

验证记录：[录制日志](runs/dual-control/record.log)、[视频与逐元素比较](runs/dual-control/validation.json)、[训练与推理校验](runs/dual-control/workflow-validation.json)、[测试日志](runs/dual-control/tests.log)。已有真实W&B离线SDK测试受当前沙箱Unix/TCP socket限制，本次相关测试排除了该项，日志模块未修改。

小模型完成20步训练及1步恢复验证，16步短推理均未成功；这用于验证接口与链路，不代表正式策略成功率。未启动新的正式模型训练，也未覆盖旧数据或权重。双分支会增加回放、渲染时间和数据空间；只比较一种控制时可显式选择单分支，或关闭录像。

专家样例：[末端控制](runs/dual-control/pickcube-dual-videos-full/ee/episode_00000_seed_0.mp4)、[Joint控制](runs/dual-control/pickcube-dual-videos-full/joint/episode_00000_seed_0.mp4)。两份都是与最终训练数据对应的场景录像。

## 每10条抽1条录像的验证

按后续授权，专家录像默认改为 `--video-every 10`；实现使用RecordEpisode的录像触发器，在整条源episode及两个分支执行期间保持同一个选择状态，未选中轨迹不调用capture_image。失败时丢弃录像缓存，继续以成功保存编号决定下一条是否录制。此改动只影响新启动的录制进程，已运行的进程不会自动切换；推理和额外面板导出的录像规则不变。

实验程序沿用现有测试和核验脚本，没有新增独立实验程序。产物全部位于 [runs/video-sampling](runs/video-sampling)：独立生成12条原始专家示范，真实回放收录前11条配对示范。

- 两分支均完整保存11条，EE共1248个动作、Joint共1245个动作，逐步点云/state和40步保持全部保留。
- 两分支只在保存编号0、10（第1、11条）生成MP4，共4份；每份完整包含T+1帧观测，帧数分别115、120，合计470帧，全部完整解码通过。其余9条仍有完整训练数据，元信息为 `video_recorded=false` 且不带视频路径。
- 与此前100条保持数据中的对应轨迹比较，全部11条的动作、state、点云特征逐元素最大误差均为0。
- 28项相关回归测试通过，包括失败不占收录编号、两个分支对应同一抽样编号、未选中轨迹实际不调用录像渲染、触发器切换不混入缓存、非法间隔拒绝及旧数据/契约兼容。旧的5条全量录像数据也通过更新后的核验脚本。
- 回放阶段耗时223.9秒，平均20.4秒/配对示范；此前全量录像5条耗时137.4秒，平均27.5秒。平均耗时粗略减少约26%，两次样本和运行负载不同，此比较不是严格配对测速。

记录：[回放日志](runs/video-sampling/record.log)、[视频与逐元素核验](runs/video-sampling/validation.json)、[回归测试](runs/video-sampling/tests-regression.log)、[旧数据核验](runs/video-sampling/legacy-validation.json)。第11条样例：[EE](runs/video-sampling/pickcube-dual-videos-full/ee/episode_00010_seed_10.mp4)、[Joint](runs/video-sampling/pickcube-dual-videos-full/joint/episode_00010_seed_10.mp4)。

## 使用命令

先按 [introduce.md第1.1节](../introduce.md) 初始化SDK、虚拟环境、源码路径、资源目录及软件Vulkan。以下从项目根目录执行，正式输出路径必须是新路径。

```bash
# PickCube默认双分支，每10条成功示范抽1条完整录像。
python -B examples/baselines/flow_dp3/prepare_demos.py \
  --env-id PickCube-v1 --generate 120 --count 100 --max-attempts 2000 \
  --output .runtime/flow_dp3/pickcube-dual.h5

# 同一份数据和同一配置，分别训练。
python -B examples/baselines/flow_dp3/train.py \
  --data .runtime/flow_dp3/pickcube-dual.h5 --control-mode ee \
  --config examples/baselines/flow_dp3/configs/pickcube.yaml \
  --output .runtime/flow_dp3/pickcube-dual-ee --wandb-mode disabled
python -B examples/baselines/flow_dp3/train.py \
  --data .runtime/flow_dp3/pickcube-dual.h5 --control-mode joint \
  --config examples/baselines/flow_dp3/configs/pickcube.yaml \
  --output .runtime/flow_dp3/pickcube-dual-joint --wandb-mode disabled

# 推理省略control-mode也会自动使用Joint；指定时必须与模型匹配。
python -B examples/baselines/flow_dp3/evaluate.py \
  --checkpoint .runtime/flow_dp3/pickcube-dual-joint/best.pt --control-mode joint \
  --episodes 20 --save-video --wandb-mode disabled \
  --output .runtime/flow_dp3/pickcube-dual-joint/evaluation.json

# 从双分支文件导出Joint点云图。
python -B visual/pointcloud.py \
  --dataset .runtime/flow_dp3/pickcube-dual.h5 --control-mode joint \
  --episode 0 --frame 0 --png .runtime/flow_dp3/joint-frame0.png
```

可以使用 `--source` 复用已有原始专家数据，避免重新调用规划器。恢复Joint训练继续传同一个控制参数。若迁移HDF5及JSON，应同时迁移相对路径指向的视频目录；额外面板录像仍需原始专家文件用于真实回放核对。

生产改动：[prepare_demos.py](../examples/baselines/flow_dp3/prepare_demos.py)、[control_modes.py](../examples/baselines/flow_dp3/control_modes.py)、[dataset.py](../examples/baselines/flow_dp3/dataset.py)、[train.py](../examples/baselines/flow_dp3/train.py)、[evaluate.py](../examples/baselines/flow_dp3/evaluate.py)，以及visual的分支读取与参数。相关说明同步至README、introduce、task、version和visual/README；实验程序全部放在testpointcloud。
