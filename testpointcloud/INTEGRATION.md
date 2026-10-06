# crop 与相机正式接入（2026-10-06）

用户确认后，五任务采用[MULTI_TASK_REPORT.md](MULTI_TASK_REPORT.md)中推荐的相机和操作区。
基座点云相机为256×256，FPS总点数仍为512，预采样4096/seed42，模型距离点云仍为4通道。
相机坐标是世界系，crop是基座系；未添加语义保底采样或合成点。

## 文件调整

| 文件 | 修改内容 |
| --- | --- |
| [task_pointcloud.py](../mani_skill/utils/task_pointcloud.py) | 新增统一的五任务eye/target/FOV/分辨率/crop配置及相机创建函数 |
| [pick_cube.py](../mani_skill/envs/tasks/tabletop/pick_cube.py) | Panda的默认点云相机：低左视角、256×256、60° |
| [push_cube.py](../mani_skill/envs/tasks/tabletop/push_cube.py) | Panda的默认点云相机：目标前移、低左75° |
| [stack_cube.py](../mani_skill/envs/tasks/tabletop/stack_cube.py) | Panda/PandaWristCam基座相机：低右视角、65° |
| [peg_insertion_side.py](../mani_skill/envs/tasks/tabletop/peg_insertion_side.py) | 默认基座相机：较高左侧、75°，保留腕部相机 |
| [draw_triangle.py](../mani_skill/envs/tasks/drawing/draw_triangle.py) | 默认点云相机：左侧斜视、60°，256×256 |
| [scene_bounds.py](../examples/baselines/flow_dp3/scene_bounds.py) | 从统一预设获取各任务操作区 |
| [obs_adapter.py](../examples/baselines/flow_dp3/obs_adapter.py) | 默认crop更新；新v2契约保存完整相机参数；按旧v1/新v2恢复对应相机；点云接口支持PandaStick的TCP link |
| [prepare_demos.py](../examples/baselines/flow_dp3/prepare_demos.py) | 转换目标环境按观测契约创建；新数据记录v2契约；CLI的默认crop同步 |
| [evaluate.py](../examples/baselines/flow_dp3/evaluate.py) | 验证checkpoint契约，并按其保存的相机创建环境 |
| [probe_runtime.py](../examples/baselines/flow_dp3/probe_runtime.py) | 真实渲染探测统一相机/crop，报告明确记录参数；默认crop在worker解析，保留父进程的驱动隔离 |
| [dataset_io.py](../visual/dataset_io.py) | 默认训练数据切换到camera-v5 |
| [export_videos.py](../visual/export_videos.py) | 按数据契约回放相机，标题显示实际原生分辨率 |
| [test_data_contract.py](../examples/baselines/flow_dp3/tests/test_data_contract.py) | 更新操作区验证，并测试旧相机恢复、保存相机参数、异常契约及PandaStick TCP |
| [项目README](../README.md)、[introduce.md](../introduce.md)、[visual/README.md](../visual/README.md) | 同步参数、版本命名、采集/训练/可视化命令及旧模型兼容说明 |

人类观看使用的`render_camera`保持原配置；变更的是生成策略点云的`base_camera`。
非实验机器人（例如Pick/Push/Stack的Fetch）的原相机配置保留。
Stack的原默认机器人仍是`panda_wristcam`，其`hand_camera`保留128×128；实验统计表的
Stack数据显式使用`panda`，因此不能直接将单相机点数套到默认双相机机器人。

## 新的正式训练数据

- 路径：`.runtime/flow_dp3/pickcube-100-camera-v5.h5`及同名`.json`。
- 来源：现有`pickcube-100-scene-v3.raw.h5`及`.raw.json`，未覆盖旧数据。
- 真实执行从原生关节动作到`pd_ee_delta_pos`的转换，只保存最终成功轨迹。
- 100/100条转换最终成功；7720个动作、7820帧观测。
- 每帧`[512,4]`点云、28维状态，动作4维，T+1观测/T动作。
- 与旧scene-v3的100条对应轨迹比较，全部状态、动作最大差异均为0；点云按新相机/crop重新生成。
- 完整检查覆盖有限值、距离通道、基座系crop范围、动作范围、源文件哈希和元信息。
  距离通道最大数值误差约`5.96e-8`，所有采样点在新操作区内，排除约z=-0.92m的地面。
- `DemoDataset`训练读取已检查：80条训练、20条验证，episode互不重叠；窗口形状符合策略接口。

数据契约v2保存相机pose的xyz+wxyz、分辨率、FOV弧度、near/far及shader。评估与视频回放
使用保存参数，所以以后调整任务默认值不会悄悄改变已有v2模型的观测。
旧v1契约显式恢复旧128×128相机及旧crop；已用真实scene-v3观测验证。
checkpoint文件的`format_version`仍为1，与观测契约版本是不同字段。

新训练应使用camera-v5与新的训练输出目录；旧checkpoint继续按旧契约运行。
本次没有训练策略，也没有测量新策略成功率。

## 验证结果与证据

| 检查 | 结果/记录 |
| --- | --- |
| 数据契约单元测试 | 8项通过，`runs/integration-unit01.log` |
| 可视化单元测试 | 10项通过，`runs/integration-visual01.log` |
| 五任务直接使用正式默认基座相机 | 45个参考帧与筛选实验FPS512逐点匹配，最大误差0；`runs/integration-defaults01.json` |
| 历史scene-v3的真实相机恢复 | 原128×128，初始状态与点云最大误差0，同上 |
| Stack默认PandaWristCam | 3个seed重置检查，基座256×256、腕部128×128，FPS512通过；`runs/integration-stack-wristcam01.json` |
| metadata/CPU仿真/真实点云编码探测 | 3项通过，`runs/integration-probe01.json` |
| 100条新数据的完整检查与训练读取 | 通过；[integration-dataset01.json](runs/integration-dataset01.json) |
| 新数据episode 0真实视频回放 | 75帧逐帧匹配，状态和点云最大误差0，最终成功，MP4解码通过；[quality.json](runs/integration-video-v5/quality.json) |
| 默认可视化导出 | [camera-v5第0帧双图](runs/integration-camera-v5-frame0.png) |

第一条轨迹第一帧进一步通过源像素和实体ID核对：原始方块411点、crop保留411点、
预采样80点、FPS512保留12点；机械臂100点、桌面400点、地面0点，合计512。
腕部84点已包含在机械臂100点中，不能再次相加。这帧特征与正式HDF5最大误差0。

- [RGB、实体上色FPS512与方块局部对照](runs/integration-camera-v5-semantics.png)
- [该帧实体计数及索引诊断](runs/integration-camera-v5-semantics.json)
- [75帧真实动作回放视频](runs/integration-video-v5/episode_00000.mp4)

审计脚本和日志都在`testpointcloud`。源数据的SHA256和正式数据的指纹保存在manifest及审计报告。

## 保留的限制

这些操作区优先目标与末端，允许裁掉整臂的其他部分。相机使目标原始点增多，但普通FPS
仍会给桌面/画布分配多数点；第一帧12个方块点不能代表所有帧的最低点数。
筛选实验中的Draw仍有3/700帧目标轮廓被FPS完全丢弃，修改正式默认值不会消除这个算法限制。
Peg插孔盒覆盖改善时，插杆FPS中位数有所下降，详见实验报告。
完整28维状态/4维动作训练与数据转换目前仍只支持PickCube；其他四个任务接入了相机和
点云crop/TCP接口，没有声称完成它们的策略状态与动作适配。

## 使用

从项目根目录读取新的默认数据：

```bash
.venv-ppu/bin/python -B visual/pointcloud.py \
  --episode 0 --frame 0 --export-backend matplotlib --export-view both \
  --png .runtime/flow_dp3/my-camera-v5-cloud.png
```

采集、训练、评估的完整命令见[操作说明](../introduce.md)。实验复现使用新输出名，所有
数据生成、图像导出和测试审计均拒绝覆盖已有输出。
