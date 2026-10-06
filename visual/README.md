# Flow DP3 数据观测可视化

所有工具只读输入 HDF5，不修改训练代码、数据或 checkpoint。默认数据为
`.runtime/flow_dp3/pickcube-100-scene-v3.h5`，可通过 `--dataset` 指定其他文件。
从项目根目录执行下面的命令。

## 1. 安装 Open3D 可视化环境

```bash
cd /mnt/workspace/ManiSkill
bash visual/setup_env.sh
```

脚本创建 `.runtime/visual-venv`，复用项目和镜像已有依赖，将新增包安装到这个独立环境。
不向 `.venv-ppu` 或系统 Python 安装包。安装时保留 PPU 框架约束及原 protobuf、click。
Linux x86_64 使用 `open3d-cpu==0.19.0`。

默认阿里云源；若实例无法访问，可切换官方源：

```bash
VISUAL_PIP_INDEX_URL=https://pypi.org/simple/ bash visual/setup_env.sh
```

域名解析失败时安装未完成，须先解决实例网络，再执行脚本。Open3D 交互、Open3D 离屏导出和 PLY 导出需要完成安装。
服务器上的点云 PNG/MP4 默认使用 Matplotlib，可直接用 `.venv-ppu/bin/python`，无需安装 Open3D。
场景视频工具使用现有 Matplotlib、ImageIO、SAPIEN，也可以直接用 `.venv-ppu/bin/python`。

在本地桌面使用时，安装 `visual/requirements.txt`，复制 `visual/` 和训练 HDF5，
通过 `--dataset /实际路径/数据.h5` 指定数据即可。点云查看不需要 ManiSkill、PPU 或原始轨迹。

## 2. 点云显示与可选点击坐标

```bash
# 默认：基座系点云、基座坐标轴、距离着色和24条从末端出发的矢量线。
.runtime/visual-venv/bin/python -B visual/pointcloud.py --episode 0

# 开启选点；点击点即可在侧栏和终端查看坐标。
.runtime/visual-venv/bin/python -B visual/pointcloud.py --episode 0 --pick-points

# 放大末端附近1米内的点；适合检查夹爪、方块等近处细节。
.runtime/visual-venv/bin/python -B visual/pointcloud.py \
  --episode 0 --pick-points --near-radius 1
```

`--pick-points` 默认关闭。开启后单击选点，显示原始点索引、基座系 XYZ（米）、
相对末端 XYZ（米）、距离（米）和原始策略四维特征。点选取使用投影后的实际数据点，
点击空白不会虚构坐标；重叠点优先选择前景点。按下鼠标时暂停播放，拖动仍旋转视角，滚轮缩放。
侧栏可切换 episode、拖动帧滑块、前后单步及播放/暂停，选点信息切帧后清空。
点索引仅在当前帧有意义，FPS 采样点之间不表示跨帧物体对应。

默认基座坐标轴位于机器人基座原点，X/Y/Z 为红/绿/蓝；TCP 为青色，目标为绿色。
前三通道是基座轴方向上的相对末端位移，不是末端局部旋转坐标。
第四通道是距离，不是 RGB/透明度。显示位置与距离按数据集 `length_scale` 还原为米：

```text
relative_xyz_m = features[:3] * length_scale
base_xyz_m = relative_xyz_m + state[tcp_base_position]
distance_m = features[3] * length_scale
```

其他参数：

| 参数 | 含义 |
| --- | --- |
| `--episode 54` / `--episode episode_00054` | 轨迹索引或完整名称 |
| `--frame 30` | 初始观测帧，包含末帧 `T` |
| `--coordinate-frame relative` | 末端平移原点视图；基座轴仍显示在正确位置 |
| `--vector-count 512` | 显示全部矢量线；`0` 关闭线条，点云仍显示 |
| `--near-radius 1` | 只过滤显示范围，不更改数据或原点索引 |
| `--color-max 1` | 固定色标上限（米）；超出上限的点颜色饱和 |
| `--point-size 8` | 点的像素大小 |
| `--play --fps 20` | 自动播放及播放速度 |
| `--export-backend matplotlib` | PNG/MP4 默认后端，CPU/Agg，无需桌面、Open3D 或仿真 |
| `--export-backend open3d` | 保留 Open3D 离屏导出；需要 Open3D 及可用的 EGL/Mesa |

整条轨迹使用固定初始相机范围、固定色标，切帧不重新缩放。

### 云实例无桌面时

普通 GUI 和点击选点需要本地桌面、远程桌面或已配置的 X11 显示。
没有 `DISPLAY`/`WAYLAND_DISPLAY` 的终端会给出提示，不能直接弹出交互窗口。
服务器默认导出使用 Matplotlib Agg 和 ImageIO/FFmpeg，CPU 渲染，无需 `DISPLAY`、Open3D、EGL、Vulkan 或 PPU。
直接从训练 HDF5 生成点云，不需要原始 `.raw.h5`，也不启动仿真。

```bash
# 指定帧的全场景 PNG；无需额外安装依赖。
.venv-ppu/bin/python -B visual/pointcloud.py --episode 0 --frame 30 \
  --png .runtime/flow_dp3/cloud-frame30.png

# 末端附近点云 PNG 和整条轨迹 MP4，可同时导出。
.venv-ppu/bin/python -B visual/pointcloud.py --episode 0 --near-radius 1 \
  --png .runtime/flow_dp3/cloud-near-frame0.png \
  --video .runtime/flow_dp3/cloud-near-episode0.mp4

# 显式选择后端；--frame 为视频起始帧，--fps 只影响播放速度。
.venv-ppu/bin/python -B visual/pointcloud.py --episode 0 --frame 30 \
  --export-backend matplotlib --fps 20 \
  --video .runtime/flow_dp3/cloud-from30.mp4

# 安装 Open3D 后仍可使用原有 EGL 离屏渲染。
.runtime/visual-venv/bin/python -B visual/pointcloud.py --episode 0 \
  --export-backend open3d --png .runtime/flow_dp3/cloud-open3d-frame0.png \
  --ply .runtime/flow_dp3/cloud-frame0.ply
```

Matplotlib PNG/MP4 默认960×768，包含基座坐标轴、末端和目标标记、从末端出发的矢量箭头，
以及以米为单位的距离色标。视频保存从 `--frame` 到末帧 `T` 的所有观测，保持相机范围与色标固定。
`--near-radius` 仅过滤显示点，矢量线从当前 TCP 到实际数据点；`--vector-count 0` 可关闭箭头。

离屏导出不能与 `--pick-points` 同时使用。PLY 只包含当前显示点及距离颜色，
不包含基座坐标轴；需要完整交互界面与原始四维特征时，应在桌面直接读取 HDF5。
PNG/MP4 的后端选择不影响原有 Open3D GUI/选点功能或 PLY 导出。
选择 Open3D 后端时使用 EGL，SAPIEN 场景视频使用 Vulkan；两种渲染环境需要分别验证。

## 3. 场景、相机、点云和状态同步视频

沿用项目 README 的 CPU/Mesa 初始化。没有初始化终端时，可执行：

```bash
export PYTHONPATH=/mnt/workspace/ManiSkill
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/mnt/workspace/ManiSkill/.runtime/matplotlib
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json

# 默认选5条代表轨迹：0、2、54、80、99（数据不足100条时取前5条）。
.venv-ppu/bin/python -B visual/export_videos.py

# 指定轨迹与新的输出目录。
.venv-ppu/bin/python -B visual/export_videos.py --episodes 0 2 54 80 99 \
  --output .runtime/flow_dp3/visual-five

# 导出训练集全部100条；不会导出训练集以外的原始轨迹。
.venv-ppu/bin/python -B visual/export_videos.py --all \
  --output .runtime/flow_dp3/visual-all

# 仅生成全数据集结构与数值检查报告，不启动仿真。
.venv-ppu/bin/python -B visual/export_videos.py --inspect-only
```

默认输出 `.runtime/flow_dp3/<数据集名称>-visualization/`。
每条输出一个1800×1200 MP4，默认帧率为环境控制频率，包含 `T+1` 观测帧：

- 左侧：回放场景 RGB，以及原128×128点云相机的 RGB（最近邻放大）。
- 右上：HDF5 保存的全场景点云、末端附近点云，展示基座坐标轴、TCP 与目标。
- 右下：HDF5 保存的关节位置、关节速度、夹爪位置、TCP到目标距离及动作曲线，游标随帧移动。

视频点云面板采用 Matplotlib，便于在云实例现有环境运行；交互点云工具采用 Open3D。
RGB 是补录回放画面，点云与曲线直接来自训练 HDF5。不是策略预测/评估视频。

工具校验 `.raw.h5` 和 `.raw.json` 的来源哈希，按 `source_episode` 找到初始场景状态，
随后执行训练集保存的 `pd_ee_delta_pos` 动作。每帧重新生成观测并与 HDF5 比较。
默认非速度状态误差阈值 `1e-4`，关节速度 `1e-3`，点云四维特征 `1e-4`（策略特征单位）。
点云逐索引严格比较，采样顺序改变也算不一致。

`quality.json` 记录全部轨迹的基础检查、选中轨迹的回放误差、成功结果、视频帧数与解码验证。
发生不一致时，视频标记 `MISMATCH`，报告记录帧号，程序以非零退出码结束。
通过检查说明回放与已保存观测一致，不能单独证明数据多样性、物体点覆盖或训练效果。

所有输出拒绝覆盖。再次导出请通过 `--output` 选择新目录，点云导出请使用新文件名。
已有样例后直接执行默认命令可能因文件已存在而退出，这是预期行为。

## 4. 验证

```bash
.venv-ppu/bin/python -B -m unittest discover -s visual -p 'test_*.py' -v
```

测试覆盖长度尺度与坐标还原、相对视图中的基座位置、前景选点/空白点击/相机后方点，
以及文件拒绝覆盖和不合法轨迹拒绝加载。新增服务器导出测试实际生成并解码 PNG/MP4，
验证无 Open3D/桌面时可导出、包含终止观测帧、近处过滤无点时仍可渲染、相机和色标保持固定。
GUI 点击和 Open3D 离屏渲染需在完成安装后分别验证。

本实例于2026-10-06完成8项测试，以及100条训练轨迹的结构/有限值/距离通道检查。
已生成 episode 0、2、54、80、99 的5条同步视频，共408帧，均通过逐帧回放比较和 MP4 解码检查，
状态与点云特征最大误差均为0，5条回放均最终成功。
样例位于 `.runtime/flow_dp3/pickcube-100-scene-v3-visualization/`：
episode 0 位于根目录，其余4条位于 `samples/`，两处各有 `quality.json`，根目录另有 `inspection.json`。
这些本地生成文件不随 Git 上传。

服务器点云导出已在现有 `.venv-ppu` 中实际验证：
`pointcloud/cloud-full-frame30.png`、`pointcloud/cloud-near-frame0.png`、
`pointcloud/cloud-near-episode0.mp4` 均位于上述样例目录。
PNG 和 MP4 均为960×768，MP4 共75帧、20 FPS；全部帧已解码检查，
结果记录于 `pointcloud/export-validation.json`。这些导出无需安装 Open3D。

本次工具环境无法解析阿里云源和 PyPI 的域名，Open3D 下载未完成；同时没有桌面显示，
因此交互窗口与 Open3D 离屏渲染尚未进行实际运行验证。请在能联网的实例终端执行安装脚本，
在已配置桌面显示的环境验证选点。原训练环境没有被可视化安装脚本升级。
