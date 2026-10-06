# Flow DP3 数据观测可视化

所有工具只读输入 HDF5，不修改训练代码、数据或 checkpoint。默认数据为
`.runtime/flow_dp3/pickcube-100-roi-v4.h5`，可通过 `--dataset` 指定其他文件。
新默认数据在基座系 crop `min=(-0.2,-0.3,-0.05)`、`max=(1.1,0.3,1.0)` 后重新执行 FPS512，
保留机械臂、方块和工作区桌面，移除地面；512点包含桌面点。旧文件仍可显式指定。
从项目根目录执行下面的命令。

## 0. 查看帮助与快速使用

两个 Python 工具都支持 `--help`，也可以写成 `-h`。帮助显示后立即退出，不读取数据、不启动窗口或仿真：

```bash
cd /mnt/workspace/ManiSkill
.venv-ppu/bin/python -B visual/pointcloud.py --help
.venv-ppu/bin/python -B visual/export_videos.py --help
```

`--help` 中的 `{both,fps,distance}` 表示只能从这三个字符串中选一个；例如写
`--export-view fps`，不要填写花括号。`PNG`、`DATASET` 等大写词是填写位置的提示，
需要替换成实际文件路径。完整取值、默认值和示例见下方参数表。

阿里云服务器上直接生成左右对照图和视频：

```bash
.venv-ppu/bin/python -B visual/pointcloud.py \
  --dataset .runtime/flow_dp3/pickcube-100-scene-v3.h5 \
  --episode 0 --frame 0 \
  --export-backend matplotlib --export-view both \
  --fps 20 \
  --png .runtime/flow_dp3/my-cloud-comparison.png \
  --video .runtime/flow_dp3/my-cloud-comparison.mp4
```

PNG 保存第0帧，MP4 保存第0帧到该轨迹末帧。只需要图片时删除 `--video` 及其路径；
只需要视频时删除 `--png` 及其路径。所有输出都拒绝覆盖，重复执行需换新文件名。

参数的填写规则：

- 开关参数只写名称，例如 `--pick-points`、`--play`、`--all`；不要在后面填 `True`、`False` 或 `1`。不写表示关闭。
- 数值参数后填一个数字，例如 `--frame 30`、`--near-radius 1.0`；米、FPS 等单位不用写进命令。
- 路径参数支持绝对路径或相对当前工作目录的路径；带空格的路径用引号包住。输入文件必须存在，输出父目录会自动创建。
- `pointcloud.py` 的 `--episode` 一次填一条；`export_videos.py` 的 `--episodes` 可用空格分隔填写多条，例如 `--episodes 0 2 54`。
- 参数属于各自脚本，不能混用；例如 `--export-view` 仅用于 `pointcloud.py`，`--all` 仅用于 `export_videos.py`。

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

### `pointcloud.py` 完整参数

除 `--help` 外，参数均可省略；但服务器没有桌面时须至少指定 `--png`、`--video`、`--ply` 中的一项。
不指定这三项会打开 Open3D 交互窗口。当前数据集有100条轨迹，索引为 `0` 到 `99`。

| 参数 | 可填内容 / 范围 | 默认值（不写参数时） | 作用与填写示例 |
| --- | --- | --- | --- |
| `-h` / `--help` | 开关，不填值 | 不显示帮助 | 显示用法并退出：`--help` |
| `--dataset` | 已存在的训练 `.h5` 文件路径 | 项目内 `.runtime/flow_dp3/pickcube-100-roi-v4.h5` | 指定输入：`--dataset /mnt/workspace/ManiSkill/.runtime/flow_dp3/pickcube-100-roi-v4.h5` |
| `--episode` | 一个非负整数索引，或 HDF5 中已有的完整轨迹名称 | `0` | 索引按名称排序，从0开始；当前可填 `0`～`99`，如 `--episode 54` 或 `--episode episode_00054` |
| `--frame` | 整数，`0`～当前轨迹的动作数 `T`，含两端 | `0` | GUI 初始帧、PNG/PLY 保存帧、MP4 起始帧；如 `--frame 30`。当前 episode 0 可填 `0`～`74`，其他轨迹长度不同 |
| `--coordinate-frame` | 只能填 `base` 或 `relative` | `base` | `base`：机器人基座坐标系；`relative`：原点平移到当前 TCP，方向仍为基座轴。例：`--coordinate-frame relative` |
| `--near-radius` | 有限正数，单位米，例如 `0.5`、`1`、`2.0`；不可填 `0` 或负数 | 不过滤，显示全部点 | 只显示距 TCP 不超过指定半径的点，两图同步过滤；例：`--near-radius 1`。想看全部点就省略此参数 |
| `--vector-count` | 任意非负整数，例如 `0`、`24`、`128`、`512` | `24` | `0` 关闭矢量线；实际数量不超过当前显示点数。当前完整点云有512点，`--vector-count 512` 可画全部；FPS 单图不画矢量 |
| `--point-size` | 有限正数，例如 `3`、`6`、`8.5` | `6` | 点的显示大小；数值越大点越明显。例：`--point-size 8` |
| `--color-max` | 有限正数，单位米，例如 `1`、`5`、`20` | 有近处过滤时取 `near-radius`，否则取整条轨迹最大距离 | 固定距离色标上限，超过上限颜色饱和，不移除点；例：`--color-max 1`。FPS 单图不用距离色标 |
| `--fps` | 有限正数，例如 `10`、`20`、`30` | `20` | GUI 播放及导出 MP4 的每秒帧数；只改播放速度，不重新采样观测。例：`--fps 20` |
| `--play` | 开关，不填值 | 关闭 | GUI 启动后自动播放：`--play`。离屏导出不使用此开关 |
| `--pick-points` | 开关，不填值 | 关闭 | GUI 单击点查看坐标、距离和策略特征：`--pick-points`。需要桌面，不可与 `--png`、`--video`、`--ply` 同用 |
| `--export-backend` | 只能填 `matplotlib` 或 `open3d` | `matplotlib` | PNG/MP4 渲染后端；`matplotlib` 适合无桌面的服务器；`open3d` 需安装 Open3D 并提供 EGL/Mesa。例：`--export-backend matplotlib`。GUI/PLY 始终使用 Open3D |
| `--export-view` | 只能填 `both`、`fps` 或 `distance` | `both` | `both`：左 FPS 点云、右距离矢量，1920×768；`fps`：仅 FPS XYZ 点云，960×768；`distance`：原距离矢量单图，960×768。例：`--export-view fps`。仅影响 Matplotlib PNG/MP4 |
| `--png` | 一个新的 `.png` 文件路径，不能是目录 | 不导出图片 | 保存 `--frame` 指定的观测帧：`--png .runtime/flow_dp3/my-frame.png` |
| `--video` | 一个新的 `.mp4` 文件路径，不能是目录 | 不导出视频 | 保存从 `--frame` 到末帧 `T` 的所有观测：`--video .runtime/flow_dp3/my-cloud.mp4` |
| `--ply` | 一个新的 `.ply` 文件路径，不能是目录 | 不导出 PLY | 保存 `--frame` 指定帧的点坐标与距离颜色；需 Open3D，不包含基座坐标轴：`--ply .runtime/flow_dp3/my-cloud.ply` |

`--png`、`--video`、`--ply` 可组合使用，但输出路径必须互不相同。
`--export-view fps` 中的 `fps` 指最远点采样（Farthest Point Sampling）；
`--fps 20` 中的 FPS 指视频每秒帧数，两者含义不同。

整条轨迹使用固定初始相机范围、固定色标，切帧不重新缩放。

### 云实例无桌面时

普通 GUI 和点击选点需要本地桌面、远程桌面或已配置的 X11 显示。
没有 `DISPLAY`/`WAYLAND_DISPLAY` 的终端会给出提示，不能直接弹出交互窗口。
服务器默认导出使用 Matplotlib Agg 和 ImageIO/FFmpeg，CPU 渲染，无需 `DISPLAY`、Open3D、EGL、Vulkan 或 PPU。
直接从训练 HDF5 生成点云，不需要原始 `.raw.h5`，也不启动仿真。

```bash
# 指定帧的全场景左右对照 PNG；无需额外安装依赖。
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

# 分别导出 FPS 后点云与距离矢量单图。
.venv-ppu/bin/python -B visual/pointcloud.py --episode 0 --frame 30 \
  --export-view fps --png .runtime/flow_dp3/cloud-fps-frame30.png
.venv-ppu/bin/python -B visual/pointcloud.py --episode 0 --frame 30 \
  --export-view distance --png .runtime/flow_dp3/cloud-distance-frame30.png

# 安装 Open3D 后仍可使用原有 EGL 离屏渲染。
.runtime/visual-venv/bin/python -B visual/pointcloud.py --episode 0 \
  --export-backend open3d --png .runtime/flow_dp3/cloud-open3d-frame0.png \
  --ply .runtime/flow_dp3/cloud-frame0.ply
```

Matplotlib PNG/MP4 默认1920×768，左右各显示一部分：左图以统一颜色展示 FPS 后的 XYZ 点云形状，
右图展示同一批点的距离颜色与从 TCP 到点的矢量箭头，距离色标以米为单位。
两图使用相同点索引、帧、坐标系、观察视角和坐标范围，均包含基座坐标轴、末端和目标标记。
保存的特征为 `[relative_xyz, norm] / length_scale`，左图通过相对位移和 TCP 位置还原采样后点坐标；
不重新执行 FPS。训练 HDF5 未保存点颜色，因此左图使用统一颜色，不能还原相机 RGB。
默认采用基座坐标系；`--coordinate-frame relative` 会将两图原点同时平移到 TCP，轴方向不旋转。
右图箭头表示恢复到米的相对位移，颜色表示其长度；默认只画24条箭头，距离着色覆盖全部显示点。
`--export-view fps` 或 `--export-view distance` 导出960×768单图，后者保留原有距离矢量展示。
视频保存从 `--frame` 到末帧 `T` 的所有观测，保持相机范围与色标固定。
`--near-radius` 仅过滤显示点，矢量线从当前 TCP 到实际数据点；`--vector-count 0` 可关闭箭头。

离屏导出不能与 `--pick-points` 同时使用。PLY 只包含当前显示点及距离颜色，
不包含基座坐标轴；需要完整交互界面与原始四维特征时，应在桌面直接读取 HDF5。
PNG/MP4 的后端选择不影响原有 Open3D GUI/选点功能或 PLY 导出。
`--export-view` 仅作用于 Matplotlib PNG/MP4；Open3D 仍使用原有距离矢量单视图。
选择 Open3D 后端时使用 EGL，SAPIEN 场景视频使用 Vulkan；两种渲染环境需要分别验证。

## 3. 场景、相机、点云和状态同步视频

### `export_videos.py` 完整参数

本工具生成 RGB 场景、原点云相机、点云和状态曲线组成的同步视频；
只需要点云 PNG/MP4 时使用上一节的 `pointcloud.py`。

| 参数 | 可填内容 / 范围 | 默认值（不写参数时） | 作用与填写示例 |
| --- | --- | --- | --- |
| `-h` / `--help` | 开关，不填值 | 不显示帮助 | 显示用法并退出：`--help` |
| `--dataset` | 已存在的训练 `.h5` 文件路径 | 项目内 `.runtime/flow_dp3/pickcube-100-roi-v4.h5` | 指定训练集：`--dataset .runtime/flow_dp3/pickcube-100-roi-v4.h5` |
| `--episodes` | 一个或多个非负整数索引或完整轨迹名称，用空格分隔，不能重复 | 有至少100条时选 `0 2 54 80 99`，否则取前5条或全部不足5条的轨迹 | 如 `--episodes 0 2 54` 或 `--episodes episode_00000 episode_00002`；当前整数索引范围 `0`～`99` |
| `--all` | 开关，不填值 | 关闭 | 导出训练 HDF5 中全部轨迹：`--all`。与 `--episodes` 互斥 |
| `--output` | 输出目录路径，不是 `.mp4` 文件路径 | 输入数据所在目录下的 `<数据集文件名去掉.h5>-visualization/` | 如 `--output .runtime/flow_dp3/my-scene-videos`，每条轨迹的 MP4 和报告写入此目录；建议选择新目录 |
| `--raw-source` | 已存在的原始 `.raw.h5` 文件路径 | 优先使用训练集同目录同名 `.raw.h5`，不存在时用数据集 manifest 记录的来源路径 | 原始数据搬迁后指定：`--raw-source /mnt/data/pickcube-100-scene-v3.raw.h5`。同目录还需对应 `.raw.json`，两者哈希须匹配 |
| `--inspect-only` | 开关，不填值 | 关闭，执行回放并录像 | 仅检查完整训练集并写入 `inspection.json`：`--inspect-only`。不启动仿真，也不需要原始轨迹；不会生成 MP4，此模式不使用轨迹选择参数 |
| `--near-radius` | 有限正数，单位米，例如 `0.5`、`1`、`2` | `1.0` | 近处点云面板的过滤半径，也用于报告的近处覆盖统计：`--near-radius 0.5`；全场景面板仍显示所有点 |
| `--fps` | 有限正数，例如 `10`、`20`、`30` | 环境控制频率 | MP4 播放帧率，只改变播放速度，仍保存所有观测：`--fps 20` |
| `--state-tolerance` | 有限正数，例如 `1e-4`、`0.0001` | `1e-4` | 除关节速度外的状态最大绝对误差容许值；各字段沿用原单位，位置为米、关节角为弧度等。例：`--state-tolerance 1e-4` |
| `--velocity-tolerance` | 有限正数，例如 `1e-3`、`0.001` | `1e-3` | `qvel` 最大绝对误差容许值，沿用关节速度原单位。例：`--velocity-tolerance 1e-3` |
| `--pointcloud-tolerance` | 有限正数，例如 `1e-4`、`0.0001` | `1e-4` | 四维点云特征最大绝对误差容许值，单位为缩放后的策略特征，不直接等于米；逐点索引比较。例：`--pointcloud-tolerance 1e-4` |

不要同时填写 `--episodes` 和 `--all`。误差阈值控制回放一致性判定，一般保留默认值。
检查模式生成 `inspection.json`；录像模式生成 `<episode名称>.mp4` 和 `quality.json`。
目录可以已存在，但将要写入的文件必须不存在，工具不会覆盖旧文件。

### 场景视频使用示例

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
.venv-ppu/bin/python -B visual/export_videos.py --inspect-only \
  --output .runtime/flow_dp3/visual-inspection
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
双图测试还验证相同点坐标、同步近处过滤、相同投影，以及只在右图显示距离颜色和矢量。
GUI 点击和 Open3D 离屏渲染需在完成安装后分别验证。

新 ROI v4 数据已完成100条成功回放、7820帧检查，每帧为512点，全部落在新 crop 内；
状态和动作与 scene-v3 数据完全一致。第0条第0帧按实体 ID 统计为机械臂121点、方块4点、
桌面387点、地面0点。新双图位于 `.runtime/flow_dp3/pickcube-100-roi-v4-frame0.png`，
整份数据验证报告位于 `.runtime/flow_dp3/pickcube-100-roi-v4-validation.json`。
五个任务的点云接口另通过2402帧对照，数据契约5项和可视化10项测试均通过。

以下视频样例使用旧 scene-v3 数据。该数据于2026-10-06完成10项测试，以及100条训练轨迹的结构/有限值/距离通道检查。
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

双图导出样例位于同一目录：`pointcloud/cloud-comparison-full-frame30.png` 展示完整512点，
`pointcloud/cloud-comparison-near-frame0.png` 展示 TCP 附近1米的点；
`pointcloud/cloud-comparison-full-episode0.mp4` 和 `pointcloud/cloud-comparison-near-episode0.mp4`
分别展示完整点云与近处点云的左右对照。双图均为1920×768，视频各75帧、20 FPS，
解码检查记录在 `pointcloud/comparison-validation.json`。

本次工具环境无法解析阿里云源和 PyPI 的域名，Open3D 下载未完成；同时没有桌面显示，
因此交互窗口与 Open3D 离屏渲染尚未进行实际运行验证。请在能联网的实例终端执行安装脚本，
在已配置桌面显示的环境验证选点。原训练环境没有被可视化安装脚本升级。
