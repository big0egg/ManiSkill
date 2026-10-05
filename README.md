# ManiSkill 3：阿里云 PPU 环境与 Flow DP3 接入验证

![仿真环境与机器人示例](figures/teaser.jpg)

ManiSkill 是基于 [SAPIEN](https://sapien.ucsd.edu/) 的开源机器人仿真与学习框架，支持桌面操作、移动操作、灵巧操作，以及强化学习、模仿学习和视觉语言动作模型。上游提供 GPU 并行物理仿真、视觉数据采集、任务构建接口和 sim2real 示例。论文见 [ManiSkill3（RSS 2025）](https://arxiv.org/abs/2410.00425)，完整功能见 [官方文档](https://maniskill.readthedocs.io/en/latest/)。

本仓库按阿里云 **`2.1.0-pytorch2.9.0-ppu-py312-cu130-ubuntu24.04`** 镜像整理操作流程，已独立接入 **Flow DP3（`flow_dp3`）**，主点云编码器为 **`ee_relation_pointnetpp`**，首个验证任务为 **`PickCube-v1` / Panda**。

已实现独立 FlowDP3、统一观测适配、成功示范回放、序列数据集、训练、EMA checkpoint 保存/恢复与闭环评估。当前可行性路线为 **CPU PhysX + CPU 软件 Vulkan 渲染 + PPU 策略计算**。第8节提供完整操作命令。

> **验证记录（2026-10-05）**：CPU 仿真和软件 Vulkan 的真实点云已通过；用户实例报告确认 PPU-ZW810E 的基础计算及关系编码器前向/反向通过。已准备5条烟雾示范及20条训练示范，末端控制回放均成功；CPU 小模型完成20次训练更新、checkpoint 恢复及闭环执行。用户实例的 PPU 小模型训练与评估也已通过；完整宽度主干训练仍待验证。短训练的闭环成功率不能作为正式性能结果，见第8.6节。

## 1. 环境与适配原则

| 项目 | 当前环境 / 选择 |
| --- | --- |
| 平台 / 镜像 | 阿里云 PAI DSW / `2.1.0-pytorch2.9.0-ppu-py312-cu130-ubuntu24.04` |
| 系统 / Python | Ubuntu 24.04.2 / `/usr/local/bin/python` 3.12.3 |
| PyTorch / torchvision / torchaudio | 2.9.0 / 0.24.0 / 2.9.0，继承镜像 |
| PyTorch CUDA 构建版本 | 13.0，属于 PPU 镜像的 CUDA 兼容接口 |
| NumPy / SciPy | 1.26.0 / 1.11.3，安装时保护当前版本 |
| SDK | `/usr/local/PPU_SDK`，`release.yaml` 为 `2.1.0-a5f865` |
| 首批物理仿真 | `physx_cpu`，单环境 |
| 任务 / 控制模式 | `PickCube-v1` / `panda` / `pd_ee_delta_pos` |
| 编码器 | 独立 `EERelationPointNetPPEncoder`，纯 PyTorch |

**保留镜像预装的 PyTorch、torchvision、torchaudio、PPU SDK 和驱动。** 不执行上游通用的 `pip install --upgrade mani_skill torch`，不安装 NVIDIA 驱动或普通 CUDA Toolkit 覆盖 PPU 环境。框架生态版本可参考 [阿里云 PPU v2.1 软件清单](https://help.aliyun.com/zh/document_detail/3030377.html)。

需要分别验证三条链路：

- **模型计算**：CPU / PPU 是否能完成关系编码器前向和反向。
- **物理仿真**：SAPIEN / PhysX 是否能创建环境并执行动作；第一阶段使用 CPU。
- **点云渲染**：相机能否获取有效视觉观测，再生成距离点云。CPU 仿真成功不能代替渲染验证。

阿里云 [PPU v2.0 图像说明](https://help.aliyun.com/en/document_detail/3012066.html) 明确说明 PPU 不支持 OpenGL/Vulkan，目前没有依据宣称本镜像的 v2.1 已支持这些图形接口。ManiSkill 的渲染依赖 Vulkan，官方主要支持 Linux/NVIDIA。[官方安装说明](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)

PPU 镜像中的 `cuda:0` 和 `nvidia-smi` 是兼容接口，不能证明设备具有 NVIDIA 图形能力。MuJoCo 的 OSMesa/EGL 设置也不能直接解决 SAPIEN 的 Vulkan 渲染问题。

## 2. 目录与模型输入

```text
ManiSkill/
├── README.md
├── mani_skill/                    # 原仿真框架
├── examples/baselines/flow_dp3/
│   ├── ee_relation_encoder.py     # 独立关系点云编码器
│   ├── probe_runtime.py           # 环境、编码器与真实点云探测
│   ├── obs_adapter.py             # 统一物理点云和本体/目标状态契约
│   ├── prepare_demos.py           # 离线生成、动作转换、成功回放过滤
│   ├── dataset.py                 # episode 划分与序列窗口
│   ├── policy.py                  # FlowDP3、一致性目标与 EMA
│   ├── conditional_unet1d.py      # 独立 FiLM U-Net 及本地组件
│   ├── train.py / evaluate.py     # 训练、恢复、闭环评估与 MP4 录制
│   ├── experiment_logging.py      # 可选 W&B 在线/离线日志
│   ├── import_training_logs.py    # 将已有 JSONL 指标导入 W&B
│   ├── configs/                  # 完整模型与小模型烟雾配置
│   ├── tests/                    # 时序、统计隔离和坐标契约验证
│   ├── PROVENANCE.md / LICENSE-DP3
│   └── requirements-*.txt         # 探测/训练新增依赖
├── .venv-ppu/                     # 本项目虚拟环境，运行时生成
└── .runtime/                      # 安装约束、报告、资产与数据，运行时生成
```

编码器与 U-Net 是用户参考算法的独立副本，策略和接口在本目录独立实现；不导入其他 benchmark，不设置其他项目路径，不共用环境、数据或 checkpoint。具体移植范围与来源见 [PROVENANCE.md](examples/baselines/flow_dp3/PROVENANCE.md)。保留两级 FPS / 球邻域、末端关系特征和双路径读出，输出64维特征。

模型单帧输入为 `[B,N,4]`，默认 `N=512`：

```text
前三通道 = (点的基座坐标 − 当前末端的基座坐标) / L
第四通道 = 相对位移的欧氏距离 / L
```

坐标单位为米、坐标轴为固定基座轴、`L` 为固定长度尺度。不重复减末端、不做每帧中心化、不对四通道分别 min-max 标准化。ManiSkill 原生 `xyzw` 第四通道是有效点标记，必须重新构造距离通道。[官方观测格式](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html)

探测依次过滤无效/非有限点、转基座坐标、工作空间裁剪、预采样和 FPS，再减当前末端。点不足时明确报错，不填零冒充点云。裁剪范围可通过 `--crop-min` / `--crop-max` 调整。

默认 `--radius1 0.10 --radius2 0.20` 是以米计的**烟雾验证半径**，不是已调优训练参数。正式训练应根据新数据的邻域距离统计选半径；程序将物理半径与输入一起除以 `L`。

## 3. 创建继承镜像框架的环境

以下命令在能访问 PPU 的 **DSW 实例终端** 执行。重新打开终端后，重新设置变量、初始化 SDK 并激活环境。

```bash
export MANISKILL_ROOT=/mnt/workspace/ManiSkill
cd "$MANISKILL_ROOT"
source /usr/local/PPU_SDK/envsetup.sh

/usr/local/bin/python -m venv --system-site-packages "$MANISKILL_ROOT/.venv-ppu"
source "$MANISKILL_ROOT/.venv-ppu/bin/activate"

export PYTHONPATH="$MANISKILL_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MS_ASSET_DIR="$MANISKILL_ROOT/.runtime/maniskill"
mkdir -p "$MANISKILL_ROOT/.runtime"
```

`--system-site-packages` 继承镜像 PPU 框架，新增依赖安装到虚拟环境。已有 `.venv-ppu` 可直接激活。源码导入路径只使用当前仓库。`.venv-ppu/`、`.runtime/` 是本机运行文件，不应作为代码提交或复制进算法发布包。

生成约束，保护预装框架及 NumPy / SciPy：

```bash
python -B - <<'PY'
import importlib.metadata as metadata
import os
from pathlib import Path

names = ['torch', 'torchvision', 'torchaudio', 'numpy', 'scipy']
pins = [f'{name}=={metadata.version(name)}' for name in names]
path = Path(os.environ['MANISKILL_ROOT']) / '.runtime/constraints-ppu.txt'
path.write_text('\n'.join(pins) + '\n', encoding='utf-8')
print(path)
print('\n'.join(pins))
PY

export PIP_CONSTRAINT="$MANISKILL_ROOT/.runtime/constraints-ppu.txt"
```

不套用旧 Python 3.8 项目的 Gym、SAPIEN、pip 或其他依赖约束。

## 4. 安装依赖与当前源码

### 4.1 补齐 OpenCV 系统运行库

当前源码在 `import mani_skill` 时会注册任务，并通过数字孪生任务模块执行 `import cv2`。因此即使只运行 `PickCube-v1`，也会触发 OpenCV 导入。实例终端已出现：

```text
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

这是 OpenCV 的系统动态库缺失，不表示 ManiSkill 的 pip 安装失败，也不是策略模型错误。Ubuntu 24.04 的 `libgl1` 包提供 `libGL.so.1`。[Ubuntu 包说明](https://packages.ubuntu.com/noble/amd64/libgl1)

在具有系统包安装权限的实例终端执行：

```bash
sudo apt-get update
sudo apt-get install -y libgl1
```

如果当前用户为 root，去掉 `sudo`。这一步安装系统运行库，不安装 NVIDIA 驱动，不替换 PPU SDK，也不证明 Vulkan 点云渲染已经可用。当前环境已能完成导入并通过第6节 CPU 任务探测；新实例仍应按本节检查系统库。

### 4.2 安装 Python 依赖与当前仓库

定义普通 Python 依赖的安装函数。它只对当前命令绕过代理和 pip 配置，保留约束；不修改全局配置，不用于重装 PPU 框架。

```bash
pip_install_public() {
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        -u http_proxy -u https_proxy -u all_proxy \
        -u PIP_PROXY -u PIP_EXTRA_INDEX_URL \
        PIP_CONFIG_FILE=/dev/null \
        PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
        python -m pip install \
        --index-url https://mirrors.aliyun.com/pypi/simple/ \
        --constraint "$MANISKILL_ROOT/.runtime/constraints-ppu.txt" \
        --timeout 20 --retries 1 "$@"
}

pip_install_public -r examples/baselines/flow_dp3/requirements-probe.txt
```

第一阶段固定 `sapien==3.0.3`、`gymnasium==0.29.1`、`mplib==0.1.1`、`pytorch_kinematics==0.7.6`，其余依赖与当前 `setup.py` 的运行要求对应。[SAPIEN 3.0.3](https://pypi.org/project/sapien/3.0.3/) 和 [MPlib 0.1.1](https://pypi.org/project/mplib/0.1.1/) 都有 Python 3.12/Linux wheel；这不代表 PPU 仿真或渲染已兼容。

依赖安装成功后，再安装当前仓库：

```bash
pip_install_public --no-deps --no-build-isolation -e .

python -B - <<'PY'
import importlib.metadata as metadata
import cv2
print('OpenCV:', cv2.__version__, cv2.__file__)
import mani_skill
import torch
print('ManiSkill 源码:', mani_skill.__file__)
for name in ['torch', 'torchvision', 'torchaudio', 'numpy', 'scipy',
             'sapien', 'gymnasium', 'mplib', 'pytorch_kinematics']:
    print(name, metadata.version(name))
print('PyTorch CUDA 构建:', torch.version.cuda)
PY

python -m pip freeze --local > .runtime/packages-local.txt
```

`--no-deps` 用在依赖已安装之后，避免源码安装再次调整框架；`--no-build-isolation` 使用镜像现有构建工具。本镜像 setuptools 80.10.2 满足仓库声明的 ≥62.3.0。

首次工具安装在请求 `dacite` 时因 `Name or service not known` 失败；随后用户在实例终端已经完成 ManiSkill 源码安装。DNS、代理或连接失败后的 `No matching distribution found` 不代表包不存在。先确认实例能访问安装源，再分析版本冲突。若仅某个版本未被公共镜像同步，可在有网络的终端将该次安装的 `--index-url` 改为 `https://pypi.org/simple/`，仍保留约束。

### 4.3 安装成功后的导入排查

`Successfully installed mani_skill-3.0.1` 表示安装完成，不代表源码导入、物理仿真或点云渲染已经通过。若出现此前的 `libGL.so.1` 报错，先按第4.1节安装 `libgl1`，再在已激活的本项目虚拟环境里单独验证：

```bash
python -B - <<'PY'
import cv2
print('OpenCV:', cv2.__version__, cv2.__file__)
import mani_skill
print('ManiSkill:', mani_skill.__file__)
PY
```

导入通过后再执行第6节 CPU 仿真探测；若还有异常，应根据新的 traceback 定位，不将所有警告当作同一个问题。

| 消息 | 原因与处理 |
| --- | --- |
| `ImportError: libGL.so.1` | 此前导致导入中断的错误；先补 `libgl1`，再验证 `cv2` 和 ManiSkill 导入。 |
| `RequestsDependencyWarning` | 当前 Requests 不接受 `chardet 6.x`；日志同时列出 urllib3 / charset-normalizer，不代表它们全部不兼容。这是警告，不是此次中断原因。 |
| `Failed to find Vulkan ICD file` / `Failed to find glvnd ICD file` | 当前 SAPIEN 根据 PPU 镜像的 `nvidia-smi` 兼容命令尝试寻找 NVIDIA 图形配置；不要据此安装 NVIDIA 驱动。点云渲染按第7节单独验证。 |

如果需要消除当前 Requests 的 chardet 版本警告，可使用第4.2节的函数，仅在本项目虚拟环境安装兼容版本：

```bash
pip_install_public 'chardet>=5,<6'
```

该调整尚未验证；不需要为修复 `libGL.so.1` 同时升级 Requests、urllib3 或 PyTorch。

只读检查发现：虚拟环境安装了 `opencv-python==4.11.0.86`，基础镜像另有 `opencv-python-headless==4.13.0.92`，实际 `cv2` 优先来自虚拟环境的普通版。两种发行包共享 `cv2` 命名空间，官方建议只选择一种。[OpenCV 安装说明](https://pypi.org/project/opencv-python/)

当前保留普通版，导入已通过。后续若改为无 GUI 的 headless 方案，应只处理本项目虚拟环境，保护基础镜像；当前继承的 headless 4.13 在 Python 3.12 下声明需要 NumPy ≥2，与本项目保护的 NumPy 1.26.0 不一致，不能直接依赖它或无约束升级。

## 5. 验证 CPU 与 PPU 模型计算

每项检查运行在独立子进程，有超时；一项崩溃不会阻止其他项目返回结果。任何所选项目失败时退出码为1，全部所选项目通过时为0。探测不安装软件、不自动下载资产、不修改驱动、不训练策略。

### 5.1 CPU 编码器：不需要仿真依赖

```bash
python -B examples/baselines/flow_dp3/probe_runtime.py \
    --checks metadata torch-cpu encoder-cpu \
    --output .runtime/probe-cpu.json
```

当前会话这三项已通过。`encoder-cpu` 使用 `[2,512,4]` 合成距离点云，验证 `[2,64]` 输出、输入梯度及全部参数梯度有限，并检查梯度不全为零。该结果不包含真实相机观测或完整 FlowDP3 U-Net。

### 5.2 PPU 编码器：在实例终端执行

```bash
ppu-smi

python -B examples/baselines/flow_dp3/probe_runtime.py \
    --checks torch-ppu encoder-ppu \
    --output .runtime/probe-ppu.json
```

`encoder-ppu` 使用 `cuda:0` 验证内部 FPS、排序、球查询、关系读出及反向。矩阵乘法通过不能代替完整编码器算子验证。用户实例报告 `.runtime/probe-ppu.json` 已确认两项通过，设备为 PPU-ZW810E；编码器输入 `[2,512,4]`、输出 `[2,64]`，梯度有限且不全为零。该结果不包括完整 FlowDP3 策略。

若返回“当前进程无法访问 PPU”，检查实例终端 `ppu-smi`、SDK 初始化和设备挂载。早期工具进程曾返回 `torch.cuda.is_available()==False`，但用户实例终端随后已通过上述探测；应区分工具进程的设备可见性与实例硬件状态，不直接覆盖驱动或重装 torch。

## 6. 验证 CPU 物理仿真

安装完第4节依赖后执行：

```bash
python -B examples/baselines/flow_dp3/probe_runtime.py \
    --checks sim --steps 3 \
    --output .runtime/probe-sim.json
```

使用 `PickCube-v1`、Panda、`physx_cpu`、`render_backend=none`、`state_dict` 和 `pd_ee_delta_pos`，检查 reset、4维动作、3次 step 和关节状态有限性。Panda 使用仓库内置资产；URDF 或关联资产缺失时先恢复完整仓库。

当前修复版本已通过该项验证：环境创建、reset、4维动作和3次 step 正常，报告保存在 `.runtime/probe-sim-fixed.json`。重新执行上面的命令会将当前结果写入 `.runtime/probe-sim.json`。随机动作不是专家示范，该项不验证任务成功率、相机或点云。

### 6.1 关闭渲染时仍创建材质的兼容修复

修复前即使设置 `render_backend=none`，`PickCube-v1` 创建方块仍会在 `mani_skill/utils/building/actors/common.py` 中执行 `sapien.render.RenderMaterial(...)`，报 `RuntimeError: failed to find a rendering device`。创建目标球也有同类问题。这是物体视觉初始化路径未完整遵循关闭渲染设置，不表示 CPU PhysX 无法运行。

本仓库已给 `build_cube()` 和 `build_sphere()` 增加 `scene.can_render()` 判断：无渲染模式只创建所需物理结构，跳过视觉形状及材质；启用渲染时仍执行原视觉创建逻辑。修复仅覆盖这两个构造函数，其他任务使用不同视觉构造路径时应单独验证。

该修复使 CPU 无渲染任务探测通过。相机点云采用第7节已通过的 Mesa 软件 Vulkan 路线。

## 7. 验证点云渲染

### 7.1 检查 Vulkan

```bash
command -v vulkaninfo
ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null
```

当前实例已安装 Mesa 软件 Vulkan，实际使用 `/usr/share/vulkan/icd.d/lvp_icd.json`，点云报告已通过。新实例仍需检查实际 ICD 路径；loader 存在不代表渲染设备可用。

如实例允许安装系统包，可由有权限的终端安装软件 Vulkan 探测工具：

```bash
sudo apt-get update
sudo apt-get install -y libvulkan1 vulkan-tools mesa-vulkan-drivers
vulkaninfo --summary
```

当前实例已通过 Mesa CPU 软件 Vulkan 路线。它使用 CPU 完成渲染，吞吐量需要单独测量；不能据此宣称 PPU 本身支持 Vulkan。无显示器时使用离屏相机即可，不需要 GUI。

### 7.2 尝试 CPU 软件渲染

SAPIEN 某些版本根据 `nvidia-smi` 是否存在自动选择 NVIDIA ICD；PPU 镜像也有这个兼容命令。因此应在探测命令中显式选择实际的 Mesa 软件 ICD。[上游相关问题](https://github.com/haosulab/SAPIEN/issues/269)

先找到文件：

```bash
find /usr/share/vulkan/icd.d -maxdepth 1 -name '*lvp*.json' -print
```

常见名称为 `lvp_icd.x86_64.json` 或 `lvp_icd.json`。以下变量按实际输出修改；文件不存在时先处理软件 Vulkan 安装，不要生成虚假的 NVIDIA ICD：

```bash
export MANISKILL_VULKAN_ICD=/usr/share/vulkan/icd.d/lvp_icd.json
test -f "$MANISKILL_VULKAN_ICD"

VK_ICD_FILENAMES="$MANISKILL_VULKAN_ICD" vulkaninfo --summary

VK_ICD_FILENAMES="$MANISKILL_VULKAN_ICD" \
python -B examples/baselines/flow_dp3/probe_runtime.py \
    --checks pointcloud --render-backend cpu --steps 3 --timeout 120 \
    --output .runtime/probe-pointcloud.json
```

`pointcloud` 检查 reset 和后续3次 step 的每帧有效点云，转换为基座坐标下 `[1,512,4]` 距离特征，再验证编码器 `[1,64]` 输出。报告包含有效点数、裁剪点数、末端位置、距离范围和距离模长误差。

默认裁剪是首批探测设置，点不足可显式调整 `--crop-min` / `--crop-max`。以后离线数据和在线评估必须采用相同的相机、变换、裁剪及采样契约。

当前 `.runtime/probe-pointcloud.json` 已通过：每帧有16384个有效像素点，裁剪后约1.1万个点，输出 `[1,512,4]` 和 `[1,64]`。第8节的采集与评估继续使用该软件 ICD。

### 7.3 兼容图形设备上的备选路线

在具有 Vulkan 的兼容 GPU 实例上，可保持 CPU 物理后端并使用 GPU 渲染：

```bash
python -B examples/baselines/flow_dp3/probe_runtime.py \
    --checks pointcloud --render-backend gpu \
    --output .runtime/probe-pointcloud-gpu.json
```

当前 `.runtime/probe-pointcloud-gpu.json` 的 GPU 渲染失败，错误为 `ErrorIncompatibleDriver`。本实例继续使用已通过的软件渲染。兼容图形实例可作为后续提速方案，迁移时需对齐框架、观测与 checkpoint 契约。

## 8. FlowDP3 数据、训练与闭环评估

### 8.1 初始化第二阶段运行环境

已经执行第3～7节时，继续激活同一个虚拟环境并设置软件 Vulkan。下面所有命令从本仓库根目录运行；新终端先重新执行这些初始化命令。

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
export MS_ASSET_DIR="$MANISKILL_ROOT/.runtime/maniskill"
export MANISKILL_VULKAN_ICD=/usr/share/vulkan/icd.d/lvp_icd.json
export VK_ICD_FILENAMES="$MANISKILL_VULKAN_ICD"
export MPLCONFIGDIR="$MANISKILL_ROOT/.runtime/matplotlib"
mkdir -p "$MPLCONFIGDIR" "$MANISKILL_ROOT/.runtime/flow_dp3"
test -f "$VK_ICD_FILENAMES"
```

按第4.2节定义 `pip_install_public` 后，安装唯一新增训练依赖：

```bash
pip_install_public -r examples/baselines/flow_dp3/requirements-train.txt
```

当前已能导入 `einops 0.8.1`，已有兼容版本可直接使用。训练入口默认 `cuda:0`，不可用时明确退出；CPU 验证需要显式指定 `--device cpu`。

### 8.2 获取并转换成功示范

**当前工作区已生成可用的小样本**：`.runtime/flow_dp3/pickcube-smoke.h5`，包含5条成功末端控制示范，可直接用于第8.3节。该文件来自仓库自带运动规划器，并非下载的公开数据集。

新实例或需要重新生成时，使用新文件名：

```bash
python -B examples/baselines/flow_dp3/prepare_demos.py \
    --generate 6 --count 5 \
    --output .runtime/flow_dp3/pickcube-smoke-new.h5
```

程序先在无渲染 CPU 仿真中生成成功 `pd_joint_pos` 原始示范，保存同名 `*.raw.h5` / `*.raw.json`；再在软件渲染环境真实执行动作转换，过滤最终失败的回放，保存紧凑训练数据与 `.json` 清单。生成有 `--max-attempts` 上限；回放过程中不逐帧强设环境状态。成功数量不足会保留已有结果并报错，便于检查和补充。

如使用公开示范，可在有网络的实例终端下载，保留原始 `.h5` 及同名 `.json`：

```bash
python -m mani_skill.utils.download_demo PickCube-v1 \
    --output_dir "$MANISKILL_ROOT/.runtime/demos"
rg --files .runtime/demos -g '*.h5' -g '*.json'

python -B examples/baselines/flow_dp3/prepare_demos.py \
    --source .runtime/demos/PickCube-v1/motionplanning/trajectory.h5 \
    --count 5 --output .runtime/flow_dp3/pickcube-public-smoke.h5
```

原始轨迹路径以实际下载结果为准。第一阶段仅支持 `PickCube-v1` / Panda 和 `pd_joint_pos` → `pd_ee_delta_pos` 转换，或已有 `pd_ee_delta_pos` 回放。输出文件已存在时拒绝覆盖；需要新数据时换一个文件名。

数据契约：

| 字段 | 形状 / 含义 |
| --- | --- |
| `pointcloud_distance` | `[T+1,512,4]`，第2节的基座轴相对末端物理特征 |
| `state` | `[T+1,28]`：9维 qpos、9维 qvel、7维基座系末端位姿（四元数 wxyz）、3维基座系目标 |
| `action` | `[T,4]`：标准化末端 xyz 增量与夹爪动作，范围 `[-1,1]` |
| 文件属性 / 清单 | 输入版本、尺度、裁剪、采样、后端、源文件 SHA256、成功状态及回放长度 |

保留仓库 Panda 默认 `base_camera`（128×128），相机、控制器和点云处理都由同一个 `make_env` / 适配器创建。目标位置是任务条件；不向策略输入 `obj_pose`、`is_grasped` 或 `success`。按整条 episode 划分训练/验证，状态和动作的 limits 统计仅拟合训练 episode，点云保持物理契约。序列边缘重复填充，执行从预测索引 `n_obs_steps-1` 开始，避免动作错一帧。

### 8.3 先验证 PPU 小模型训练

```bash
python -B examples/baselines/flow_dp3/train.py \
    --config examples/baselines/flow_dp3/configs/pickcube_smoke.yaml \
    --data .runtime/flow_dp3/pickcube-smoke.h5 \
    --output .runtime/flow_dp3/ppu-smoke \
    --device cuda:0
```

小模型保持 `ee_relation_pointnetpp` 编码器、2帧观测、16步预测和8步执行，将 U-Net 通道缩小为 `[64,128,256]`（约554万参数），训练20次更新。它检查真实数据、完整网络前向/反向、AdamW、验证和 checkpoint；不能用于宣称完整模型性能。

输出目录包含 `config.yaml`、`run.json`、`metrics.jsonl`、`last.pt` 与按验证损失选择的 `best.pt`。checkpoint 保存模型、EMA、normalizer、优化器、调度器、数据指纹、观测契约与随机状态。`best.pt` 按验证损失选择，不等于任务成功率最高的模型。

检查 `metrics.jsonl` 的 `train_loss` / `val_loss` / `grad_norm` 应为有限值。`run.json` 应显示 `cuda:0`，同时列出互不重叠的训练/验证 episode。若显存不足，可显式减小 `--batch-size`；程序不会自动改变主干宽度。

如果当前终端无法访问 PPU，可做 CPU 验证；使用新的输出目录：

```bash
python -B examples/baselines/flow_dp3/train.py \
    --config examples/baselines/flow_dp3/configs/pickcube_smoke.yaml \
    --data .runtime/flow_dp3/pickcube-smoke.h5 \
    --output .runtime/flow_dp3/cpu-local-smoke --device cpu
```

### 8.4 加载 checkpoint 做闭环评估

```bash
python -B examples/baselines/flow_dp3/evaluate.py \
    --checkpoint .runtime/flow_dp3/ppu-smoke/last.pt \
    --device cuda:0 --episodes 2 --start-seed 1000 --max-steps 200 \
    --output .runtime/flow_dp3/eval-ppu-smoke.json
```

环境物理与点云渲染使用 CPU，策略使用 PPU。每次 reset 清空动作缓存，首帧重复构造观测历史；随后按8步动作块执行，逐步更新历史。默认使用 EMA 权重；`--raw-weights` 可选择训练权重。对非有限预测直接报错，对超出控制器范围的预测裁剪，并记录裁剪比例。

评估设置 `reconfiguration_freq=1`，忽略瞬时成功产生的 terminated，执行到所选步数上限，报告 `success_once_rate` / `success_end_rate`，并记录预处理、推理以及仿真/渲染耗时。默认200步是本次末端控制可行性实验的上限，正式横向比较需统一任务、控制模式、评估种子与步数预算，不能把这个结果直接当作其他论文的标准成绩。

同一报告路径拒绝覆盖。CPU checkpoint 可以用 `--device cpu` 评估，也可加载到 PPU；精确**恢复训练**则要求保持计算设备类型、配置、数据指纹和 batch size 一致。

### 8.5 增加数据，再进入完整主干训练

PPU 小模型训练与评估都通过后，再用20条成功示范进行完整主干训练。**当前工作区 `.runtime/flow_dp3/pickcube-20.h5` 已准备好**；新实例数据缺失时生成，给失败回放留出原始示范余量：

```bash
if [ ! -f .runtime/flow_dp3/pickcube-20.h5 ]; then
    python -B examples/baselines/flow_dp3/prepare_demos.py \
        --generate 30 --count 20 \
        --output .runtime/flow_dp3/pickcube-20.h5
fi

python -B examples/baselines/flow_dp3/train.py \
    --config examples/baselines/flow_dp3/configs/pickcube.yaml \
    --data .runtime/flow_dp3/pickcube-20.h5 \
    --output .runtime/flow_dp3/ppu-full --device cuda:0 \
    --batch-size 4 --steps 1000
```

完整配置保留参考主干 `[512,1024,2048]`，AdamW 学习率 `1e-4`、500步 warmup、cosine 调度及分段一致性参数。`--steps` 是**目标总更新次数**，按步数随机抽取训练窗口；1000步是首轮验证预算，不是收敛标准。调度总长度由配置的 `training.steps=30000` 决定，提前停止不改变调度；增大训练预算时应先明确实验配置。

继续同一次训练：

```bash
python -B examples/baselines/flow_dp3/train.py \
    --config examples/baselines/flow_dp3/configs/pickcube.yaml \
    --data .runtime/flow_dp3/pickcube-20.h5 \
    --output .runtime/flow_dp3/ppu-full --device cuda:0 \
    --batch-size 4 --steps 30000 \
    --resume .runtime/flow_dp3/ppu-full/last.pt
```

新训练的输出目录必须为空；恢复写回原目录，并严格检查数据/配置。改变模型、数据、半径或训练设置时新建实验目录。`.10/.20` 米是本次初始半径候选，正式实验需要邻域统计和消融，不视为已调优参数。

训练后使用同一评估入口，将 checkpoint 换为 `ppu-full/last.pt`，增加独立种子的 episodes；先确认成功率，再决定其他任务与基线实验。当前已实现的是独立新任务训练，旧任务 checkpoint 的状态/动作维度不同，不能直接加载。

### 8.6 当前实测与验收范围

| 检查 | 实测结果 / 报告 |
| --- | --- |
| CPU 仿真、真实点云、共享适配器 | 已通过，`.runtime/flow_dp3/probe-shared-adapter.json` |
| PPU 编码器 | 用户实例已通过，`.runtime/probe-ppu.json` |
| 成功示范转换 | 小样本5/5；另已生成20条成功回放用于后续训练，见 `pickcube-smoke.json` / `pickcube-20.json` |
| CPU 小模型训练 | 5条/20条示范各完成20次更新，验证损失约 `1.30e-4` / `1.27e-4`；`cpu-verified/metrics.jsonl` / `cpu-small-20demos/metrics.jsonl` |
| 原目标一致性 | 同一输入/随机种子下损失一致，梯度最大误差约 `5.8e-11`；`objective-parity.json` |
| checkpoint 恢复 | 10+10步与连续20步的模型、EMA、采样器/框架随机状态一致；`resume-equivalence.json` |
| CPU 小模型闭环 | 5条/20条数据训练的模型各评估2条独立种子、每条200步；两组成功率均0/2，见 `eval-cpu-verified.json` / `eval-cpu-20demos.json` |
| PPU 小模型训练/评估 | 用户实例报告已通过20次更新和2局200步评估，见 `ppu-smoke/metrics.jsonl` / `eval-ppu-smoke.json` |
| 完整宽度主干 PPU 训练 | 尚未验证，按8.5节开展；当前工具进程不可访问 PPU |

表中未写完整路径的文件位于 `.runtime/flow_dp3/`，总验收摘要为 `acceptance.json`。这些结果确认移植的数据与执行链路可运行，尚不证明原模型在 ManiSkill 上的学习效果。完整宽度模型性能及吞吐量需要后续实测。

可重复执行数据契约检查：

```bash
python -B -m unittest discover -s examples/baselines/flow_dp3/tests -v
```

源代码运行时只依赖当前仓库和本环境；参考项目未修改，也不需要部署。算法来源与许可证保留在 [PROVENANCE.md](examples/baselines/flow_dp3/PROVENANCE.md) 和 [LICENSE-DP3](examples/baselines/flow_dp3/LICENSE-DP3)。

### 8.7 W&B 可视化与推理视频

训练默认继续保存本地 `metrics.jsonl`，每步都有指标；终端每10步打印一次。W&B 是可选功能，默认 `--wandb-mode disabled`，录制默认关闭。

#### 安装与登录

先按8.1节初始化环境，在本项目 venv 中使用4.2节的安装函数：

```bash
pip_install_public -r examples/baselines/flow_dp3/requirements-logging.txt
wandb login
```

如果当前终端还没有定义 `pip_install_public`，先重新执行4.2节的函数定义。W&B 登录由实例终端完成，不将 API key 写入训练配置。SDK 依赖仍受 `.runtime/constraints-ppu.txt` 保护。[W&B SDK](https://github.com/wandb/wandb)

#### 直接查看已有训练曲线，无需重新训练

现有 `.runtime/flow_dp3/ppu-smoke/` 包含20次更新的完整指标，可直接导入：

```bash
python -B examples/baselines/flow_dp3/import_training_logs.py \
    --run-dir .runtime/flow_dp3/ppu-smoke \
    --wandb-mode online --wandb-project flow_dp3 \
    --wandb-name ppu-smoke-history
```

终端会打印 W&B 页面链接，网页中查看 `train/loss`、`val/loss`、`train/lr`、`train/grad_norm` 和 `train/ema_decay`，横轴为 `global_step`。导入目录默认为训练目录内的 `wandb-history/`；重复导入需指定新的 `--output` 目录。历史导入不会上传 `.pt` 权重或示范数据。

#### 新训练实时显示曲线

```bash
python -B examples/baselines/flow_dp3/train.py \
    --config examples/baselines/flow_dp3/configs/pickcube_smoke.yaml \
    --data .runtime/flow_dp3/pickcube-smoke.h5 \
    --output .runtime/flow_dp3/ppu-wandb-smoke --device cuda:0 \
    --wandb-mode online --wandb-project flow_dp3 \
    --wandb-name ppu-wandb-smoke --wandb-log-every 1
```

完整模型训练同样可以添加这些参数。默认每10步向 W&B 记录一次，首步、验证和末步始终记录；本地 JSONL 始终保留每步指标。`--wandb-entity` 可以指定个人/团队 workspace。默认关闭 W&B 自动硬件监控，在 PPU 环境只记录程序明确给出的实验指标。

W&B run ID 与 project/entity 保存到 `wandb_run.json` 和 checkpoint。恢复模型时继续使用原数据、配置、batch size、输出目录和 `--resume`，加上 `--wandb-mode online`，会自动续接同一个 W&B run。旧 checkpoint 也可加载；其未包含 W&B 信息时新建日志 run。绘图使用明确的训练 `global_step`，与 W&B 内部记录序号分开。[W&B SDK 恢复行为](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py)

#### 阿里云网络受限时离线记录

将上面的 `--wandb-mode online` 改为 `--wandb-mode offline`。离线模式无需登录即可保存 SDK 日志，但不能实时显示云端曲线。网络恢复后登录并同步实际目录：

```bash
find .runtime/flow_dp3/ppu-wandb-smoke/wandb \
    -maxdepth 1 -type d -name 'offline-run-*' -print

wandb sync .runtime/flow_dp3/ppu-wandb-smoke/wandb/offline-run-实际目录名
```

历史导入也支持 offline；对应 SDK 日志位于 `ppu-smoke/wandb-history/wandb/`。离线恢复训练继续沿用逻辑 run ID，但每次执行会产生一个新的离线目录，需逐个同步。这里不承诺 SDK 原地恢复同一个离线文件。

#### 录制现有 checkpoint 的推理 MP4

不需要 W&B 就可以录制：

```bash
python -B examples/baselines/flow_dp3/evaluate.py \
    --checkpoint .runtime/flow_dp3/ppu-smoke/last.pt \
    --device cuda:0 --episodes 2 --start-seed 1000 --max-steps 200 \
    --save-video --video-dir .runtime/flow_dp3/videos/ppu-smoke \
    --output .runtime/flow_dp3/eval-ppu-smoke-video.json

ls -lh .runtime/flow_dp3/videos/ppu-smoke/*.mp4
```

每局分别保存 `seed_1000.mp4`、`seed_1001.mp4`；JSON 的每局结果包含 `video_path`。默认 FPS 使用任务控制频率（本任务为20），可用 `--video-fps` 改变播放帧率。未指定 `--video-dir` 时，自动使用报告旁的 `<报告名>-videos/`。已有同名视频或报告时拒绝覆盖，重新实验使用新路径。

录像复用 ManiSkill `RecordEpisode`，只开启展示用 RGB 相机，训练观测相机、点云契约和动作执行流程保持一致。录制会增加 CPU 渲染与视频编码时间；评估报告记录这些耗时。

当前已经有两段可看的完整预览，来自现有 PPU 训练 checkpoint，在 CPU 上执行200步并录制：

- `.runtime/flow_dp3/videos/ppu-smoke-preview/seed_1000.mp4`
- `.runtime/flow_dp3/videos/ppu-smoke-preview/seed_1001.mp4`

已解码验证每段201帧、512×512、20 FPS，约10.05秒；画面包含 Panda、方块和目标，结果仍为失败演示，不能当作成功策略视频。报告为 `.runtime/flow_dp3/eval-ppu-smoke-preview.json`。可以在阿里云 DSW 文件浏览器中打开/下载 MP4；`.runtime` 是隐藏目录，需要允许显示隐藏文件。视频、SDK 日志与权重受 `.gitignore` 排除，保留在服务器上。

#### 同时在 W&B 查看评估结果和视频

```bash
python -B examples/baselines/flow_dp3/evaluate.py \
    --checkpoint .runtime/flow_dp3/ppu-smoke/last.pt \
    --device cuda:0 --episodes 2 --max-steps 200 \
    --save-video --video-dir .runtime/flow_dp3/videos/ppu-wandb-eval \
    --output .runtime/flow_dp3/eval-ppu-wandb-video.json \
    --wandb-mode online --wandb-project flow_dp3 \
    --wandb-name ppu-smoke-eval --wandb-upload-videos
```

评估单独创建 W&B run，记录每局成功状态、种子、耗时和动作裁剪比例，summary 中显示 `eval/success_once_rate` / `eval/success_end_rate`。只有显式添加 `--wandb-upload-videos` 时，才将 MP4 同时记录到 W&B；否则仅保留本地视频。评估日志位于报告旁的 `<报告名>-logging/`，W&B 页面链接也写入报告。

本次验证：CPU 训练恢复及启用日志的 RNG 隔离通过；历史指标导入和视频记录接口通过 mock SDK 检查；真实 MP4 已生成并解码检查。当前工具未能获取 W&B 安装包，真实 SDK 离线检查暂时跳过，在线登录与同步需要在实例终端验证。安装 SDK 后可执行下列测试，其中 `test_real_sdk_offline` 会使用真实 SDK 创建离线日志：

```bash
python -B -m unittest discover -s examples/baselines/flow_dp3/tests -v
```

## 9. 上游资源与平台支持

- [快速开始](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html)
- [示例脚本](https://maniskill.readthedocs.io/en/latest/user_guide/demos/index.html)
- [学习 baselines](examples/baselines/README.md)
- [Colab 入门](https://colab.research.google.com/github/mani-skill/ManiSkill/blob/main/examples/tutorials/1_quickstart.ipynb)
- [问题反馈](https://github.com/mani-skill/ManiSkill/issues/) / [讨论](https://github.com/mani-skill/ManiSkill/discussions/) / [Discord](https://discord.gg/x8yUZe5AdN)
- [原 ManiSkill2 v0.5.3](https://github.com/mani-skill/ManiSkill/tree/v0.5.3)

以下为上游平台支持说明，不能当作本 PPU 镜像的实测结果：

| 系统 / GPU | CPU 仿真 | GPU 仿真 | 渲染 |
| --- | --- | --- | --- |
| Linux / NVIDIA | 支持 | 支持 | 支持 |
| Windows / NVIDIA | 支持 | 不支持 | 支持 |
| Windows / AMD | 支持 | 不支持 | 支持 |
| WSL | 支持 | 不支持 | 不支持 |
| macOS | 支持 | 不支持 | 支持 |
| 本 PPU 镜像 | PickCube CPU 通过 | 未验证 | CPU 软件 Vulkan 通过，PPU GPU 渲染失败 |

## 10. 引用与许可证

使用 ManiSkill3（`mani_skill>=3.0.0`）时，请引用：

```bibtex
@article{taomaniskill3,
  title={ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI},
  author={Stone Tao and Fanbo Xiang and Arth Shukla and Yuzhe Qin and Xander Hinrichsen and Xiaodi Yuan and Chen Bao and Xinsong Lin and Yulin Liu and Tse-kai Chan and Yuan Gao and Xuanlin Li and Tongzhou Mu and Nan Xiao and Arnav Gurha and Viswesh Nagaswamy Rajesh and Yong Woo Choi and Yen-Ru Chen and Zhiao Huang and Roberto Calandra and Rui Chen and Shan Luo and Hao Su},
  journal={Robotics: Science and Systems},
  year={2025}
}
```

使用 ManiSkill2（`mani_skill==0.5.3` 或更早版本）时，请引用：

```bibtex
@inproceedings{gu2023maniskill2,
  title={ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills},
  author={Gu, Jiayuan and Xiang, Fanbo and Li, Xuanlin and Ling, Zhan and Liu, Xiqiang and Mu, Tongzhou and Tang, Yihe and Tao, Stone and Wei, Xinyue and Yao, Yunchao and Yuan, Xiaodi and Xie, Pengwei and Huang, Zhiao and Chen, Rui and Su, Hao},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

使用具体算法和第三方资产时，还应保留来源和对应引用。FlowDP3 基于用户修改版 DP3，主编码器为 `ee_relation_pointnetpp`。DP3 原方法见 [论文](https://arxiv.org/abs/2403.03954) 与 [项目](https://3d-diffusion-policy.github.io/)；新增实现与参考差异见本目录的来源说明。

ManiSkill 刚体环境采用宽松许可证，例如 [Apache-2.0](LICENSE)。资产遵循 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode)，第三方组件参见 [LICENSE-3RD-PARTY](LICENSE-3RD-PARTY)。
