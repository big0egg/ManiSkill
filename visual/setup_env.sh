#!/usr/bin/env bash
# 在 .runtime 中创建可视化环境，复用本地 ROCm 主环境的框架。
set -euo pipefail
unset PYTHONPATH LD_LIBRARY_PATH
visual_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
visual_python="$visual_root/.runtime/env-rocm/bin/python"
visual_env="$visual_root/.runtime/visual-venv"
visual_constraints="$visual_root/.runtime/constraints-rocm.txt"
test -x "$visual_python"
test -f "$visual_constraints"
"$visual_python" -B -m venv --system-site-packages "$visual_env"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    -u PIP_PROXY -u PIP_EXTRA_INDEX_URL \
    PIP_CONFIG_FILE=/dev/null \
    "$visual_env/bin/python" -B -m pip install \
    --index-url "${VISUAL_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}" \
    --constraint "$visual_constraints" \
    --timeout 120 --retries 3 --requirement "$visual_root/visual/requirements.txt"
"$visual_env/bin/python" -B -c 'import open3d, numpy, h5py; print("Open3D:", open3d.__version__, "NumPy:", numpy.__version__, "h5py:", h5py.__version__)'
