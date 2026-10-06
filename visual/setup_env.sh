#!/usr/bin/env bash
# 在 .runtime 中创建可视化环境；不向 .venv-ppu 或系统环境安装包。
set -euo pipefail
visual_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
visual_python="$visual_root/.venv-ppu/bin/python"
visual_env="$visual_root/.runtime/visual-venv"
test -x "$visual_python"
test -f "$visual_root/.runtime/constraints-ppu.txt"
"$visual_python" -B -m venv --system-site-packages "$visual_env"
"$visual_env/bin/python" -B - "$visual_root" <<'PY'
import sys
import sysconfig
from pathlib import Path
root = Path(sys.argv[1])
ppu_packages = root / '.venv-ppu' / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
Path(sysconfig.get_path('purelib'), 'maniskill_ppu_packages.pth').write_text(str(ppu_packages) + '\n')
PY
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    -u PIP_PROXY -u PIP_EXTRA_INDEX_URL \
    PIP_CONFIG_FILE=/dev/null \
    "$visual_env/bin/python" -B -m pip install \
    --index-url "${VISUAL_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}" \
    --constraint "$visual_root/.runtime/constraints-ppu.txt" \
    'protobuf==3.20.3' 'click==8.1.7' \
    --timeout 20 --retries 1 --requirement "$visual_root/visual/requirements.txt"
"$visual_env/bin/python" -B -c 'import open3d, numpy, h5py; print("Open3D:", open3d.__version__, "NumPy:", numpy.__version__, "h5py:", h5py.__version__)'
