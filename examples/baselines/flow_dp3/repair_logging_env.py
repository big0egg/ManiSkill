"""在项目 venv 内修复 W&B 0.30 引起的依赖覆盖，下载失败时不卸载包。"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
TARGET = {"wandb": "0.22.3", "protobuf": "3.20.3", "click": "8.1.7"}
# 只处理本次 W&B 0.30 安装的已确认版本，不卸载镜像系统包。
OVERRIDES = {
    "opentelemetry-api": "1.45.0",
    "opentelemetry-sdk": "1.45.0",
    "opentelemetry-proto": "1.45.0",
    "opentelemetry-semantic-conventions": "0.66b0",
    "opentelemetry-exporter-otlp-proto-common": "1.45.0",
    "opentelemetry-exporter-otlp-proto-http": "1.45.0",
    "opentelemetry-exporter-http-transport": "0.66b0",
    "opentelemetry-exporter-otlp-common": "0.66b0",
}


def run(command, env):
    print("执行：", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="仅检查并显示修复范围")
    parser.add_argument("--index-url", default="https://mirrors.aliyun.com/pypi/simple/")
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv-ppu").resolve() or sys.prefix == sys.base_prefix:
        raise RuntimeError("请先激活本项目 .venv-ppu，再用该环境的 python 执行")
    constraints = ROOT / ".runtime/constraints-ppu.txt"
    requirements = Path(__file__).with_name("requirements-logging.txt")
    protected = {}
    for line in constraints.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, version = line.strip().split("==", 1)
        actual = metadata.version(name)
        if actual != version:
            raise RuntimeError(f"受保护依赖 {name} 当前为 {actual}，约束要求 {version}；停止修复")
        protected[name] = version
    local_path = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    local = {d.metadata["Name"].lower().replace("_", "-"): d
             for d in metadata.distributions(path=[str(local_path)])}
    removals = []
    for name, expected in OVERRIDES.items():
        if name not in local:
            continue
        if local[name].version != expected:
            raise RuntimeError(f"{name} 的本地版本为 {local[name].version}，不是预期 {expected}；停止修复")
        removals.append(name)
    print("安装固定版本：", TARGET, flush=True)
    print("移除本项目 venv 中的 OpenTelemetry 覆盖包：", removals, flush=True)
    print("保留框架版本：", protected, flush=True)
    if args.dry_run:
        return
    report_dir = ROOT / ".runtime/wandb-dependency-repair"
    report_dir.mkdir(parents=True, exist_ok=True)
    wheels = Path(tempfile.mkdtemp(prefix="wheels-", dir=report_dir))
    before = [{"name": d.metadata.get("Name"), "version": d.version,
               "location": str(d.locate_file(""))} for d in metadata.distributions()]
    (wheels / "packages-before.json").write_text(json.dumps(before, indent=2) + "\n")
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                "PIP_PROXY", "PIP_EXTRA_INDEX_URL"):
        env.pop(key, None)
    env["PIP_CONFIG_FILE"] = os.devnull
    pip = [sys.executable, "-B", "-m", "pip"]
    # 所有 wheel 就绪后才修改已安装包，避免下载失败留下半卸载的环境。
    try:
        run(pip + ["download", "--only-binary=:all:", "--dest", wheels,
                   "--index-url", args.index_url, "--timeout", "20", "--retries", "1",
                   "--constraint", constraints, "--requirement", requirements], env)
    except subprocess.CalledProcessError:
        print("下载失败；已安装包未作修改。检查实例网络或改用 --index-url https://pypi.org/simple/。", flush=True)
        raise
    run(pip + ["install", "--no-index", "--find-links", wheels,
               "--constraint", constraints, "--requirement", requirements], env)
    if removals:
        run(pip + ["uninstall", "-y"] + removals, env)
    # 新进程检查真实 import 路径与版本，避免使用旧进程内的模块缓存。
    check = (
        "from importlib import metadata; import wandb, google.protobuf, click; "
        f"expected={TARGET | protected!r}; "
        "actual={name:metadata.version(name) for name in expected}; "
        "assert actual==expected, (actual, expected); "
        "assert wandb.Settings(x_disable_stats=True).x_disable_stats; "
        "print('固定版本和核心依赖检查通过:',actual); "
        "print('W&B:',wandb.__file__); print('protobuf:',google.protobuf.__file__)"
    )
    run([sys.executable, "-B", "-c", check], env)
    (report_dir / "last-success.json").write_text(json.dumps(
        {"versions": TARGET, "protected": protected, "removed_local_overrides": removals,
         "download_directory": str(wheels)}, ensure_ascii=False, indent=2) + "\n")
    print("依赖修复完成。镜像原有的依赖冲突可能仍出现在 pip check 中。", flush=True)


if __name__ == "__main__":
    main()
