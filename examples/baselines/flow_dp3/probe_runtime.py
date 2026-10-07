"""隔离检查 PPU 计算、CPU 仿真与点云渲染；不训练策略。

每项检查运行在独立子进程，驱动异常、崩溃或超时不会中断其余检查。
不安装软件、不自动下载资产、不修改驱动；--output 显式保存 JSON 报告。
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

CHECKS = ("metadata", "torch-cpu", "torch-ppu", "encoder-cpu", "encoder-ppu",
          "sim", "pointcloud")
RESULT_PREFIX = "FLOW_DP3_PROBE_RESULT="


def check_metadata():
    packages = {}
    for name in ("torch", "torchvision", "torchaudio", "numpy", "scipy", "sapien",
                 "mani_skill", "gymnasium", "h5py", "mplib", "pytorch_kinematics"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    icds = []
    for directory in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"):
        icds.extend(str(p) for p in sorted(Path(directory).glob("*.json")))
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": packages,
            "vulkan_icds": icds, "VK_ICD_FILENAMES": os.getenv("VK_ICD_FILENAMES"),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES")}


def get_device(use_ppu):
    import torch
    if use_ppu and not torch.cuda.is_available():
        raise RuntimeError("当前进程无法访问 PPU 的 CUDA 接口；请在可访问 PPU 的实例终端重试。")
    return torch.device("cuda:0" if use_ppu else "cpu")


def check_torch(use_ppu):
    import torch
    device = get_device(use_ppu)
    torch.manual_seed(42)
    x = torch.randn(16, 16, device=device, requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    if use_ppu:
        torch.cuda.synchronize(device)
    if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
        raise RuntimeError("矩阵计算或反向传播出现 NaN/Inf")
    return {"device": str(device), "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device) if use_ppu else "CPU",
            "loss": float(loss.detach().cpu()), "gradient_norm": float(x.grad.norm().cpu())}


def check_encoder(args, use_ppu):
    import torch
    from ee_relation_encoder import EERelationPointNetPPEncoder
    device = get_device(use_ppu)
    torch.manual_seed(42)
    # 半径仅用于合成数据烟雾验证；不作为 ManiSkill 任务训练半径。
    relative = torch.rand(2, args.num_points, 3, device=device) * 0.4 - 0.2
    x = torch.cat((relative, relative.norm(dim=-1, keepdim=True)), dim=-1)
    x = (x / args.length_scale).detach().requires_grad_(True)
    model = EERelationPointNetPPEncoder(
        radius1=args.radius1 / args.length_scale,
        radius2=args.radius2 / args.length_scale, debug_checks=True).to(device)
    output = model(x)
    if output.shape != (2, 64) or not torch.isfinite(output).all():
        raise RuntimeError("编码器输出形状错误或包含 NaN/Inf")
    output.square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all() or x.grad.norm() == 0:
        raise RuntimeError("编码器输入梯度缺失、非有限或为零")
    grads = [p.grad for p in model.parameters()]
    if any(g is None or not torch.isfinite(g).all() for g in grads):
        raise RuntimeError("编码器参数梯度缺失或包含 NaN/Inf")
    if not any(g.norm() > 0 for g in grads):
        raise RuntimeError("编码器参数梯度全部为零")
    if use_ppu:
        torch.cuda.synchronize(device)
    return {"device": str(device), "input_shape": list(x.shape),
            "output_shape": list(output.shape), "input_gradient_norm": float(x.grad.norm().cpu()),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "input_frame": "base", "input_mode": "physical",
            "length_scale": args.length_scale,
            "synthetic_radius_m": [args.radius1, args.radius2]}


def check_env(args, visual):
    import gymnasium as gym
    import torch
    import mani_skill.envs  # 注册当前独立仓库的任务
    from ee_relation_encoder import EERelationPointNetPPEncoder
    from obs_adapter import ObservationConfig, pointcloud_features  # 仅 worker 导入
    from task_registry import get_task
    from mani_skill.utils.task_pointcloud import pointcloud_sensor_configs
    from mani_skill import PACKAGE_ASSET_DIR
    # Panda 使用仓库内置资产；资产缺失时立即报错，不触发交互下载。
    urdf = PACKAGE_ASSET_DIR / "robots/panda/panda_v2.urdf"
    if not urdf.is_file():
        raise FileNotFoundError(f"Panda 内置资产缺失：{urdf}")
    config = (ObservationConfig(
        num_points=args.num_points, length_scale=args.length_scale,
        crop_min=tuple(args.crop_min) if args.crop_min is not None else ObservationConfig.crop_min,
        crop_max=tuple(args.crop_max) if args.crop_max is not None else ObservationConfig.crop_max,
    ) if visual else None)
    task = get_task("PickCube-v1")
    env = gym.make("PickCube-v1", robot_uids=task.robot, num_envs=1,
                   obs_mode="pointcloud" if visual else "state_dict",
                   control_mode=task.control_mode, sim_backend="physx_cpu",
                   render_backend=args.render_backend if visual else "none",
                   sensor_configs=pointcloud_sensor_configs(), reconfiguration_freq=1,
                   render_mode=None)
    try:
        env.action_space.seed(42)
        obs, _ = env.reset(seed=42)
        action_shape = list(env.action_space.shape)
        if action_shape[-1] != task.action_dim:
            raise RuntimeError(f"{task.control_mode} 动作应为{task.action_dim}维，实际 {action_shape}")
        frames = []
        model = None
        if visual:
            model = EERelationPointNetPPEncoder(
                radius1=args.radius1 / args.length_scale,
                radius2=args.radius2 / args.length_scale, debug_checks=True).eval()
        # 检查 reset 观测，以及每个环境 step 后的新观测。
        for step in range(args.steps + 1):
            if visual:
                features, frame = pointcloud_features(obs, env.unwrapped.agent, config)
                with torch.no_grad():
                    encoded = model(features)
                if encoded.shape != (1, 64) or not torch.isfinite(encoded).all():
                    raise RuntimeError("真实点云编码结果异常")
                frame["encoder_output_shape"] = list(encoded.shape)
                frames.append(frame)
            else:
                qpos = torch.as_tensor(obs["agent"]["qpos"])
                if not torch.isfinite(qpos).all():
                    raise RuntimeError("机器人关节状态包含 NaN/Inf")
            if step < args.steps:
                obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
                if torch.as_tensor(terminated).any() or torch.as_tensor(truncated).any():
                    raise RuntimeError("随机探测在指定步数之前结束")
        result = {"env_id": "PickCube-v1", "control_mode": task.control_mode, "sim_backend": "physx_cpu",
                  "render_backend": args.render_backend if visual else "none",
                  "steps": args.steps, "action_shape": action_shape,
                  "task_goal_available": "goal_pos" in obs.get("extra", {})}
        if visual:
            result["frames"] = frames
            result["sensor_configs"] = pointcloud_sensor_configs()
            result["crop_min"] = config.crop_min
            result["crop_max"] = config.crop_max
        return result
    finally:
        env.close()


def execute_check(name, args):
    if name == "metadata":
        return check_metadata()
    if name.startswith("torch-"):
        return check_torch(name.endswith("ppu"))
    if name.startswith("encoder-"):
        return check_encoder(args, name.endswith("ppu"))
    return check_env(args, visual=name == "pointcloud")


def worker(name, args):
    started = time.monotonic()
    try:
        detail = execute_check(name, args)
        result = {"check": name, "status": "pass", "detail": detail}
    except Exception as exc:
        result = {"check": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()}
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "pass" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", nargs="+", choices=CHECKS, default=list(CHECKS))
    parser.add_argument("--render-backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90, help="每项检查的子进程超时秒数")
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--radius1", type=float, default=0.10, help="烟雾验证半径，单位米")
    parser.add_argument("--radius2", type=float, default=0.20, help="烟雾验证半径，单位米")
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--crop-min", type=float, nargs=3,
                        help="基座系 xyz 下界（米），默认目标优先的 PickCube 操作区")
    parser.add_argument("--crop-max", type=float, nargs=3,
                        help="基座系 xyz 上界（米），默认目标优先的 PickCube 操作区")
    parser.add_argument("--output", type=Path, help="可选 JSON 报告路径")
    parser.add_argument("--worker", choices=CHECKS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.steps <= 40 or args.num_points < 128:
        parser.error("steps 必须为1~40，num-points 必须至少128")
    if any(not math.isfinite(x) or x <= 0 for x in
           (args.timeout, args.radius1, args.radius2, args.length_scale)):
        parser.error("timeout、半径和 length-scale 必须是有限正数")
    if any(not math.isfinite(x) for x in (args.crop_min or []) + (args.crop_max or [])) or (
            args.crop_min is not None and args.crop_max is not None and
            any(a >= b for a, b in zip(args.crop_min, args.crop_max))):
        parser.error("裁剪边界必须有限，且每个 min 都小于对应 max")
    if args.worker:
        return worker(args.worker, args)

    results = []
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    for name in dict.fromkeys(args.checks):
        print(f"检查 {name} …", flush=True)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", name,
                   "--render-backend", args.render_backend, "--steps", str(args.steps),
                   "--num-points", str(args.num_points), "--radius1", str(args.radius1),
                   "--radius2", str(args.radius2), "--length-scale", str(args.length_scale)]
        if args.crop_min is not None:
            command += ["--crop-min", *map(str, args.crop_min)]
        if args.crop_max is not None:
            command += ["--crop-max", *map(str, args.crop_max)]
        started = time.monotonic()
        try:
            completed = subprocess.run(command, capture_output=True, text=True, env=env,
                                       timeout=args.timeout)
            records = [line[len(RESULT_PREFIX):] for line in completed.stdout.splitlines()
                       if line.startswith(RESULT_PREFIX)]
            result = json.loads(records[-1]) if records else {
                "check": name, "status": "fail", "error": "子进程退出但未返回检查结果"}
            result["returncode"] = completed.returncode
            if completed.returncode != 0:
                result["status"] = "fail"
            result["stdout_tail"] = completed.stdout[-4000:]
            result["stderr_tail"] = completed.stderr[-4000:]
        except subprocess.TimeoutExpired as exc:
            def tail(value):
                return value.decode(errors="replace")[-4000:] if isinstance(value, bytes) else (value or "")[-4000:]
            result = {"check": name, "status": "fail", "error": f"子进程超过 {args.timeout} 秒",
                      "stdout_tail": tail(exc.stdout), "stderr_tail": tail(exc.stderr)}
        result["wall_seconds"] = round(time.monotonic() - started, 3)
        results.append(result)
        print(f"  {result['status'].upper()}: {result.get('error', '通过')}", flush=True)
    report = {"scope": "第一阶段环境与编码器探测；未训练或评估完整 FlowDP3 策略",
              "all_selected_checks_passed": all(r["status"] == "pass" for r in results),
              "results": results}
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"报告: {args.output}", flush=True)
    else:
        print(encoded)
    return 0 if report["all_selected_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
