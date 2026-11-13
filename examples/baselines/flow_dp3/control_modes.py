"""EE-only policy controller validation and single-stream dataset access."""
from __future__ import annotations

import json


EE_CONTROLLERS = ("pd_ee_delta_pose", "pd_ee_delta_pos")
CONTROL_CHOICES = ("ee",) + EE_CONTROLLERS


def control_matches(requested, actual):
    if actual not in EE_CONTROLLERS:
        raise ValueError(f"策略仅支持 EE 控制，实际控制器为 {actual}")
    if requested is not None and requested not in CONTROL_CHOICES:
        raise ValueError(f"未知 EE 控制方式：{requested}")
    if requested is None:
        return True
    if requested == "ee":
        return True
    return requested == actual


def check_control_mode(requested, contract):
    if not control_matches(requested, contract.get("control_mode")):
        raise ValueError(f"所选控制方式 {requested} 与数据/模型的 {contract['control_mode']} "
                         f"({contract['action_dim']}维) 不匹配；请选择对应分支或 checkpoint")


def select_dataset_group(stream, control_mode=None):
    """Accept only a root-level EE episode stream; retired layouts are rejected."""
    if "layout" in stream.attrs or "contract" not in stream.attrs:
        raise ValueError("需要新录制的单分支 EE 数据；旧双分支格式已停用")
    group = stream
    contract = json.loads(group.attrs["contract"])
    check_control_mode(control_mode, contract)
    return group, contract, json.loads(group.attrs.get("manifest", "{}"))
