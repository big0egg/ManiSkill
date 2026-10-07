"""Controller selection and read-only access to legacy/dual-control HDF5 files."""
from __future__ import annotations

import json


CONTROL_CHOICES = ("ee", "joint", "pd_ee_delta_pose", "pd_joint_pos", "pd_ee_delta_pos")
DUAL_LAYOUT = "dual_control_v1"


def control_matches(requested, actual):
    if requested is None:
        return True
    if requested == "ee":
        return actual in ("pd_ee_delta_pos", "pd_ee_delta_pose")
    if requested == "joint":
        return actual == "pd_joint_pos"
    if requested not in CONTROL_CHOICES:
        raise ValueError(f"未知控制方式：{requested}")
    return requested == actual


def check_control_mode(requested, contract):
    if requested is None:
        return
    if not control_matches(requested, contract["control_mode"]):
        raise ValueError(f"所选控制方式 {requested} 与数据/模型的 {contract['control_mode']} "
                         f"({contract['action_dim']}维) 不匹配；请选择对应分支或 checkpoint")


def select_dataset_group(stream, control_mode=None):
    """Return the actual episode group, its contract and its branch manifest."""
    if stream.attrs.get("layout") == DUAL_LAYOUT:
        key = "joint" if control_mode in ("joint", "pd_joint_pos") else "ee"
        if key not in stream:
            raise ValueError(f"双控制数据缺少 {key} 分支")
        group = stream[key]
    else:
        group = stream
    contract = json.loads(group.attrs["contract"])
    check_control_mode(control_mode, contract)
    return group, contract, json.loads(group.attrs.get("manifest", "{}"))
