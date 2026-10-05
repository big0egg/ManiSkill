"""将已有 metrics.jsonl 导入 W&B，无需重新训练或上传模型权重。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from experiment_logging import ExperimentLogger, add_wandb_args, require_wandb, preserve_rng


def import_logs(args):
    if args.wandb_mode == "disabled":
        raise ValueError("历史导入需要 --wandb-mode online 或 offline")
    directory = args.run_dir.resolve()
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if not records or any(not isinstance(row.get("step"), int) or row["step"] < 1 for row in records):
        raise ValueError("历史日志为空或 step 无效")
    if any(a["step"] >= b["step"] for a, b in zip(records, records[1:])):
        raise ValueError("历史日志 step 必须严格递增；请先检查恢复训练的记录")
    if any(not math.isfinite(value) for row in records for value in row.values() if isinstance(value, (int, float))):
        raise ValueError("历史日志包含非有限指标")
    output = args.output or directory / "wandb-history"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"导入输出目录非空：{output}；请选择新目录")
    with preserve_rng():
        require_wandb(args)
    config = yaml.safe_load((directory / "config.yaml").read_text())
    runtime = json.loads((directory / "run.json").read_text())
    logger = ExperimentLogger(args, output, {**config, "runtime": runtime,
                                            "imported_metrics": str(directory / "metrics.jsonl")}, "history")
    completed = False
    try:
        for record in records:
            logger.log_training(record)
        completed = True
    finally:
        logger.finish({"train/final_step": records[-1]["step"], "history/record_count": len(records)},
                      exit_code=0 if completed else 1)
    print(f"已导入 {len(records)} 条训练记录；W&B 信息={output / 'wandb_run.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="已有 metrics.jsonl/config.yaml/run.json 的训练目录")
    parser.add_argument("--output", type=Path, help="默认训练目录内 wandb-history")
    add_wandb_args(parser)
    parser.set_defaults(wandb_mode="online")
    import_logs(parser.parse_args())


if __name__ == "__main__":
    main()
