#!/usr/bin/env python3
"""Submit cache/train/analysis jobs with afterok dependencies."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.config import load_config  # noqa: E402


def add_optional(command, flag, value):
    if value not in (None, ""):
        command.extend((flag, str(value)))


def sequence_for(group: str, groups: dict):
    if group == "all":
        sequence = [*groups["core"], *groups["architecture"], *groups["spectrum"]]
        sequence.append("analyze_results")
        return sequence
    if group not in groups:
        raise ValueError(f"Unknown group {group!r}; choose from {sorted(groups)} or all")
    return list(groups[group])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True)
    parser.add_argument("--slurm-config", default="configs/slurm.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.slurm_config)
    cluster = config["cluster"]
    sequence = sequence_for(args.group, config["experiment_groups"])

    workdir = Path(cluster.get("workdir", ".")).resolve()
    log_dir = Path(cluster.get("log_dir", "slurm/logs"))
    if not log_dir.is_absolute():
        log_dir = workdir / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    export_value = (
        f"ALL,WORKDIR={workdir},PYTHON_EXECUTABLE={cluster.get('python_executable', 'python')},"
        f"CONDA_ENV={cluster.get('conda_env') or ''}"
    )
    previous_job = None
    for step in sequence:
        is_cache = step.startswith("cache_")
        is_analysis = step == "analyze_results"
        experiment = step[len("cache_"):] if is_cache else step
        command = ["sbatch", "--parsable", f"--job-name={step}"]
        command.extend(("--nodes", str(cluster.get("nodes", 1))))
        command.extend(("--ntasks", str(cluster.get("tasks", 1))))
        command.extend(("--cpus-per-task", str(cluster.get("cpus_per_task", 8))))
        command.extend(("--mem", str(cluster.get("memory", "64G"))))
        command.extend(("--time", str(cluster.get("time", "24:00:00"))))
        add_optional(command, "--partition", cluster.get("partition"))
        add_optional(command, "--account", cluster.get("account"))
        add_optional(command, "--qos", cluster.get("qos"))
        add_optional(command, "--gres", cluster.get("gres", "gpu:a100:1"))
        command.extend(("--output", str(log_dir / "%x-%j.out"), "--export", export_value))
        dependency = f"afterok:{previous_job}" if previous_job else None
        if dependency:
            command.extend(("--dependency", dependency))

        if is_cache:
            command.extend((str(PROJECT_ROOT / "slurm" / "cache.slurm"),
                            f"configs/experiments/{experiment}.yaml"))
        elif is_analysis:
            command.extend((
                str(PROJECT_ROOT / "slurm" / "train.slurm"), "analyze",
                "--root", "outputs", "--baseline", "00_baseline_peft", "--experiments",
                "01_ds_default", "02_ds_only", "03_ds_channel_gate", "04_ds_post_proj",
                "05_ds_mlp_encoder", "06_ds_bpo36", "07_ds_hop512", "08_ds_hop2048",
            ))
        else:
            command.extend((str(PROJECT_ROOT / "slurm" / "train.slurm"), "train",
                            f"configs/experiments/{experiment}.yaml"))

        print(f"experiment: {step}")
        print(f"dependency: {dependency or 'none'}")
        print(f"command: {shlex.join(command)}")
        if args.dry_run:
            previous_job = f"DRYRUN_{step}"
            print(f"log: {log_dir / (step + '-<job_id>.out')}")
            continue
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        previous_job = result.stdout.strip().split(";", 1)[0]
        print(f"job ID: {previous_job}")
        print(f"log: {log_dir / (step + '-' + previous_job + '.out')}")
        print(f"squeue: squeue -j {previous_job}")
        print(f"tail: bash scripts/tail_log.sh {previous_job} {shlex.quote(str(log_dir))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

