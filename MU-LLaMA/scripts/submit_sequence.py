#!/usr/bin/env python3
"""Submit cache/train/analysis jobs with afterok dependencies."""

from __future__ import annotations

import argparse
import json
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


def experiment_config_path(workdir: Path, experiment: str, group: str) -> Path:
    if group in {"stage2_core", "stage2_followup"}:
        budget_path = (
            workdir / "configs" / "experiments" / "stage2_core_budget" / f"{experiment}.yaml"
        )
        if budget_path.is_file():
            return budget_path
    return workdir / "configs" / "experiments" / f"{experiment}.yaml"


def completed_run(config_path: Path, workdir: Path, experiment: str, seed: int) -> Path | None:
    if not config_path.is_file():
        return None
    experiment_config = load_config(config_path)
    output_root = Path(experiment_config.get("output", {}).get("root", "outputs"))
    if not output_root.is_absolute():
        output_root = workdir / output_root
    run_dirs = sorted(
        (output_root / experiment / f"seed_{seed}").glob("run_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        marker = run_dir / "completed.json"
        if marker.is_file():
            try:
                state = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            expected_split = experiment_config.get("data", {}).get("split_seed")
            expected_checkpoint = experiment_config.get("model", {}).get("pretrained_path")
            if (
                state.get("status") == "completed"
                and int(state.get("seed", -1)) == int(seed)
                and state.get("split_seed") == expected_split
                and str(state.get("pretrained_path")) == str(expected_checkpoint)
            ):
                return marker
        # Stage 1 runs predate completed.json. Reuse only a fully materialized
        # run with the same seed, data split, and common pretrained checkpoint.
        required = ("metrics.jsonl", "predictions.jsonl", "evaluation.json", "config_resolved.yaml")
        if not all((run_dir / name).is_file() for name in required):
            continue
        try:
            old_config = load_config(run_dir / "config_resolved.yaml")
        except (OSError, ValueError):
            continue
        old_training = old_config.get("training", {})
        old_data = old_config.get("data", {})
        old_model = old_config.get("model", {})
        same_seed = int(old_training.get("seed", -1)) == int(seed)
        same_split = old_data.get("split_seed") == experiment_config.get("data", {}).get("split_seed")
        same_checkpoint = str(old_model.get("pretrained_path")) == str(
            experiment_config.get("model", {}).get("pretrained_path")
        )
        if same_seed and same_split and same_checkpoint:
            return run_dir / "evaluation.json"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True)
    parser.add_argument("--slurm-config", default="configs/slurm.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.slurm_config)
    cluster = config["cluster"]
    sequence = sequence_for(args.group, config["experiment_groups"])

    configured_workdir = cluster.get("workdir", ".")
    workdir = PROJECT_ROOT if configured_workdir in (None, "", ".", "auto") else Path(configured_workdir).resolve()
    if args.group == "stage2_followup":
        decision_path = workdir / "outputs" / "stage2_followup.json"
        if not decision_path.is_file():
            raise RuntimeError("Run the stage2_core analysis before requesting follow-up seeds")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if not decision.get("eligible", False):
            raise RuntimeError(f"Stage 2 follow-up condition was not met: {decision.get('rule')}")
    log_dir = Path(cluster.get("log_dir", "slurm/logs"))
    if not log_dir.is_absolute():
        log_dir = workdir / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    export_value = (
        f"ALL,WORKDIR={workdir},PYTHON_EXECUTABLE={cluster.get('python_executable', 'python')},"
        f"CONDA_ENV={cluster.get('conda_env') or ''}"
    )
    previous_job = None
    for raw_step in sequence:
        step, separator, explicit_seed = raw_step.partition("@")
        explicit_seed = int(explicit_seed) if separator else None
        is_cache = step.startswith("cache_")
        is_analysis = step in {"analyze_results", "analyze_stage2"}
        experiment = step[len("cache_"):] if is_cache else step
        selected_config = None if is_analysis else experiment_config_path(workdir, experiment, args.group)
        if selected_config is not None and not selected_config.is_file():
            raise FileNotFoundError(f"Experiment config not found: {selected_config}")
        config_argument = (
            str(selected_config.relative_to(workdir))
            if selected_config is not None and selected_config.is_relative_to(workdir)
            else str(selected_config) if selected_config is not None else None
        )
        stage2_seed = explicit_seed if explicit_seed is not None else (42 if args.group == "stage2_core" else None)
        completion_target = experiment
        if is_cache:
            completion_target = experiment
        completion = None
        if not is_analysis:
            completion = completed_run(
                selected_config, workdir, completion_target,
                stage2_seed if stage2_seed is not None else int(
                    load_config(selected_config)
                    .get("training", {}).get("seed", 0)
                ),
            )
        elif step == "analyze_stage2":
            marker = workdir / "outputs" / "stage2_analysis_complete.json"
            completion = marker if marker.is_file() and previous_job is None else None
        if completion is not None and not args.force:
            print(f"skip completed {step}: {completion}")
            continue
        job_label = f"{step}_seed{stage2_seed}" if explicit_seed is not None else step
        if args.group in {"stage2_core", "stage2_followup"} and not is_analysis:
            job_label = f"{job_label}_b6"
        command = ["sbatch", "--parsable", f"--job-name={job_label}"]
        command.extend(("--nodes", str(cluster.get("nodes", 1))))
        command.extend(("--ntasks", str(cluster.get("tasks", 1))))
        command.extend(("--cpus-per-task", str(cluster.get("cpus_per_task", 8))))
        command.extend(("--mem", str(cluster.get("memory", "64G"))))
        time_limit = (
            cluster.get("stage2_time", "40:00:00")
            if args.group in {"stage2_core", "stage2_followup"}
            else cluster.get("time", "24:00:00")
        )
        command.extend(("--time", str(time_limit)))
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
                            config_argument))
            if args.force:
                command.append("--force")
        elif step == "analyze_stage2":
            command.extend((
                str(PROJECT_ROOT / "slurm" / "train.slurm"), "analyze", "--stage2",
                "--root", "outputs",
            ))
        elif is_analysis:
            command.extend((
                str(PROJECT_ROOT / "slurm" / "train.slurm"), "analyze",
                "--root", "outputs", "--baseline", "00_baseline_peft", "--experiments",
                "01_ds_default", "02_ds_only", "03_ds_channel_gate", "04_ds_post_proj",
                "05_ds_mlp_encoder", "06_ds_bpo36", "07_ds_hop512", "08_ds_hop2048",
            ))
        else:
            command.extend((str(PROJECT_ROOT / "slurm" / "train.slurm"), "train",
                            config_argument))
            if stage2_seed is not None:
                command.extend(("--seed", str(stage2_seed)))

        print(f"experiment: {step}" + (f" (seed {stage2_seed})" if stage2_seed is not None else ""))
        print(f"dependency: {dependency or 'none'}")
        print(f"command: {shlex.join(command)}")
        if args.dry_run:
            previous_job = f"DRYRUN_{job_label}"
            print(f"log: {log_dir / (job_label + '-<job_id>.out')}")
            continue
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        previous_job = result.stdout.strip().split(";", 1)[0]
        print(f"job ID: {previous_job}")
        print(f"log: {log_dir / (job_label + '-' + previous_job + '.out')}")
        print(f"squeue: squeue -j {previous_job}")
        print(f"tail: bash scripts/tail_log.sh {previous_job} {shlex.quote(str(log_dir))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
