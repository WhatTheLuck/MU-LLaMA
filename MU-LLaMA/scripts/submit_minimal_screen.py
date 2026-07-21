#!/usr/bin/env python3
"""Submit the minimal 00/10/11/12 screening DAG without serializing independent jobs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.submit_sequence import add_optional, completed_run  # noqa: E402
from util.config import load_config  # noqa: E402


CONFIG_DIR = Path("configs/experiments/stage2_minimal_screen")
TRAIN_CONFIGS = {
    "00_baseline_peft": CONFIG_DIR / "00_baseline_peft.yaml",
    "10_ds_temporal_pre_proj": CONFIG_DIR / "10_ds_temporal_pre_proj.yaml",
    "11_ds_temporal_post_bridge": CONFIG_DIR / "11_ds_temporal_post_bridge.yaml",
    "12_cqt_temporal_post_bridge": CONFIG_DIR / "12_cqt_temporal_post_bridge.yaml",
}


def verify_token_audit(path: Path, max_words: int) -> None:
    if max_words >= 512:
        return
    if not path.is_file():
        raise RuntimeError(
            f"max_words={max_words} requires a token audit: {path}. "
            "Run tools/audit_token_lengths.py first."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    result = report.get("candidates", {}).get(str(max_words))
    if not result or not result.get("passes", False):
        raise RuntimeError(
            f"Token audit does not permit max_words={max_words}: {result}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slurm-config", default="configs/slurm.yaml")
    parser.add_argument("--max-words", type=int, default=512)
    parser.add_argument("--token-audit", type=Path, default=Path("outputs/token_length_audit.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.slurm_config)
    cluster = config["cluster"]
    configured_workdir = cluster.get("workdir", ".")
    workdir = (
        PROJECT_ROOT
        if configured_workdir in (None, "", ".", "auto")
        else Path(configured_workdir).resolve()
    )
    audit_path = args.token_audit if args.token_audit.is_absolute() else workdir / args.token_audit
    verify_token_audit(audit_path, args.max_words)

    log_dir = Path(cluster.get("log_dir", "slurm/logs"))
    if not log_dir.is_absolute():
        log_dir = workdir / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    export_value = (
        f"ALL,WORKDIR={workdir},PYTHON_EXECUTABLE={cluster.get('python_executable', 'python')},"
        f"CONDA_ENV={cluster.get('conda_env') or ''}"
    )

    jobs = {}

    def submit(label: str, script_args: list[str], dependencies=(), kind="train"):
        dependency_ids = [jobs[name] for name in dependencies if jobs.get(name)]
        command = ["sbatch", "--parsable", f"--job-name={label}"]
        command.extend(("--nodes", str(cluster.get("nodes", 1))))
        command.extend(("--ntasks", str(cluster.get("tasks", 1))))
        command.extend(("--cpus-per-task", str(cluster.get("cpus_per_task", 8))))
        command.extend(("--mem", str("32G" if kind == "analysis" else cluster.get("memory", "64G"))))
        if kind == "train":
            time_limit = cluster.get("minimal_time", "30:00:00")
        elif kind == "analysis":
            time_limit = "02:00:00"
        else:
            time_limit = cluster.get("time", "24:00:00")
        command.extend(("--time", str(time_limit)))
        add_optional(command, "--partition", cluster.get("partition"))
        add_optional(command, "--account", cluster.get("account"))
        add_optional(command, "--qos", cluster.get("qos"))
        if kind != "analysis":
            add_optional(command, "--gres", cluster.get("gres", "gpu:a100:1"))
        command.extend(("--output", str(log_dir / "%x-%j.out"), "--export", export_value))
        if dependency_ids:
            command.extend(("--dependency", "afterok:" + ":".join(dependency_ids)))
        command.extend(script_args)
        print(f"task: {label}")
        print(f"dependencies: {dependency_ids or ['none']}")
        print(f"command: {shlex.join(command)}")
        if args.dry_run:
            jobs[label] = f"DRYRUN_{label}"
        else:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            jobs[label] = result.stdout.strip().split(";", 1)[0]
        print(f"job ID: {jobs[label]}")

    cache_ds = "cache_minimal_ds"
    cache_cqt = "cache_minimal_cqt"
    ds_config = str(TRAIN_CONFIGS["10_ds_temporal_pre_proj"])
    cqt_config = str(TRAIN_CONFIGS["12_cqt_temporal_post_bridge"])
    submit(cache_ds, [str(PROJECT_ROOT / "slurm" / "cache.slurm"), ds_config]
           + (["--force"] if args.force else []), kind="cache")
    submit(cache_cqt, [str(PROJECT_ROOT / "slurm" / "cache.slurm"), cqt_config]
           + (["--force"] if args.force else []), kind="cache")

    train_labels = []
    for experiment, relative_config in TRAIN_CONFIGS.items():
        config_path = workdir / relative_config
        label = f"{experiment}_m4"
        completion = None if args.force else completed_run(
            config_path, workdir, experiment, 42, expected_max_words=args.max_words,
            require_exact_config=True,
        )
        if completion is not None:
            print(f"skip completed {label}: {completion}")
            jobs[label] = None
            train_labels.append(label)
            continue
        dependencies = ()
        if experiment in {"10_ds_temporal_pre_proj", "11_ds_temporal_post_bridge"}:
            dependencies = (cache_ds,)
        elif experiment == "12_cqt_temporal_post_bridge":
            dependencies = (cache_cqt,)
        submit(label, [
            str(PROJECT_ROOT / "slurm" / "train.slurm"), "train", str(relative_config),
            "--seed", "42", "--max-words", str(args.max_words),
        ], dependencies=dependencies, kind="train")
        train_labels.append(label)

    report_dir = Path("outputs/minimal_stage2_analysis")
    marker = workdir / report_dir / "minimal_stage2_analysis_complete.json"
    newly_submitted = any(jobs.get(label) for label in train_labels)
    if marker.is_file() and not newly_submitted and not args.force:
        print(f"skip completed analyze_minimal_stage2: {marker}")
    else:
        submit("analyze_minimal_stage2", [
            str(PROJECT_ROOT / "slurm" / "train.slurm"), "analyze",
            "--minimal-stage2", "--root", "outputs", "--report-dir", str(report_dir),
        ], dependencies=tuple(train_labels), kind="analysis")
    print(json.dumps({key: value for key, value in jobs.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
