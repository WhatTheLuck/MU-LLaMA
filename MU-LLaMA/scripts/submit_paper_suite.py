#!/usr/bin/env python3
"""Submit the complete single-seed paper suite within a four-job Slurm quota."""

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

from scripts.submit_minimal_screen import verify_token_audit  # noqa: E402
from scripts.submit_sequence import add_optional, completed_run  # noqa: E402
from util.config import load_config  # noqa: E402


CONFIG_DIR = Path("configs/experiments/paper_single_seed")
GROUPS = {
    "paper_a_baseline_global": [
        CONFIG_DIR / "00_baseline_peft.yaml",
        CONFIG_DIR / "09_ds_staged_global.yaml",
    ],
    "paper_b_position": [
        CONFIG_DIR / "10_ds_temporal_pre_proj.yaml",
        CONFIG_DIR / "11_ds_temporal_post_bridge.yaml",
    ],
    "paper_c_controls": [
        CONFIG_DIR / "12_cqt_temporal_post_bridge.yaml",
        CONFIG_DIR / "13_ds_temporal_shuffled_post_bridge.yaml",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slurm-config", default="configs/slurm.yaml")
    parser.add_argument("--max-words", type=int, default=256)
    parser.add_argument(
        "--token-audit", type=Path,
        default=Path("outputs/paper_single_seed/token_length_audit.json"),
    )
    parser.add_argument(
        "--cache-ready", action="store_true",
        help="Confirm that the DS and CQT caches have already completed successfully.",
    )
    parser.add_argument(
        "--cache-only", action="store_true",
        help="Submit only the two cache-extension jobs for the official evaluation set.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.cache_only and not args.cache_ready:
        raise RuntimeError(
            "The paper suite reuses completed DS/CQT caches; pass --cache-ready only after verifying them."
        )

    slurm = load_config(args.slurm_config)["cluster"]
    configured_workdir = slurm.get("workdir", ".")
    workdir = (
        PROJECT_ROOT
        if configured_workdir in (None, "", ".", "auto")
        else Path(configured_workdir).resolve()
    )
    if not args.cache_only:
        audit_path = args.token_audit if args.token_audit.is_absolute() else workdir / args.token_audit
        verify_token_audit(audit_path, args.max_words)
        if args.max_words < 512:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if not audit.get("includes_test_config", False):
                raise RuntimeError(
                    "Paper token audit must include both FinetuneMusicQA and EvalMusicQA."
                )
    log_dir = Path(slurm.get("log_dir", "slurm/logs"))
    if not log_dir.is_absolute():
        log_dir = workdir / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    export_value = (
        f"ALL,WORKDIR={workdir},PYTHON_EXECUTABLE={slurm.get('python_executable', 'python')},"
        f"CONDA_ENV={slurm.get('conda_env') or ''}"
    )
    jobs = {}

    def submit(label: str, script_args: list[str], dependencies=(), kind="group"):
        dependency_ids = [jobs[name] for name in dependencies if jobs.get(name)]
        command = ["sbatch", "--parsable", f"--job-name={label}"]
        command.extend(("--nodes", str(slurm.get("nodes", 1))))
        command.extend(("--ntasks", str(slurm.get("tasks", 1))))
        command.extend(("--cpus-per-task", str(slurm.get("cpus_per_task", 8))))
        command.extend((
            "--mem", str("32G" if kind == "analysis" else slurm.get("memory", "64G"))
        ))
        time_limit = {
            "analysis": "02:00:00",
            "cache": slurm.get("time", "24:00:00"),
            "group": slurm.get("paper_group_time", "47:30:00"),
        }[kind]
        command.extend(("--time", str(time_limit)))
        add_optional(command, "--partition", slurm.get("partition"))
        add_optional(command, "--account", slurm.get("account"))
        add_optional(command, "--qos", slurm.get("qos"))
        if kind != "analysis":
            add_optional(command, "--gres", slurm.get("gres", "gpu:a100:1"))
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

    if args.cache_only:
        for label, relative_config in (
            ("cache_paper_ds_eval", CONFIG_DIR / "11_ds_temporal_post_bridge.yaml"),
            ("cache_paper_cqt_eval", CONFIG_DIR / "12_cqt_temporal_post_bridge.yaml"),
        ):
            cache_args = [
                str(PROJECT_ROOT / "slurm" / "cache.slurm"), str(relative_config),
            ]
            if args.force:
                cache_args.append("--force")
            submit(label, cache_args, kind="cache")
        print(json.dumps(jobs, ensure_ascii=False, indent=2))
        return 0

    group_labels = []
    for group_name, configs in GROUPS.items():
        incomplete = []
        for relative_config in configs:
            resolved = load_config(workdir / relative_config)
            experiment = str(resolved["experiment"]["name"])
            marker = None if args.force else completed_run(
                workdir / relative_config, workdir, experiment, 42,
                expected_max_words=args.max_words, require_exact_config=True,
            )
            if marker is None:
                incomplete.append(relative_config)
            else:
                print(f"reuse exact paper run {experiment}: {marker}")
        if not incomplete:
            jobs[group_name] = None
            group_labels.append(group_name)
            continue
        group_args = [
            str(PROJECT_ROOT / "slurm" / "train.slurm"), "group",
            "--group-name", group_name, "--configs", *map(str, configs),
            "--seed", "42", "--max-words", str(args.max_words),
        ]
        if args.force:
            group_args.append("--force")
        submit(group_name, group_args)
        group_labels.append(group_name)

    report_dir = Path("outputs/paper_single_seed/analysis")
    marker = workdir / report_dir / "paper_single_seed_analysis_complete.json"
    newly_submitted = any(jobs.get(label) for label in group_labels)
    if marker.is_file() and not newly_submitted and not args.force:
        print(f"skip completed paper analysis: {marker}")
    else:
        submit("analyze_paper_single_seed", [
            str(PROJECT_ROOT / "slurm" / "train.slurm"), "analyze",
            "--paper-single-seed", "--root", "outputs/paper_single_seed",
            "--report-dir", str(report_dir), "--bootstrap-samples", "10000",
            "--seed", "42",
        ], dependencies=tuple(group_labels), kind="analysis")
    print(json.dumps(jobs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
