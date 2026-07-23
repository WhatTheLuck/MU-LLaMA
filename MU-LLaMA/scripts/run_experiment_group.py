#!/usr/bin/env python3
"""Run a completion-aware sequence of paper experiments in one GPU allocation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.submit_sequence import completed_run  # noqa: E402
from util.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-name", required=True)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-words", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    completed = []
    for config_path in args.configs:
        absolute_config = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
        resolved = load_config(absolute_config)
        experiment = str(resolved.get("experiment", {}).get("name") or absolute_config.stem)
        marker = None if args.force else completed_run(
            absolute_config, PROJECT_ROOT, experiment, args.seed,
            expected_max_words=args.max_words, require_exact_config=True,
        )
        if marker is not None:
            print(f"skip completed {experiment}: {marker}", flush=True)
            completed.append({"experiment": experiment, "marker": str(marker), "reused": True})
            continue
        command = [
            sys.executable, str(PROJECT_ROOT / "train.py"),
            "--config", str(config_path), "--seed", str(args.seed),
            "--max-words", str(args.max_words),
        ]
        print(f"run {experiment}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        marker = completed_run(
            absolute_config, PROJECT_ROOT, experiment, args.seed,
            expected_max_words=args.max_words, require_exact_config=True,
        )
        if marker is None:
            raise RuntimeError(f"{experiment} exited without an exact completed marker")
        completed.append({"experiment": experiment, "marker": str(marker), "reused": False})

    marker_dir = PROJECT_ROOT / "outputs" / "paper_single_seed" / "group_markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"{args.group_name}.json"
    marker_path.write_text(json.dumps({
        "status": "completed",
        "group": args.group_name,
        "seed": args.seed,
        "max_words": args.max_words,
        "experiments": completed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"completed group {args.group_name}: {marker_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
