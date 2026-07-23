#!/usr/bin/env python3
"""Summarize real experiment outputs and create paired DS analysis figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml


STAGE2_EXPERIMENTS = [
    "01_ds_default", "09_ds_staged_global", "10_ds_temporal_pre_proj",
    "11_ds_temporal_post_bridge", "12_cqt_temporal_post_bridge",
]
STAGE2_COMPARISONS = [
    ("00_baseline_peft", "01_ds_default"),
    ("01_ds_default", "09_ds_staged_global"),
    ("09_ds_staged_global", "10_ds_temporal_pre_proj"),
    ("10_ds_temporal_pre_proj", "11_ds_temporal_post_bridge"),
    ("11_ds_temporal_post_bridge", "12_cqt_temporal_post_bridge"),
]
MINIMAL_STAGE2_EXPERIMENTS = [
    "10_ds_temporal_pre_proj", "11_ds_temporal_post_bridge", "12_cqt_temporal_post_bridge",
]
MINIMAL_STAGE2_COMPARISONS = [
    ("00_baseline_peft", "11_ds_temporal_post_bridge"),
    ("10_ds_temporal_pre_proj", "11_ds_temporal_post_bridge"),
    ("11_ds_temporal_post_bridge", "12_cqt_temporal_post_bridge"),
]
PAPER_SINGLE_SEED_EXPERIMENTS = [
    "09_ds_staged_global",
    "10_ds_temporal_pre_proj",
    "11_ds_temporal_post_bridge",
    "12_cqt_temporal_post_bridge",
    "13_ds_temporal_shuffled_post_bridge",
]
PAPER_SINGLE_SEED_COMPARISONS = [
    ("00_baseline_peft", "11_ds_temporal_post_bridge"),
    ("09_ds_staged_global", "10_ds_temporal_pre_proj"),
    ("10_ds_temporal_pre_proj", "11_ds_temporal_post_bridge"),
    ("12_cqt_temporal_post_bridge", "11_ds_temporal_post_bridge"),
    ("13_ds_temporal_shuffled_post_bridge", "11_ds_temporal_post_bridge"),
]
HARMONY_KEYWORDS = (
    "harmony", "harmonic", "chord", "tonal", "tonality", "key", "mode",
    "dissonance", "consonance", "cadence", "interval", "pitch",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_run(root: Path, experiment: str, seed: Optional[int] = None) -> Optional[Path]:
    candidates = []
    seed_dirs = [(root / experiment / f"seed_{seed}")] if seed is not None else (root / experiment).glob("seed_*")
    for seed_dir in seed_dirs:
        if not seed_dir.is_dir():
            continue
        candidates.extend(path for path in seed_dir.glob("run_*") if path.is_dir())
    valid = []
    for path in candidates:
        marker = read_json(path / "completed.json")
        complete = marker.get("status") == "completed"
        legacy_complete = all(
            (path / name).is_file()
            for name in ("metrics.jsonl", "predictions.jsonl", "evaluation.json")
        )
        if complete or legacy_complete:
            valid.append(path)
    return max(valid, key=lambda path: path.stat().st_mtime) if valid else None


def prediction_key(record: dict) -> Tuple[str, str]:
    return str(record.get("audio_id")), str(record.get("question_id"))


def question_group(record: dict) -> str:
    question_type = str(record.get("question_type") or "").strip().lower()
    if question_type and question_type not in {"unknown", "none", "null"}:
        return "harmony" if any(word in question_type for word in HARMONY_KEYWORDS) else "other"
    question = str(record.get("question") or "").lower()
    return "harmony" if any(word in question for word in HARMONY_KEYWORDS) else "other"


def mean_metric(records: List[dict], key: str):
    values = [float(record[key]) for record in records if isinstance(record.get(key), (int, float))]
    return float(np.mean(values)) if values else None


def subgroup_metrics(records: List[dict], group: str) -> dict:
    selected = records if group == "overall" else [r for r in records if question_group(r) == group]
    return {
        "samples": len(selected),
        "BLEU": mean_metric(selected, "bleu"),
        "METEOR": mean_metric(selected, "meteor"),
        "ROUGE-L": mean_metric(selected, "rouge_l"),
        "BERTScore": mean_metric(selected, "bertscore_f1"),
        "evaluation_loss": mean_metric(selected, "sample_loss"),
    }


def resolved_config(run: Path) -> dict:
    path = run / "config_resolved.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}


def paired_metric(baseline: List[dict], experiment: List[dict]) -> Tuple[Optional[str], np.ndarray, List[dict]]:
    preferred = ("bertscore_f1", "rouge_l", "bleu", "sample_loss")
    baseline_map = {prediction_key(record): record for record in baseline}
    experiment_map = {prediction_key(record): record for record in experiment}
    keys = sorted(set(baseline_map) & set(experiment_map))
    for metric in preferred:
        paired = [key for key in keys if baseline_map[key].get(metric) is not None
                  and experiment_map[key].get(metric) is not None]
        if paired:
            direction = -1.0 if metric == "sample_loss" else 1.0
            differences = np.asarray([
                direction * (float(experiment_map[key][metric]) - float(baseline_map[key][metric]))
                for key in paired
            ])
            records = [experiment_map[key] for key in paired]
            return metric, differences, records
    return None, np.asarray([]), []


def paired_statistics(
    differences: np.ndarray, bootstrap_samples: int, seed: int,
    clusters: Optional[List[str]] = None,
) -> dict:
    if differences.size == 0:
        return {}
    bootstrap_values = differences
    cluster_count = None
    if clusters:
        grouped = {}
        for cluster, difference in zip(clusters, differences):
            grouped.setdefault(str(cluster), []).append(float(difference))
        bootstrap_values = np.asarray([
            np.mean(grouped[name]) for name in sorted(grouped)
        ], dtype=float)
        cluster_count = int(bootstrap_values.size)
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        boot[index] = rng.choice(
            bootstrap_values, size=bootstrap_values.size, replace=True
        ).mean()
    p_value = min(1.0, 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean())))
    result = {
        "paired_samples": int(differences.size),
        "paired_mean_difference": float(bootstrap_values.mean()),
        "paired_median_difference": float(np.median(bootstrap_values)),
        "paired_improved_fraction": float((bootstrap_values > 0).mean()),
        "paired_ci95_low": float(np.percentile(boot, 2.5)),
        "paired_ci95_high": float(np.percentile(boot, 97.5)),
        "paired_bootstrap_p": p_value,
    }
    if cluster_count is not None:
        result["bootstrap_clusters"] = cluster_count
        result["bootstrap_unit"] = "audio"
    return result


def write_summary(rows: List[dict], root: Path) -> None:
    preferred = [
        "experiment", "group", "feature_type", "temporal_enabled", "fusion_type", "position",
        "trainable_parameters", "best_epoch", "evaluation_split", "evaluation_loss",
        "val_loss", "test_loss", "BLEU", "METEOR", "ROUGE-L",
        "BERTScore", "runtime", "peak_GPU", "samples", "run",
    ]
    available = {key for row in rows for key in row}
    keys = [key for key in preferred if key in available] + sorted(available - set(preferred))
    with (root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(keys) + " |\n")
        handle.write("| " + " | ".join("---" for _ in keys) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |\n")


def no_data(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def make_figures(root: Path, baseline: str, experiments: List[str], runs: Dict[str, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    names = [baseline, *experiments]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plotted = False
    for name in names:
        metrics = runs.get(name, {}).get("metrics", [])
        if metrics:
            epochs = [row.get("epoch") for row in metrics]
            axes[0].plot(epochs, [row.get("train_loss") for row in metrics], marker="o", label=name)
            axes[1].plot(epochs, [row.get("val_loss") for row in metrics], marker="o", label=name)
            plotted = True
    if plotted:
        axes[0].set_title("Training loss"); axes[1].set_title("Validation loss")
        for ax in axes: ax.set_xlabel("Epoch"); ax.grid(alpha=.25); ax.legend(fontsize=7)
    else:
        no_data(axes[0], "No real loss records"); no_data(axes[1], "No real loss records")
    fig.tight_layout(); fig.savefig(figures / "01_loss_curves.png", dpi=180); plt.close(fig)

    metric_names = [
        "bleu", "meteor", "rouge_l", "bertscore_f1",
        "test_loss", "test_perplexity", "val_loss", "val_perplexity",
    ]
    available = [metric for metric in metric_names if any(
        isinstance(runs.get(name, {}).get("evaluation", {}).get(metric), (int, float)) for name in names
    )]
    fig, ax = plt.subplots(figsize=(max(8, len(available) * 1.4), 4.5))
    if available:
        x = np.arange(len(available)); width = .8 / len(names)
        for index, name in enumerate(names):
            values = [runs.get(name, {}).get("evaluation", {}).get(metric, np.nan) for metric in available]
            ax.bar(x + (index - (len(names) - 1) / 2) * width, values, width, label=name)
        ax.set_xticks(x, available, rotation=25, ha="right"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=.25)
    else:
        no_data(ax, "No real evaluation metrics")
    fig.tight_layout(); fig.savefig(figures / "02_metric_comparison.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5)); plotted = False
    baseline_predictions = runs.get(baseline, {}).get("predictions", [])
    for name in experiments:
        metric, differences, _ = paired_metric(baseline_predictions, runs.get(name, {}).get("predictions", []))
        if differences.size:
            ax.hist(differences, bins=min(30, max(5, differences.size // 4)), alpha=.45,
                    label=f"{name} ({metric}, n={differences.size})")
            plotted = True
    if plotted:
        ax.axvline(0, color="black", linewidth=1); ax.set_xlabel("DS improvement over baseline")
        ax.set_ylabel("Samples"); ax.legend(fontsize=7); ax.grid(alpha=.2)
    else:
        no_data(ax, "No paired real sample metrics")
    fig.tight_layout(); fig.savefig(figures / "03_paired_sample_improvement.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5)); plotted = False
    for name in experiments:
        metric, differences, paired_records = paired_metric(
            baseline_predictions, runs.get(name, {}).get("predictions", [])
        )
        if not differences.size:
            continue
        ds_values = np.asarray([record.get("ds_p90") for record in paired_records], dtype=float)
        valid = np.isfinite(ds_values)
        if valid.sum() < 4:
            continue
        quantiles = np.quantile(ds_values[valid], [0, .25, .5, .75, 1])
        groups = np.clip(np.digitize(ds_values[valid], quantiles[1:-1], right=True), 0, 3)
        means = [differences[valid][groups == group].mean() if np.any(groups == group) else np.nan
                 for group in range(4)]
        ax.plot(range(1, 5), means, marker="o", label=f"{name} ({metric})")
        plotted = True
    if plotted:
        ax.axhline(0, color="black", linewidth=1); ax.set_xticks(range(1, 5))
        ax.set_xlabel("Dissonance p90 quartile"); ax.set_ylabel("Mean paired improvement")
        ax.legend(fontsize=7); ax.grid(alpha=.25)
    else:
        no_data(ax, "Insufficient paired real data for quartiles")
    fig.tight_layout(); fig.savefig(figures / "04_improvement_by_dissonance_quartile.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4)); fields = (
        "gate_mean", "ds_residual_ratio", "ds_encoder_grad_norm"
    ); plotted = False
    for name in experiments:
        metrics = runs.get(name, {}).get("metrics", [])
        for ax, field in zip(axes, fields):
            points = [(row.get("epoch"), row.get(field)) for row in metrics if row.get(field) is not None]
            if points:
                ax.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=name)
                plotted = True
    if plotted:
        for ax, field in zip(axes, fields):
            ax.set_title(field); ax.set_xlabel("Epoch"); ax.grid(alpha=.25); ax.legend(fontsize=7)
    else:
        for ax in axes: no_data(ax, "No real DS training records")
    fig.tight_layout(); fig.savefig(figures / "05_gate_residual_curves.png", dpi=180); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--baseline")
    parser.add_argument("--experiments", nargs="+")
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument("--minimal-stage2", action="store_true")
    parser.add_argument("--paper-single-seed", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if sum((args.stage2, args.minimal_stage2, args.paper_single_seed)) > 1:
        parser.error("--stage2, --minimal-stage2, and --paper-single-seed are mutually exclusive")
    comparisons = None
    result_prefix = None
    followup_controls = None
    require_test_split = False
    if args.stage2:
        args.baseline = "00_baseline_peft"
        args.experiments = STAGE2_EXPERIMENTS
        comparisons = STAGE2_COMPARISONS
        result_prefix = "stage2"
        followup_controls = ["00_baseline_peft", "01_ds_default", "12_cqt_temporal_post_bridge"]
    elif args.minimal_stage2:
        args.baseline = "00_baseline_peft"
        args.experiments = MINIMAL_STAGE2_EXPERIMENTS
        comparisons = MINIMAL_STAGE2_COMPARISONS
        result_prefix = "minimal_stage2"
        followup_controls = [
            "00_baseline_peft", "10_ds_temporal_pre_proj", "12_cqt_temporal_post_bridge",
        ]
    elif args.paper_single_seed:
        args.baseline = "00_baseline_peft"
        args.experiments = PAPER_SINGLE_SEED_EXPERIMENTS
        comparisons = PAPER_SINGLE_SEED_COMPARISONS
        result_prefix = "paper_single_seed"
        followup_controls = [
            "00_baseline_peft", "09_ds_staged_global",
            "10_ds_temporal_pre_proj", "12_cqt_temporal_post_bridge",
            "13_ds_temporal_shuffled_post_bridge",
        ]
        require_test_split = True
    if not args.baseline or not args.experiments:
        parser.error("--baseline and --experiments are required unless --stage2 is used")
    args.root.mkdir(parents=True, exist_ok=True)
    report_root = args.report_dir or args.root
    report_root.mkdir(parents=True, exist_ok=True)

    runs = {}
    rows = []
    names = [args.baseline, *args.experiments]
    for index, name in enumerate(names):
        run = latest_run(args.root, name, seed=42 if comparisons is not None else None)
        if run is None:
            print(f"skip {name}: no completed run found")
            continue
        metrics = read_jsonl(run / "metrics.jsonl")
        predictions = read_jsonl(run / "predictions.jsonl")
        evaluation = read_json(run / "evaluation.json")
        split = read_json(run / "data_split.json")
        if require_test_split:
            if evaluation.get("evaluation_split") != "test":
                raise RuntimeError(f"{name} was not evaluated on the held-out test split")
            overlap = split.get("group_overlap", {})
            if not overlap or any(int(value) != 0 for value in overlap.values()):
                raise RuntimeError(f"{name} does not prove leakage-free grouped splits: {overlap}")
        parameters = read_json(run / "parameter_summary.json")
        environment = read_json(run / "environment.txt")
        config = resolved_config(run)
        runs[name] = {
            "path": run, "metrics": metrics, "predictions": predictions,
            "evaluation": evaluation, "split": split,
        }
        ds = config.get("model", {}).get("dissonance", {})
        temporal = ds.get("temporal", {})
        fusion = ds.get("fusion", {})
        enabled = bool(ds.get("enabled", False))
        best_epoch = evaluation.get("best_epoch")
        if best_epoch is None and metrics:
            best_epoch = min(metrics, key=lambda item: item.get("val_loss", float("inf"))).get("epoch")
        common = {
            "experiment": name,
            "run": str(run),
            "feature_type": ds.get("input_feature", "dissonance_spectrum") if enabled else "none",
            "temporal_enabled": bool(temporal.get("enabled", False)),
            "fusion_type": fusion.get("type", "gated_residual") if enabled else "none",
            "position": fusion.get("position", "none") if enabled else "none",
            "trainable_parameters": parameters.get("trainable_parameters"),
            "best_epoch": best_epoch,
            "evaluation_split": evaluation.get("evaluation_split", "validation"),
            "runtime": environment.get("training_duration_seconds"),
            "peak_GPU": environment.get("peak_gpu_memory_mb"),
        }
        for group in ("overall", "harmony", "other"):
            aggregate = subgroup_metrics(predictions, group)
            if group == "overall":
                aggregate.update({
                    "BLEU": evaluation.get("bleu", aggregate["BLEU"]),
                    "METEOR": evaluation.get("meteor", aggregate["METEOR"]),
                    "ROUGE-L": evaluation.get("rouge_l", aggregate["ROUGE-L"]),
                    "BERTScore": evaluation.get("bertscore_f1", aggregate["BERTScore"]),
                    "evaluation_loss": evaluation.get(
                        "test_loss", evaluation.get("val_loss", aggregate["evaluation_loss"])
                    ),
                    "val_loss": evaluation.get("val_loss"),
                    "test_loss": evaluation.get("test_loss"),
                })
            rows.append({**common, "group": group, **aggregate})

    if not rows:
        raise RuntimeError("No real experiment outputs were found; analysis will not fabricate data")
    if comparisons is not None:
        missing = [name for name in names if name not in runs]
        if missing:
            raise RuntimeError(f"{result_prefix} analysis requires all selected runs; missing: {missing}")
    if args.paper_single_seed:
        baseline_keys = {prediction_key(record) for record in runs[args.baseline]["predictions"]}
        baseline_split = runs[args.baseline]["split"].get("splits")
        for name in args.experiments:
            keys = {prediction_key(record) for record in runs[name]["predictions"]}
            if keys != baseline_keys:
                raise RuntimeError(
                    f"{name} test predictions do not match baseline: "
                    f"{len(keys)} vs {len(baseline_keys)} samples"
                )
            if runs[name]["split"].get("splits") != baseline_split:
                raise RuntimeError(f"{name} uses a different train/validation/test partition")
    write_summary(rows, report_root)
    make_figures(report_root, args.baseline, args.experiments, runs)
    if comparisons is not None:
        comparison_rows = []
        for comparison_index, (left, right) in enumerate(comparisons):
            if left not in runs or right not in runs:
                continue
            for group in ("overall", "harmony", "other"):
                left_records = runs[left]["predictions"]
                right_records = runs[right]["predictions"]
                if group != "overall":
                    left_records = [record for record in left_records if question_group(record) == group]
                    right_records = [record for record in right_records if question_group(record) == group]
                metric, differences, paired_records = paired_metric(left_records, right_records)
                comparison_rows.append({
                    "comparison": f"{left} vs {right}",
                    "group": group,
                    "paired_metric": metric,
                    **paired_statistics(
                        differences, args.bootstrap_samples, args.seed + comparison_index,
                        clusters=[str(record.get("audio_id")) for record in paired_records]
                        if args.paper_single_seed else None,
                    ),
                })
        with (report_root / f"{result_prefix}_comparisons.json").open("w", encoding="utf-8") as handle:
            json.dump(comparison_rows, handle, ensure_ascii=False, indent=2)
        if args.paper_single_seed:
            claim_names = {
                "00_baseline_peft vs 11_ds_temporal_post_bridge": "DS temporal efficacy",
                "09_ds_staged_global vs 10_ds_temporal_pre_proj": "temporal modeling benefit",
                "10_ds_temporal_pre_proj vs 11_ds_temporal_post_bridge": "post-bridge placement",
                "12_cqt_temporal_post_bridge vs 11_ds_temporal_post_bridge": "DS feature specificity",
                "13_ds_temporal_shuffled_post_bridge vs 11_ds_temporal_post_bridge":
                    "chronological-order contribution",
            }
            claims = []
            for row in comparison_rows:
                if row["group"] != "overall":
                    continue
                claims.append({
                    "claim": claim_names[row["comparison"]],
                    "comparison": row["comparison"],
                    "metric": row.get("paired_metric"),
                    "supported": (
                        isinstance(row.get("paired_ci95_low"), (int, float))
                        and row["paired_ci95_low"] > 0
                    ),
                    "ci95": [row.get("paired_ci95_low"), row.get("paired_ci95_high")],
                    "paired_samples": row.get("paired_samples"),
                })
            claims_payload = {
                    "primary_rule": (
                        "audio-clustered paired bootstrap 95% CI lower bound must be above zero"
                    ),
                    "single_seed_limitation": (
                        "Sample-level uncertainty only; training-seed variability is not estimated."
                    ),
                    "claims": claims,
                    "core_efficacy_supported": next(
                        item["supported"] for item in claims
                        if item["claim"] == "DS temporal efficacy"
                    ),
                    "all_mechanism_claims_supported": all(
                        item["supported"] for item in claims
                        if item["claim"] != "DS temporal efficacy"
                    ),
                }
            (report_root / "paper_single_seed_claims.json").write_text(
                json.dumps(claims_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            recommended = runs.get("11_ds_temporal_post_bridge", {}).get("evaluation", {})
            controls = [runs.get(name, {}).get("evaluation", {}) for name in followup_controls]
            def score(item):
                for key in ("bertscore_f1", "rouge_l", "meteor", "bleu"):
                    value = item.get(key)
                    if isinstance(value, (int, float)):
                        return float(value)
                return -float("inf")
            eligible = bool(recommended) and all(
                control and score(recommended) > score(control) for control in controls
            )
            rule = "11 must outperform " + ", ".join(followup_controls)
            (report_root / f"{result_prefix}_followup.json").write_text(json.dumps({
                "eligible": eligible,
                "rule": rule + " before running seeds 3407 and 2026",
                "experiments": [
                    "00_baseline_peft", "11_ds_temporal_post_bridge",
                    "12_cqt_temporal_post_bridge",
                ],
                "seeds": [3407, 2026],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        (report_root / f"{result_prefix}_analysis_complete.json").write_text(json.dumps({
            "status": "completed", "experiments": [args.baseline, *args.experiments]
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {report_root / 'summary.csv'}, {report_root / 'summary.md'}, and {report_root / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
