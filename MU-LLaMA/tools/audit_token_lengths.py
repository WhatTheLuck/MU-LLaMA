#!/usr/bin/env python3
"""Audit prompt+answer token lengths before enabling a shorter training sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.config import load_config  # noqa: E402


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)


def percentile(values, fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def resolve_path(path_value, base: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.is_file():
        return path.resolve()
    return (base / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/stage2_minimal_screen/00_baseline_peft.yaml")
    parser.add_argument("--candidates", nargs="+", type=int, default=[128, 192, 256, 512])
    parser.add_argument("--max-truncation-rate", type=float, default=0.01)
    parser.add_argument("--require-max-words", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs/token_length_audit.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    llama_root = Path(config["model"]["llama_path"])
    tokenizer_path = resolve_path(llama_root / "tokenizer.model", PROJECT_ROOT)
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    data_config_path = resolve_path(config["data"]["train_config"], PROJECT_ROOT)
    data_config = load_config(data_config_path)

    lengths = []
    for meta_ref in data_config.get("META", []):
        meta_path = resolve_path(meta_ref, data_config_path.parent)
        records = json.loads(meta_path.read_text(encoding="utf-8"))
        for record in records:
            conversation = record.get("conversation") or []
            if len(conversation) < 2:
                continue
            question = str(conversation[0].get("value", ""))
            answer = str(conversation[1].get("value", ""))
            prompt = PROMPT_TEMPLATE.format_map({"instruction": question, "input": input})
            lengths.append(len(processor.encode(prompt + answer)) + 2)  # BOS + EOS
    if not lengths:
        raise RuntimeError("No question-answer records were found for token auditing")

    candidates = {}
    for candidate in sorted(set(args.candidates)):
        truncated = sum(length > candidate for length in lengths)
        candidates[str(candidate)] = {
            "truncated_samples": truncated,
            "truncation_fraction": truncated / len(lengths),
            "passes": truncated / len(lengths) <= args.max_truncation_rate,
        }
    passing = [int(value) for value, result in candidates.items() if result["passes"]]
    report = {
        "config_source": config.get("_config_path"),
        "data_config": str(data_config_path),
        "tokenizer_path": str(tokenizer_path),
        "samples": len(lengths),
        "min": min(lengths),
        "p50": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
        "max_truncation_rate": args.max_truncation_rate,
        "candidates": candidates,
        "recommended_max_words": min(passing) if passing else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_max_words is not None:
        result = candidates.get(str(args.require_max_words))
        if result is None or not result["passes"]:
            print(
                f"max_words={args.require_max_words} exceeds the allowed truncation rate",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
