"""Reusable, dependency-tolerant text evaluation for MU-LLaMA predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


def evaluate_records(records: List[dict]) -> Tuple[Dict[str, Any], List[dict]]:
    augmented = [dict(record) for record in records]
    if not augmented:
        return {"sample_count": 0, "missing_dependencies": []}, augmented
    references = [str(record.get("reference", "")) for record in augmented]
    predictions = [str(record.get("prediction", "")) for record in augmented]
    metrics: Dict[str, Any] = {"sample_count": len(augmented)}
    missing = []

    try:
        from nltk.tokenize import wordpunct_tokenize
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        smoothing = SmoothingFunction().method1
        bleu_values = []
        bleu4_values = []
        for record, reference, prediction in zip(augmented, references, predictions):
            ref_tokens = wordpunct_tokenize(reference)
            pred_tokens = wordpunct_tokenize(prediction)
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
            bleu4 = sentence_bleu(
                [ref_tokens], pred_tokens, weights=(0.0, 0.0, 0.0, 1.0),
                smoothing_function=smoothing,
            )
            record["bleu"] = float(bleu)
            record["bleu4"] = float(bleu4)
            bleu_values.append(float(bleu))
            bleu4_values.append(float(bleu4))
        metrics["bleu"] = mean(bleu_values)
        metrics["bleu4"] = mean(bleu4_values)
    except Exception as error:
        missing.append(f"BLEU (nltk): {error}")

    try:
        from nltk.tokenize import wordpunct_tokenize
        from nltk.translate.meteor_score import meteor_score

        values = []
        for record, reference, prediction in zip(augmented, references, predictions):
            value = float(meteor_score([wordpunct_tokenize(reference)], wordpunct_tokenize(prediction)))
            record["meteor"] = value
            values.append(value)
        metrics["meteor"] = mean(values)
    except Exception as error:
        missing.append(f"METEOR (nltk/resources): {error}")

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        values = []
        for record, reference, prediction in zip(augmented, references, predictions):
            value = float(scorer.score(reference, prediction)["rougeL"].fmeasure)
            record["rouge_l"] = value
            values.append(value)
        metrics["rouge_l"] = mean(values)
    except Exception as error:
        missing.append(f"ROUGE-L (rouge-score): {error}")

    try:
        from bert_score import score as bert_score

        precision, recall, f1 = bert_score(predictions, references, lang="en", verbose=False)
        p_values = precision.detach().cpu().tolist()
        r_values = recall.detach().cpu().tolist()
        f_values = f1.detach().cpu().tolist()
        for record, p_value, r_value, f_value in zip(augmented, p_values, r_values, f_values):
            record["bertscore_precision"] = float(p_value)
            record["bertscore_recall"] = float(r_value)
            record["bertscore_f1"] = float(f_value)
        metrics.update({
            "bertscore_precision": mean(p_values),
            "bertscore_recall": mean(r_values),
            "bertscore_f1": mean(f_values),
        })
    except Exception as error:
        missing.append(f"BERTScore (bert-score/model): {error}")

    metrics["missing_dependencies"] = missing
    return metrics, augmented


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics, records = evaluate_records(read_jsonl(args.predictions))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

