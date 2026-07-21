#!/usr/bin/env python3
"""Config-driven MU-LLaMA fine-tuning with optional Dissonance Spectrum."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import llama.utils
import util.misc as misc
from data.dataset import FinetuneDataset, finetune_collate, transform_train
from engine_finetune import train_one_epoch
from llama.llama_adapter import LLaMA_adapter
from util.config import config_fingerprint, dump_config, load_config


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_jsonl_for_summary(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_output_dir(config: Dict[str, Any], smoke_test: bool) -> Path:
    experiment = config.get("experiment", {}).get("name")
    if not experiment:
        experiment = Path(config["_config_path"]).stem
        config.setdefault("experiment", {})["name"] = experiment
    seed = int(config.get("training", {}).get("seed", 0))
    root = Path(config.get("output", {}).get("root", "outputs"))
    prefix = "smoke_" if smoke_test else "run_"
    run_id = prefix + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = root / experiment / f"seed_{seed}" / run_id
    for folder in ("checkpoints", "tensorboard", "figures"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    return output.resolve()


def git_value(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, check=False, capture_output=True, text=True
        )
        return (result.stdout or result.stderr).strip()
    except OSError as error:
        return f"unavailable: {error}"


def write_environment(path: Path, config: Dict[str, Any], checkpoint_report: Dict[str, Any]) -> None:
    repo = Path(__file__).resolve().parents[1]
    lines = {
        "command": " ".join(sys.argv),
        "git_commit": git_value(repo, "rev-parse", "HEAD"),
        "git_dirty": bool(git_value(repo, "status", "--porcelain")),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": checkpoint_report,
        "config_source": config.get("_config_path"),
    }
    path.write_text(json.dumps(lines, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def parameter_groups(model: torch.nn.Module, config: Dict[str, Any]) -> Tuple[List[dict], Dict[str, Any]]:
    learning_rates = config.get("training", {}).get("learning_rates", {})
    group_lrs = {
        "ds_encoder": float(learning_rates.get("ds_encoder", 3e-4)),
        "ds_temporal": float(learning_rates.get("ds_temporal", learning_rates.get("ds_encoder", 3e-4))),
        "ds_fusion": float(learning_rates.get("ds_fusion", 3e-4)),
        "prefix_query": float(learning_rates.get("prefix_query", 2e-5)),
        "bridge_norm": float(learning_rates.get("bridge_norm", 2e-5)),
        "llama_peft": float(learning_rates.get("llama_peft", 1e-4)),
    }
    weight_decay = float(config.get("training", {}).get("weight_decay", 0.05))
    stage2_no_decay = config.get("training", {}).get("trainable_mode") == "ds_stage2_minimal"
    grouped: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {key: [] for key in group_lrs}
    unexpected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("ds_encoder."):
            grouped["ds_encoder"].append((name, parameter))
        elif name.startswith("ds_temporal."):
            grouped["ds_temporal"].append((name, parameter))
        elif name.startswith("ds_fusion."):
            grouped["ds_fusion"].append((name, parameter))
        elif name.startswith("prefix_query."):
            grouped["prefix_query"].append((name, parameter))
        elif name.startswith(("mu_mert_norm_1.", "mu_mert_norm_2.", "mu_mert_norm_3.")):
            grouped["bridge_norm"].append((name, parameter))
        elif name.startswith("llama.") and any(token in name for token in ("lora", "bias", "norm")):
            grouped["llama_peft"].append((name, parameter))
        else:
            unexpected.append(name)
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected}")

    base_lr = max(group_lrs.values())
    optimizer_groups = []
    summary_groups = []
    for group_name, named_parameters in grouped.items():
        if not named_parameters:
            continue
        for decay in (False, True):
            selected = [parameter for name, parameter in named_parameters
                        if (
                            parameter.ndim > 1
                            and not name.endswith(".bias")
                            and (not stage2_no_decay or "gate" not in name.lower())
                            and (not stage2_no_decay or "norm" not in name.lower())
                        ) == decay]
            if selected:
                optimizer_groups.append({
                    "params": selected,
                    "lr": group_lrs[group_name],
                    "lr_scale": group_lrs[group_name] / base_lr,
                    "weight_decay": weight_decay if decay else 0.0,
                    "group_name": group_name,
                })
        summary_groups.append({
            "name": group_name,
            "learning_rate": group_lrs[group_name],
            "parameters": [name for name, _ in named_parameters],
            "parameter_count": sum(parameter.numel() for _, parameter in named_parameters),
        })

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen_modules = []
    for module_name, module in model.named_modules():
        parameters = list(module.parameters(recurse=False))
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            frozen_modules.append(module_name or "<root>")
    summary = {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": trainable / total if total else 0.0,
        "trainable_mode": config.get("training", {}).get("trainable_mode"),
        "groups": summary_groups,
        "frozen_top_level_modules": frozen_modules,
        "unexpected_trainable_parameters": unexpected,
        "new_dissonance_parameters": sum(
            parameter.numel() for name, parameter in model.named_parameters()
            if name.startswith(("ds_encoder.", "ds_temporal.", "ds_fusion."))
        ),
    }
    return optimizer_groups, summary


def split_dataset(dataset, validation_fraction: float, seed: int):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("data.validation_fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    validation_size = max(1, int(round(len(order) * validation_fraction)))
    validation = order[:validation_size]
    training = order[validation_size:]
    if not training:
        raise ValueError("Dataset is too small for the requested validation split")
    return Subset(dataset, training), Subset(dataset, validation)


def move_batch(batch, device):
    examples, labels, masks, audio, extras = batch
    ds = extras.get("dissonance")
    ds_mask = extras.get("dissonance_mask")
    return (
        examples.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        masks.to(device, non_blocking=True),
        audio.to(device, non_blocking=True),
        ds.to(device, non_blocking=True) if ds is not None else None,
        ds_mask.to(device, non_blocking=True) if ds_mask is not None else None,
        extras["audio_lengths"].to(device, non_blocking=True),
        extras["metadata"],
    )


def ds_sample_statistics(spectrum, mask) -> Dict[str, Any]:
    if spectrum is None or mask is None or not mask.any():
        return {key: None for key in ("ds_mean", "ds_std", "ds_p90", "ds_max", "ds_active_ratio")}
    values = spectrum[:, mask].detach().float().cpu().numpy().reshape(-1)
    return {
        "ds_mean": float(values.mean()),
        "ds_std": float(values.std()),
        "ds_p90": float(np.percentile(values, 90)),
        "ds_max": float(values.max()),
        "ds_active_ratio": float(np.count_nonzero(values) / values.size),
    }


def stable_generation_seed(base_seed: int, question_id: str) -> int:
    suffix = int(hashlib.sha1(question_id.encode("utf-8")).hexdigest()[:8], 16)
    return (base_seed + suffix) % (2 ** 31)


def evaluate_loader(model, loader, device, config, generate_predictions=False, smoke_test=False):
    model.eval()
    total_loss = 0.0
    sample_count = 0
    predictions = []
    max_batches = 1 if smoke_test else config.get("validation", {}).get("max_batches")
    generation = config.get("generation", {})
    precision = config.get("training", {}).get("precision", "fp16")
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    max_generation_samples = 1 if smoke_test else generation.get("max_samples")
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            examples, labels, _, audio, ds, ds_mask, audio_lengths, metadata = move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=autocast_dtype):
                loss, _, sample_losses = model(
                    examples, labels, audio, dissonance=ds, dissonance_mask=ds_mask,
                    audio_lengths=audio_lengths, return_sample_losses=True,
                )
            total_loss += float(sample_losses.sum().item())
            sample_count += int(sample_losses.numel())
            if not generate_predictions:
                continue
            for index, item in enumerate(metadata):
                if max_generation_samples is not None and len(predictions) >= int(max_generation_samples):
                    break
                seed = stable_generation_seed(int(generation.get("seed", 0)), item["question_id"])
                set_seed(seed)
                prompt = llama.utils.format_prompt(item["question"])
                length = int(audio_lengths[index].item())
                sample_audio = audio[index:index + 1, :length]
                sample_ds = ds[index:index + 1] if ds is not None else None
                sample_mask = ds_mask[index:index + 1] if ds_mask is not None else None
                prediction = model.generate(
                    {"Audio": [sample_audio, 1]}, [prompt],
                    max_gen_len=int(generation.get("max_gen_len", 128)),
                    temperature=float(generation.get("temperature", 0.1)),
                    top_p=float(generation.get("top_p", 0.75)),
                    dissonance=sample_ds, dissonance_mask=sample_mask,
                    audio_lengths=torch.tensor([length], device=device),
                )[0]
                model_stats = getattr(model, "last_dissonance_stats", {})
                record = {
                    "audio_id": item["audio_id"],
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "question_type": item["question_type"],
                    "reference": item["reference"],
                    "prediction": prediction,
                    "sample_loss": float(sample_losses[index].item()),
                    **ds_sample_statistics(
                        sample_ds[0] if sample_ds is not None else None,
                        sample_mask[0] if sample_mask is not None else None,
                    ),
                    "gate_mean": float(model_stats["gate_mean"].item()) if "gate_mean" in model_stats else None,
                    "attention_entropy": float(model_stats["attention_entropy"].item())
                    if "attention_entropy" in model_stats else None,
                    "ds_residual_ratio": float(model_stats["ds_residual_ratio"].item())
                    if "ds_residual_ratio" in model_stats else None,
                    "generation_seed": seed,
                    "generation": {
                        "max_gen_len": int(generation.get("max_gen_len", 128)),
                        "temperature": float(generation.get("temperature", 0.1)),
                        "top_p": float(generation.get("top_p", 0.75)),
                    },
                }
                predictions.append(record)
    mean_loss = total_loss / max(1, sample_count)
    return {
        "val_loss": mean_loss,
        "val_perplexity": math.exp(min(20.0, mean_loss)),
        "validation_samples": sample_count,
    }, predictions


def automatic_evaluation(records: List[dict]) -> Tuple[Dict[str, Any], List[dict]]:
    evaluation_path = Path(__file__).resolve().parents[1] / "ModelEvaluations" / "evaluate.py"
    try:
        spec = importlib.util.spec_from_file_location("mu_llama_evaluate", evaluation_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {evaluation_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.evaluate_records(records)
    except Exception as error:
        return {"missing_dependencies": [f"evaluation: {error}"]}, records


def save_checkpoint(path, model, optimizer, scaler, epoch, config):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    }, path)


def load_resume(path, model, optimizer, scaler) -> int:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except ValueError as error:
        print(f"resume optimizer state reset after a training-stage change: {error}")
    scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]) + 1


def resume_start_epoch(path) -> int:
    if not path:
        return 0
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    return int(checkpoint["epoch"]) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-words", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = int(args.seed)
        config.setdefault("generation", {})["seed"] = int(args.seed)
    if args.max_words is not None:
        if args.max_words < 32:
            raise ValueError("--max-words must be at least 32")
        config.setdefault("model", {})["max_words"] = int(args.max_words)
    output_dir = run_output_dir(config, args.smoke_test)
    dump_config(config, output_dir / "config_resolved.yaml")

    log_handle = (output_dir / "train.log").open("a", encoding="utf-8")
    with contextlib.redirect_stdout(Tee(sys.stdout, log_handle)), contextlib.redirect_stderr(Tee(sys.stderr, log_handle)):
        run_start = time.time()
        print(f"resolved output: {output_dir}")
        training = config.get("training", {})
        model_config = config.get("model", {})
        seed = int(training.get("seed", 0))
        set_seed(seed)
        device = torch.device(training.get("device", "cuda"))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        llama_path = Path(model_config["llama_path"])
        llama_type = str(model_config.get("llama_type", "7B"))
        tokenizer_path = llama_path / "tokenizer.model"
        model = LLaMA_adapter(
            str(llama_path / llama_type), str(tokenizer_path), model_config.get("mert_path", "m-a-p/MERT-v1-330M"),
            phase="finetune", dissonance_config=model_config.get("dissonance"),
            trainable_mode=training.get("trainable_mode", "baseline_peft"),
        )
        checkpoint_path = model_config.get("pretrained_path")
        if not checkpoint_path:
            raise ValueError("model.pretrained_path must identify the common MU-LLaMA checkpoint")
        checkpoint_report = misc.load_model(model, checkpoint_path)
        model.to(device)

        trainable_mode = training.get("trainable_mode", "baseline_peft")
        stage1_epochs = int(training.get("stage1_epochs", 2))
        planned_start_epoch = resume_start_epoch(training.get("resume"))
        current_stage = 2 if trainable_mode == "ds_stage2_minimal" and planned_start_epoch >= stage1_epochs else 1
        if trainable_mode == "ds_stage2_minimal":
            model.set_stage2_training_stage(1)
            _, stage1_summary = parameter_groups(model, config)
            model.set_stage2_training_stage(2)
            _, stage2_summary = parameter_groups(model, config)
            model.set_stage2_training_stage(current_stage)
        else:
            stage1_summary = stage2_summary = None
        groups, parameter_summary = parameter_groups(model, config)
        if stage1_summary is not None:
            parameter_summary = dict(stage2_summary)
            parameter_summary["stages"] = {
                "stage_1": stage1_summary,
                "stage_2": stage2_summary,
            }
            parameter_summary["active_stage_at_start"] = current_stage
        (output_dir / "parameter_summary.json").write_text(
            json.dumps(parameter_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(parameter_summary, ensure_ascii=False, indent=2))
        base_lr = max(group["lr"] for group in groups)
        optimizer = torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.95))
        scaler = misc.NativeScalerWithGradNormCount()
        start_epoch = 0
        if training.get("resume"):
            start_epoch = load_resume(training["resume"], model, optimizer, scaler)
            checkpoint_report["resume"] = str(training["resume"])
            checkpoint_report["resume_start_epoch"] = start_epoch
        write_environment(output_dir / "environment.txt", config, checkpoint_report)

        data_config = config.get("data", {})
        dataset = FinetuneDataset(
            data_config["train_config"], transform=transform_train,
            max_words=int(model_config.get("max_words", 512)), tokenizer_path=str(tokenizer_path),
            audio_root=data_config.get("audio_root", "../MusicQA/audios"),
            dissonance_config=model_config.get("dissonance"), return_metadata=True,
        )
        train_dataset, validation_dataset = split_dataset(
            dataset, float(data_config.get("validation_fraction", 0.1)),
            int(data_config.get("split_seed", seed)),
        )
        loader_kwargs = {
            "num_workers": int(data_config.get("num_workers", 4)),
            "pin_memory": bool(data_config.get("pin_memory", True)),
            "collate_fn": finetune_collate,
        }
        train_loader = DataLoader(
            train_dataset, batch_size=int(training.get("batch_size", 1)), shuffle=True,
            drop_last=True, generator=torch.Generator().manual_seed(seed), **loader_kwargs,
        )
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, **loader_kwargs)
        writer = SummaryWriter(log_dir=output_dir / "tensorboard")

        configured_epochs = int(training.get(
            "epochs", int(training.get("stage1_epochs", 0)) + int(training.get("stage2_epochs", 0)) or 20
        ))
        epochs = 1 if args.smoke_test else configured_epochs
        warmup_epochs = training.get("warmup_epochs")
        if warmup_epochs is None:
            warmup_epochs = float(training.get("warmup_ratio", 0.0)) * epochs
        engine_args = SimpleNamespace(
            accum_iter=1 if args.smoke_test else int(training.get("accum_iter", 1)),
            warmup_epochs=float(warmup_epochs),
            epochs=epochs,
            min_lr=float(training.get("min_lr", 0.0)),
            lr=base_lr,
            max_train_batches=2 if args.smoke_test else training.get("max_train_batches"),
            precision=training.get("precision", "fp16"),
            gradient_clip_norm=float(training.get("gradient_clip_norm", 0.0)) or None,
        )
        metrics_path = output_dir / "metrics.jsonl"
        final_predictions = []
        final_evaluation = {}
        best_val_loss = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        patience = int(training.get("early_stopping", {}).get("patience", 0))
        best_checkpoint = output_dir / "checkpoints" / "checkpoint_best.pth"
        stopped_early = False
        for epoch in range(start_epoch, epochs):
            desired_stage = 2 if trainable_mode == "ds_stage2_minimal" and epoch >= stage1_epochs else 1
            if desired_stage != current_stage:
                model.set_stage2_training_stage(desired_stage)
                groups, active_summary = parameter_groups(model, config)
                base_lr = max(group["lr"] for group in groups)
                optimizer = torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.95))
                scaler = misc.NativeScalerWithGradNormCount()
                engine_args.lr = base_lr
                current_stage = desired_stage
                epochs_without_improvement = 0
                print(json.dumps({
                    "training_stage": current_stage,
                    "trainable_parameters": active_summary["trainable_parameters"],
                }, ensure_ascii=False))
            epoch_start = time.time()
            train_stats = train_one_epoch(
                model, train_loader, optimizer, device, epoch, scaler,
                log_writer=writer, args=engine_args,
            )
            is_last = epoch + 1 == epochs
            validation_stats, _ = evaluate_loader(
                model, validation_loader, device, config,
                generate_predictions=False, smoke_test=args.smoke_test,
            )
            field_names = [
                "gate_mean", "gate_std", "ds_embedding_norm", "mert_embedding_norm",
                "fused_embedding_norm", "ds_residual_ratio", "ds_encoder_grad_norm",
                "attention_entropy", "ds_temporal_grad_norm", "ds_fusion_grad_norm",
                "llama_peft_grad_norm", "stage2_extra_grad_norm",
            ]
            record = {
                "epoch": epoch,
                "train_loss": train_stats.get("closs"),
                **validation_stats,
                "learning_rate": train_stats.get("lr"),
                **{name: train_stats.get(name) for name in field_names},
                "epoch_time": time.time() - epoch_start,
                "peak_gpu_memory": train_stats.get("peak_gpu_memory", 0.0),
            }
            if trainable_mode == "ds_stage2_minimal":
                record["training_stage"] = current_stage
            append_jsonl(metrics_path, record)
            for key, value in record.items():
                if isinstance(value, (int, float)) and key != "epoch":
                    writer.add_scalar(key, value, epoch)
            save_every = int(training.get("save_every", 1))
            if is_last or (epoch + 1) % save_every == 0:
                save_checkpoint(
                    output_dir / "checkpoints" / f"checkpoint_epoch_{epoch:03d}.pth",
                    model, optimizer, scaler, epoch, config,
                )
            current_val = float(validation_stats["val_loss"])
            if best_epoch < 0 or current_val < best_val_loss:
                best_val_loss = current_val
                best_epoch = epoch
                epochs_without_improvement = 0
                if patience > 0:
                    save_checkpoint(best_checkpoint, model, optimizer, scaler, epoch, config)
            else:
                epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                stopped_early = True
                print(f"early stopping at epoch {epoch}; best epoch was {best_epoch}")
                break

        if patience > 0 and best_checkpoint.is_file():
            try:
                best_state = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
            except TypeError:
                best_state = torch.load(best_checkpoint, map_location="cpu")
            model.load_state_dict(best_state["model"], strict=True)
        final_validation, final_predictions = evaluate_loader(
            model, validation_loader, device, config,
            generate_predictions=True, smoke_test=args.smoke_test,
        )
        final_evaluation, final_predictions = automatic_evaluation(final_predictions)
        final_evaluation.update(final_validation)
        final_evaluation.update({
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "stopped_early": stopped_early,
        })

        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for record in final_predictions:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        (output_dir / "evaluation.json").write_text(
            json.dumps(final_evaluation, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        environment_path = output_dir / "environment.txt"
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        environment["training_duration_seconds"] = time.time() - run_start
        environment["peak_gpu_memory_mb"] = max((
            (record.get("peak_gpu_memory") or 0.0)
            for record in read_jsonl_for_summary(metrics_path)
        ), default=0.0) if metrics_path.is_file() else 0.0
        environment_path.write_text(
            json.dumps(environment, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        (output_dir / "completed.json").write_text(json.dumps({
            "status": "completed",
            "experiment": config.get("experiment", {}).get("name"),
            "seed": seed,
            "split_seed": data_config.get("split_seed", seed),
            "pretrained_path": str(checkpoint_path),
            "max_words": int(model_config.get("max_words", 512)),
            "best_epoch": best_epoch,
            "config_source": config.get("_config_path"),
            "config_fingerprint": config_fingerprint(config),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        writer.close()
        print(f"completed: {output_dir}")
    log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
