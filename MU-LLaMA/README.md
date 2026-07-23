---
license: gpl-3.0
tags:
- music
---
# MU-LLaMA: Music Understanding Large Language Model

This contains the weights for *MU-LLaMA: Large Language Model for Music Question Answering*. The code for the model is provided [here](https://github.com/crypto-code/MU-LLaMA).

## Dissonance Spectrum Stage 2

Stage 2 keeps experiments `00`–`08` intact and adds the following controlled chain:

| Config | Change |
| --- | --- |
| `09_ds_staged_global` | Stage 1 global DS fusion with the new two-stage minimal training schedule |
| `10_ds_temporal_pre_proj` | Ordered temporal DS tokens and masked gated attention before MERT projection |
| `11_ds_temporal_post_bridge` | The same temporal fusion after the three MERT bridge blocks |
| `12_cqt_temporal_post_bridge` | Processed-CQT control with the same tensor shape, encoder, fusion, and training setup |

The new cache stores both the full `[frequency, time]` Dissonance Spectrum and the processed CQT. Config `12` uses the CQT tensor directly; it does not reconstruct it from the Dissonance Spectrum.

Run a local smoke check with one config:

```bash
python train.py --config configs/experiments/11_ds_temporal_post_bridge.yaml --smoke-test
```

Preview or submit the single-A100 Stage 2 chain:

```bash
python scripts/submit_sequence.py --group stage2_core --dry-run
python scripts/submit_sequence.py --group stage2_core
```

Completed runs are skipped by default. Add `--force` to resubmit them. After Stage 2 analysis, the optional extra seeds can only be submitted when the generated `outputs/stage2_followup.json` marks the documented comparison rule as eligible:

```bash
python scripts/submit_sequence.py --group stage2_followup --dry-run
python scripts/submit_sequence.py --group stage2_followup
```

With `cluster.workdir: .`, the submitter resolves the repository directory from its own location, so it can also be called from outside the checkout. Null partition/account/QoS values are omitted and left to the cluster defaults; a null Conda environment keeps the agent's currently active environment.

Because a full epoch was observed to take about 4.8 hours on the target A100, `stage2_core` now selects additive configs under `configs/experiments/stage2_core_budget/`: at most 6 epochs, early-stopping patience 2, full data and validation, and one periodic save at epoch 6. The original 00–12 configs are unchanged. Budgeted Slurm jobs carry a `_b6` suffix and receive a 40-hour limit, leaving headroom beyond the roughly 28.8-hour worst-case training body for validation, generation, and checkpoint I/O.

No benchmark values are bundled or fabricated. See `../docs/ds_stage2_update.md` for implementation and verification details.

### Minimal screening DAG

For the shortest position/feature comparison, use the additive four-run screen `00/10/11/12`. It uses 4 epochs, a 1+3 staged schedule, 128-dimensional temporal attention, a one-parameter learned scalar gate, and parallel cache/training branches. Original and six-run budget configs remain unchanged.

Audit whether 256 tokens preserves at least 99% of real samples:

```bash
python tools/audit_token_lengths.py --require-max-words 256
```

If the audit passes:

```bash
python scripts/submit_minimal_screen.py --max-words 256 --dry-run
python scripts/submit_minimal_screen.py --max-words 256
```

If it fails, omit `--max-words 256`; the safe default is 512. The DAG starts the DS cache, CQT cache, and baseline independently; 10/11 wait only for the DS cache, 12 waits only for the CQT cache, and analysis joins all four completed training jobs.

### Single-seed paper suite

The paper suite runs `00/09/10/11/12/13` with seed 42. Config `13` is a parameter-matched deterministic temporal-order shuffle. FinetuneMusicQA is split into audio-grouped 90/10 train/validation partitions, and the repository's separate EvalMusicQA set is used only for final testing. Validation selects the best checkpoint; only the untouched official evaluation set supplies paper metrics.

First extend the existing DS and CQT caches to EvalMusicQA and audit both datasets:

```bash
python scripts/submit_paper_suite.py --cache-only --slurm-config /path/to/runtime/slurm.yaml --dry-run
python scripts/submit_paper_suite.py --cache-only --slurm-config /path/to/runtime/slurm.yaml
python tools/audit_token_lengths.py --config configs/experiments/paper_single_seed/00_baseline_peft.yaml --require-max-words 256 --output outputs/paper_single_seed/token_length_audit.json
```

After both cache jobs complete successfully, preview and submit:

```bash
python scripts/submit_paper_suite.py --cache-ready --slurm-config /path/to/runtime/slurm.yaml --dry-run
python scripts/submit_paper_suite.py --cache-ready --slurm-config /path/to/runtime/slurm.yaml
```

The submitter packs two experiments into each of three parallel GPU jobs and submits one dependent CPU analysis job, staying within a four-job user quota. Analysis uses 10,000 audio-clustered paired bootstrap samples and writes claim-by-claim confidence intervals under `outputs/paper_single_seed/analysis/`. Existing minimal-screen results use a validation-only split and remain development evidence; they are not mixed into the paper table.

See `../docs/reproducibility_checklist.md` for dataset provenance, exact metric definitions, the code-appendix index, infrastructure fields, hyperparameter provenance, and the manuscript checklist answers.
