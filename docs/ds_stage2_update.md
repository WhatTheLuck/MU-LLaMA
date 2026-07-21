# Dissonance Spectrum Stage 2 update

## Pre-change inspection

The Stage 1 branch encoded each Dissonance Spectrum as one global vector. The CNN reduced frequency and then performed a masked mean over time, so the fusion layer could not distinguish event order. Fusion was a gated residual at `pre_proj` or `post_proj`; there was no insertion point after the three MERT bridge feed-forward blocks. Fine-tuning supported baseline PEFT, DS-only, or DS-plus-LoRA, but not the requested two-stage parameter schedule. The cache retained the Dissonance Spectrum and intensity but discarded the already-computed processed CQT.

## Implemented design

- The legacy global CNN forward path and its parameter names remain unchanged for configs `01`–`08`.
- When `temporal.enabled` is true, the CNN feature map is frequency-pooled without time pooling, projected to `token_dim`, passed through depthwise and pointwise temporal convolutions, combined with parameter-free sinusoidal positions, normalized, and masked. The model-side DS contract contains `global_embedding`, `temporal_tokens`, and `temporal_mask`.
- `temporal_gated_attention` uses the MERT-side vector as one query and the ordered DS tokens as keys and values. Invalid frames are excluded before softmax. It logs `attention_entropy` and `gate_mean`. The context output projection is zero-initialized, so the initial residual is exactly zero.
- `post_bridge` applies fusion after all three MERT bridge blocks and before the audio vector is combined with the LLaMA prefix queries. Training, validation, inference, and generation all use the same `forward_audio` path.
- `ds_stage2_minimal` trains only DS encoder/temporal/fusion parameters for epochs 1–2. From epoch 3 it additionally trains `prefix_query` and `mu_mert_norm_1/2/3`; MERT, `mu_mert_proj`, bridge linears, LLaMA/LoRA, and the output head stay frozen.
- Optimizer groups use `1e-4` for DS encoder/temporal/fusion and `2e-5` for prefix queries and bridge norms. Stage 2 configs use bf16, batch size 1, accumulation 32, gradient clipping 1, weight decay 0.01, 10% warmup, cosine decay, and early-stopping patience 2. Biases, gates, and norms have no weight decay in this mode.
- The original 00–12 definitions remain available for provenance. Operational `stage2_core` submissions select additive `stage2_core_budget` overlays capped at 6 epochs. Staged experiments retain two DS-only epochs followed by up to four minimal-unfreeze epochs. Full training and validation data are retained; this is not a smoke-test or batch truncation.
- Cache metadata now records configured/effective FPS and hop length plus frequency and time dimensions. If FPS overrides hop length, both configured and effective values remain visible. Config `12` gets a separate cache hash and reads the original `processed_calc_cqt` returned by the feature implementation.

The temporal module is below 1M parameters. The largest temporal fusion variant (`post_bridge`, dimension 4096) is below 5M parameters. MERT and LLaMA remain frozen in the Stage 2 mode.

## Experiment and scheduler contract

Round 1 is `00`, `01`, `09`, `10`, `11`, and `12` with seed 42. `stage2_core` submits them in order, including the config-12 cache and final analysis, using `afterok` dependencies. Its budgeted training jobs have a `_b6` suffix and a 40-hour Slurm limit. At the observed rate of about 4.8 hours per epoch, six epochs require about 28.8 hours before final validation/generation overhead. A successful `completed.json` causes default skipping; `--force` overrides that behavior. Dry-run prints every command and dependency without submitting.

Analysis writes overall, harmony, and other rows. It uses an existing `question_type` when available; otherwise it classifies only the question text with the fixed harmony keyword list in `scripts/analyze_results.py`. It reports the five adjacent comparisons required by the design. It never hides the overall result.

Extra seeds 3407 and 2026 are gated by analysis: experiment `11` must outperform `00`, `01`, and `12`. Only then does `stage2_followup` allow `00`, `11`, and `12` to be submitted for those seeds.

## Verification commands

```bash
python -m compileall -q .
pytest -q
python train.py --config configs/experiments/09_ds_staged_global.yaml --smoke-test
python train.py --config configs/experiments/10_ds_temporal_pre_proj.yaml --smoke-test
python train.py --config configs/experiments/11_ds_temporal_post_bridge.yaml --smoke-test
python train.py --config configs/experiments/12_cqt_temporal_post_bridge.yaml --smoke-test
```

Each smoke run is capped by the existing smoke path to one epoch, at most two training batches, one validation batch, and one generated example. Formal training and performance claims require the target single-A100 cluster and real checkpoints/data; no formal run has been performed in this repository update.
