---
license: mit
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

No benchmark values are bundled or fabricated. See `../docs/ds_stage2_update.md` for implementation and verification details.
