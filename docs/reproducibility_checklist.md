# Reproducibility checklist evidence

This file is the code-appendix index for the single-seed paper suite. It records what the artifact proves and what still has to be stated in the manuscript. It does not claim that derived Dissonance Spectrum or CQT cache tensors are a new dataset; they are deterministic intermediate representations of existing MusicQA audio.

## Recommended checklist responses

| Item | Current answer | Evidence or remaining manuscript action |
| --- | --- | --- |
| 3.1 | yes | The experiments train and evaluate on MusicQA. |
| 3.2 | yes | MusicQA directly measures music question answering; its MTT-derived fine-tuning split and disjoint MTG-Jamendo-derived evaluation split match the target task while separating source collections. State this motivation in the dataset section. |
| 3.3 | NA | This work introduces no dataset. DS/CQT caches are deterministic preprocessing artifacts, not a dataset contribution. |
| 3.4 | NA | No novel dataset is introduced. Do not promise redistribution of upstream audio under a new license. |
| 3.5 | partial until manuscript citations are inserted | Use the citations below for MusicQA/MU-LLaMA, MusicCaps, MagnaTagATune, and MTG-Jamendo. |
| 3.6 | partial | MusicQA and its source datasets have public access points, but upstream audio availability and reuse terms differ; MusicCaps audio depends on source videos and MTG-Jamendo audio has track-specific Creative Commons licenses. |
| 3.7 | NA | No private dataset is used. If any local replacement audio is introduced, change this answer and document it. |
| 4.1 | yes | The paper suite contains six computational experiments. |
| 4.2 | yes after the manuscript summarizes the provenance file | `hyperparameter_provenance.yaml` states the number/values considered or tried and the selection criterion. Parameters fixed without a sweep are explicitly identified as single-value choices. |
| 4.3 | yes if this repository is submitted as the code appendix | Dataset generation and feature caching entry points are indexed below. |
| 4.4 | yes if this repository snapshot is submitted/archived | Training, scheduling, evaluation, statistics, and all resolved configs are included. |
| 4.5 | yes | The repository root `LICENSE` is GPL-3.0, which permits free research use, modification, and redistribution subject to its terms. Archive the exact release commit with the paper. |
| 4.6 | partial | New-method modules contain stable paper anchors and implementation comments. Replace the provisional anchor names with final manuscript section numbers before submission. |
| 4.7 | yes | Training, split, generation, bootstrap, and shuffle seeds are all 42 or deterministically derived from 42; deterministic PyTorch controls are enabled and recorded. |
| 4.8 | yes after formal runs finish | Every run writes OS, CPU, RAM, GPU model/count/memory, CUDA, Python, PyTorch, package versions, Slurm allocation, Git commit, and deterministic settings to `environment.txt`. Report the values from the formal runs, not workstation values. |
| 4.9 | yes after the manuscript includes the metric text below | The primary and secondary metrics, implementations, directions, and motivation are fixed before paper analysis. |
| 4.10 | yes | Each reported condition uses one run, seed 42. State this explicitly and describe it as a limitation. |
| 4.11 | yes | Reports include per-audio distributions, harmony/other subgroups, 95% clustered bootstrap confidence intervals, and improved fractions. |
| 4.12 | yes after analysis completes | Directed hypotheses use one-sided paired Wilcoxon signed-rank tests on per-audio mean differences, with Holm correction across the five planned overall comparisons. Claims additionally require the clustered-bootstrap 95% CI lower bound to exceed zero. |
| 4.13 | yes | Versioned YAML configs list final parameters; each run stores `config_resolved.yaml`, and analysis exports all final resolved configs together. |

## Dataset appendix evidence

The paper does not introduce a new dataset. It uses the existing MusicQA artifact:

| Partition | Records | Unique audio | Source role |
| --- | ---: | ---: | --- |
| `FinetuneMusicQA.json` | 70,011 | 7,779 | MTT-derived training data, deterministically split by audio into 90% train and 10% validation |
| `EvalMusicQA.json` | 5,040 | 560 | MTG-Jamendo-derived official test data, never used for model or hyperparameter selection |

Public access and licensing must be described without implying that a dataset-repository label overrides the rights attached to source recordings:

- MusicQA: <https://huggingface.co/datasets/mu-llama/MusicQA>. The card is publicly accessible and tagged MIT, with MTT fine-tuning and MTG evaluation splits. Upstream audio terms still apply.
- MusicCaps: <https://huggingface.co/datasets/google/MusicCaps>. The metadata/captions are listed as CC BY-SA 4.0; audio is referenced through source video identifiers.
- MagnaTagATune: <https://mirg.city.ac.uk/datasets/magnatagatune/>. Cite Law et al. (2009), *Evaluation of Algorithms Using Games: The Case of Music Tagging*, ISMIR.
- MTG-Jamendo: <https://mtg.github.io/mtg-jamendo-dataset/>. Metadata is CC BY-NC-SA 4.0; recordings carry individual Creative Commons licenses and the official site restricts the dataset to non-commercial research/academic use absent separate authorization.

Required paper citations:

1. Liu et al. (2023), *MU-LLaMA: Music Understanding Large Language Model*, arXiv:2308.11276.
2. Agostinelli et al. (2023), *MusicLM: Generating Music From Text*, arXiv:2301.11325 (MusicCaps).
3. Law, West, Mandel, Bay, and Downie (2009), *Evaluation of Algorithms Using Games: The Case of Music Tagging*, ISMIR (MagnaTagATune).
4. Bogdanov, Won, Tovstogan, Porter, and Serra (2019), *The MTG-Jamendo Dataset for Automatic Music Tagging*, ML4MD at ICML.

## Hyperparameters and run count

The authoritative development-range and selection record is:

`MU-LLaMA/configs/experiments/paper_single_seed/hyperparameter_provenance.yaml`

The lightweight 128-dimensional temporal/attention width and learned-scalar gate were selected from the compute and parameter budget before any successful Stage 2 performance run; they were not chosen by comparing official-test scores. The provenance file records zero completed performance trials for those choices rather than presenting implementation variants as successful sweeps.

The final six conditions are `00`, `09`, `10`, `11`, `12`, and `13`; each has exactly one formal run with seed 42. The official test set must not be inspected to choose among configurations. `00/09/10/11/12/13` are controlled comparisons, not six independent hyperparameter attempts selected by test performance.

## Metric definitions

- **Primary metric — BERTScore F1:** token-level contextual similarity between generated and reference answers, computed with `bert-score==0.3.13`, `roberta-large`, English mode, and no baseline rescaling. It is primary because valid music-QA answers can be semantically correct without exact lexical overlap. Higher is better.
- **BLEU:** mean sentence BLEU with uniform 1–4 gram weights, NLTK `wordpunct_tokenize`, and smoothing method 1. It measures local lexical/phrase overlap. Higher is better.
- **METEOR:** mean NLTK METEOR over tokenized answer/reference pairs. It supplements BLEU with alignment that accounts for more flexible lexical matching. Higher is better.
- **ROUGE-L:** mean longest-common-subsequence F1 from `rouge-score`, with stemming enabled. It measures sequence-level content overlap. Higher is better.
- **Test loss/perplexity:** teacher-forced answer loss and its exponentiated value on the untouched official test set. They diagnose probabilistic fit but are secondary to semantic generation quality. Lower is better.

The five directional comparisons are fixed in advance: `00→11` (core efficacy), `09→10` (temporal representation), `10→11` (fusion placement), `12→11` (DS specificity), and `13→11` (chronological order). Statistical units are audio tracks, not individual questions.

## Code appendix index

- MusicQA construction: `MusicQA/generate_dataset.py` and the source-specific processing scripts in `MusicQA/`.
- DS/CQT preprocessing and cache validation: `MU-LLaMA/tools/cache_dissonance.py`, `MU-LLaMA/features/dissonance_adapter.py`, and `DissonanceSpectrum/dissonance_spectrum.py`.
- Leakage-free partitioning: `MU-LLaMA/util/data_split.py`.
- Temporal DS encoder, order control, and fusion: `MU-LLaMA/llama/dissonance_modules.py`.
- Model insertion points and trainability: `MU-LLaMA/llama/llama_adapter.py`.
- Training, best-checkpoint selection, independent test evaluation, seeds, and environment capture: `MU-LLaMA/train.py`.
- Exact paper configs: `MU-LLaMA/configs/experiments/paper_single_seed/`.
- Four-job Slurm workflow: `MU-LLaMA/scripts/submit_paper_suite.py` and `MU-LLaMA/scripts/run_experiment_group.py`.
- Text metrics: `ModelEvaluations/evaluate.py`.
- Clustered confidence intervals, Wilcoxon tests, Holm correction, plots, and claim decisions: `MU-LLaMA/scripts/analyze_results.py`.

Before submission, archive the exact Git commit used for formal runs, replace provisional method anchors with final paper section numbers, cite the four dataset papers, and copy the actual formal-run infrastructure values from `environment.txt` into the manuscript.
