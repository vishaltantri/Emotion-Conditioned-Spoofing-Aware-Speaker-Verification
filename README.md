# Emotion-Conditioned Spoofing-Aware Speaker Verification (EASV)

Research codebase for the question: **does emotion-conditioned fusion of speaker and
anti-spoofing evidence improve spoofing-aware speaker verification under
cross-emotion and cross-generator evaluation, without unacceptably increasing
rejection of genuine speech?**

The plan this repo implements is [Emotion_Aware_Speaker_Verification_Research_Plan.md](Emotion_Aware_Speaker_Verification_Research_Plan.md);
the literature-review design is in [Emotion_Aware_Anti_Spoofing_Speaker_Verification-1.pdf](Emotion_Aware_Anti_Spoofing_Speaker_Verification-1.pdf).

> **Status: real-data evaluation complete on a three-generator corpus.**
> The evaluation and fusion layer is implemented, tested, and exercised end to
> end on real audio. Progress beyond the commit history is tracked in
> [docs/PROGRESS_LOG.md](docs/PROGRESS_LOG.md); the stage-by-stage plan is
> [docs/REMAINING_WORK_PLAN.md](docs/REMAINING_WORK_PLAN.md) and the roadmap is
> [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).
>
> Verified so far: 88 tests, the novelty gate closed on the mechanism and open only
> for a published-baseline reproduction (4+1 papers read at recorded depth; two
> held-out-emotion folds run), a 53,900-trial real merged manifest (17,900 ESD bona fide +
> 36,000 EmoSpoof-TTS clones from StyleTTS2, F5-TTS and CosyVoice, `easv audit`
> clean with 0 errors and 4 documented warnings), real ASV (ECAPA-TDNN) and SER
> (wav2vec2-superb-ER)
> producers over every trial, and a working CUDA torch stack
> (`2.14.1+cu130`, RTX 3050 6 GB) built from offline wheels.
>
> **Headline results.** On the held-out test split the ECAPA SV front-end accepts
> 82.9 % of bona fide targets and **38.8 % of clones presented as the cloned
> speaker** — but the rate is generator-specific: CosyVoice 59.0 %, F5-TTS
> 51.5 %, StyleTTS2 5.9 %. Standard fusion removes the vulnerability when the
> countermeasure works: SASV-EER falls from **33.1 % (ASV only) to 0.60 %** on the
> two-generator corpus. The LA-pretrained AASIST does **not** transfer to these
> clones (SPF-EER 50.8 %), while a log-mel CM trained on two generators detects
> the third, unseen one at **2.50 %** EER. On the complete three-generator design
> with that countermeasure the proposed gated arm is the best model
> (**1.54 %** SASV-EER vs 2.27 % linear), and its edge over the interaction
> control (+0.48 pts, CI 0.06–0.99) and the capacity control (+0.85 pts) is
> significant — but its learned gate weights are flat across emotions, so the
> gain cannot be attributed to emotion conditioning, and with an in-domain CM it
> is still 0.14 pts worse than plain fusion. With a failing CM the emotion arms
> gain ~5.5 pts, but SER is confidently wrong on clones (32.8 % accuracy at 0.82
> confidence), so that gain is most likely a spoof cue — and replacing the emotion
> vector with one SER max-probability scalar reproduces the gated arm's whole
> three-generator edge (**1.49 %** vs 1.54 %, p = 0.76). Excluding an entire emotion
> from both countermeasure training and the fusion fit leaves that emotion's error
> rate unchanged to within **0.03 pts**. Three independent measurements therefore
> agree that the gate is not using emotion. Full tables, CIs and caveats:
> [docs/REAL_RESULTS.md](docs/REAL_RESULTS.md).
>
> **Stage status at a glance**
>
> | stage | state |
> |---|---|
> | Evaluation core (manifest/audit/fusion/metrics/CLI, 88 tests) | done |
> | ESD English subset + manifest (17,900 rows) | done |
> | EmoSpoof-TTS: all three generators (36,000 wavs), manifest | done |
> | Merged real manifest + audit | done — 53,900 trials, 0 errors |
> | ASV + SER scores on every trial | done — 53,900 rows each |
> | ASV vulnerability summary | done — [scripts/summarize_real_scores.py](scripts/summarize_real_scores.py), 38.8 % spoof acceptance on test |
> | CM scores (AASIST-LA, in-domain CNN, cross-generator CNN) | done — AASIST 50.8 % (no transfer), in-domain 0.09 %, StyleTTS2-held-out 1.60 % |
> | Real SASV fusion, 5 arms × 4 CM regimes | done — [docs/REAL_RESULTS.md](docs/REAL_RESULTS.md) |
> | Three-generator cross-generator evaluation | done — `results/real/run4_3gen_crossgen` |
> | Paired arm-vs-arm contrasts | done — [scripts/compare_arms.py](scripts/compare_arms.py) |
> | ASVspoof 2019 LA corpus | done — 25,380 train utterances via HF mirror |
> | AASIST retrained on LA (in-domain) | done — [scripts/train_cm_aasist_la.py](scripts/train_cm_aasist_la.py); 2.11 % EER on held-out LA speakers, 43.8 % on EASV test |
> | SER-confidence ablation, held-out-emotion folds | done — [docs/REAL_RESULTS.md](docs/REAL_RESULTS.md) |
> | Figure set | done — five generated figures in [figures/](figures/), from [scripts/make_figures.py](scripts/make_figures.py) |

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q                       # 88 tests, ~4 s
.venv/bin/easv demo --out results/demo --config configs/demo.yaml
```

`easv demo` generates a fully synthetic scenario, runs the audit, fits every
fusion model and writes [results/demo/results.md](results/demo/results.md). The
simulated numbers validate the pipeline only: the demo prints a caveat and the
report is labelled `provenance: synthetic`.

## What the pipeline does

```
manifest.csv ─┐
              ├─► join on trial_id ─► audit ─► fit fusion models (train split)
scores.csv  ──┘                                      │
                                                     ├─► select threshold (dev split)
                                                     └─► report on test split once
```

* **Manifest** = ground truth: which trial, which split, claimed vs true speaker,
  trial type, emotion label, attack system. Validated with explicit error codes.
* **Score file** = system outputs only: `speaker_score` (higher = more like the
  claimed speaker), `cm_score` (higher = more bona fide), optional `emo_<label>`
  probability columns. Ground-truth emotion never enters the model inputs, which
  keeps label leakage visible.
* **Fusion ladder** (each arm is a required ablation, not decoration):

  | model | what it is |
  |---|---|
  | `speaker_only` | ASV score only |
  | `linear` | conventional calibrated speaker + countermeasure fusion (baseline) |
  | `emotion_additive` | same fusion plus emotion features, no conditioning |
  | `emotion_interaction` | linear model with speaker/CM × emotion interactions (capacity control) |
  | `emotion_gated` | **proposed**: gate maps emotion evidence + uncertainty to per-trial non-negative weights on speaker and countermeasure evidence, plus an emotion-conditioned bias |

* **Metrics**: SASV-EER (primary), SV-EER and SPF-EER components, min a-DCF,
  min t-DCF (ASVspoof 2021 and legacy 2019 normalization), FAR/FRR/precision/
  recall/F1 at a dev-selected threshold, Brier score and ECE for calibration, and
  grouped-bootstrap confidence intervals with paired comparisons against the
  baseline. Definitions and sources: [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Commands

```bash
# validate a manifest, report leakage/coverage, exit 2 on errors
easv audit --manifest data_manifests/example.csv --out results/audit

# fit + evaluate; test split is touched once, thresholds come from dev
easv evaluate --manifest data_manifests/example.csv \
              --scores    results/scores.csv \
              --config    configs/sasv2022_template.yaml \
              --out       results/run1 \
              --set       provenance.kind=real

# synthetic end-to-end smoke test
easv demo --out results/demo --config configs/demo.yaml

# ablations: --config is repeatable, later files win, so partial overlays keep the
# base hyperparameters identical across arms (only the one factor changes)
easv evaluate --manifest data_manifests/example.csv --scores results/scores.csv \
              --config configs/demo.yaml \
              --config configs/ablation_no_emotion_uncertainty.yaml \
              --out results/ablation_no_uncertainty

# apply a saved model to new scores (models live in results/<run>/models/*.json)
easv predict --model results/run1/models/emotion_gated.json \
             --manifest data_manifests/example.csv \
             --scores results/scores.csv --split test --out results/run1/predictions.csv

# print the manifest header
easv manifest-template --out data_manifests/example.csv
```

## Repository layout

| path | contents |
|---|---|
| `src/easv/` | `manifest.py`, `scores.py`, `audit.py`, `metrics.py`, `fusion.py`, `evaluation.py`, `demo.py`, `cli.py` |
| `tests/` | unit and end-to-end tests (metrics direction conventions, leakage audits, dev-only threshold selection, serialization, CLI) |
| `configs/` | demo, real-data template, and ablation configurations |
| `data_manifests/` | manifest/schema documentation; real manifests are kept out of version control |
| `results/` | generated reports, per-trial scores and fitted models (git-ignored except `README.md`) |
| `docs/` | implementation plan, protocol, novelty gate, corpus strategy, progress log, real results, remaining-work plan |
| `scripts/` | manifest builders, ASV/SER/CM score producers, score combiner, in-domain and LA CM trainers, arm-vs-arm comparison, corpus fetchers (`hf_fetch.sh`, `extract_hf_corpora.py`), `run_real_pipeline.sh` (end-to-end entry point) |
| `figures/` | the five generated manuscript figures (`.png` + vector `.pdf`) and their regeneration notes |
| `paper/` | manuscript workspace — outline only so far |

## What is implemented vs not

Implemented and tested:

* manifest schema, validation and split/leakage audits (speaker, generator, emotion);
* leakage-safe fitting: fusion parameters and scalers from `train`, thresholds and
  model selection from `dev`, test reported once, with a regression test that
  proves thresholds do not depend on test data;
* metric suite with explicit acceptance conventions and configurable cost models;
* the five-arm fusion ladder including the proposed gated model, with JSON checkpoints;
* grouped and paired bootstrap statistics, per-emotion and per-attack breakdowns,
  gate-weight reporting and arm-vs-arm contrasts;
* real score producers: ECAPA-TDNN ASV, wav2vec2 SER, pretrained AASIST, in-domain
  log-mel CM with leave-one-generator-out training, and AASIST retrained on
  ASVspoof 2019 LA, plus the corpus fetchers and manifest builders;
* CLI, synthetic demo, 88 tests;
* the figure set, generated from stored results rather than hand-drawn
  ([scripts/make_figures.py](scripts/make_figures.py), needs the optional
  `matplotlib` extra).

Not implemented, and known to be open:

* WavLM/HuBERT speaker or countermeasure encoders — the self-supervised part of the
  plan is realised only through the wav2vec2 SER front-end;
* a held-out-emotion evaluation against a *failing* countermeasure: the two folds
  that exist (`angry`, `neutral`) were run in the working-CM regime, where the CM
  itself transfers to the unseen emotion at 0.03 % EER, so they stress the fusion
  stage only (see PROTOCOL section 6 and REAL_RESULTS);
* the two ablation overlays (`ablation_no_emotion_uncertainty.yaml`,
  `ablation_target_far_threshold.yaml`) have been exercised on synthetic data only,
  not on the real corpus;
* reproduction of a published baseline under a matched protocol, which is the
  remaining novelty-gate item and would also validate the metric code against
  reference numbers;
* the ASVspoof 2019 LA evaluation partition (only train was fetched, so the
  retrained CM is validated on two held-out speakers);
* the manuscript ([paper/README.md](paper/README.md)) — the course report is the
  current write-up; the figures it will need now exist in
  [figures/](figures/README.md).

### Real-data stage (complete)

* [x] Offline wheel stack for the audio venv (`.venv-rm`, 69 wheels fetched +
  verified offline install with import checks) — see
  [scripts/fetch_wheels.sh](scripts/fetch_wheels.sh);
* [x] ESD English subset (speakers 0011–0020) fetched via public HF mirror,
  real manifest, `easv audit` clean. **Licence caveat:** a mirror is a
  distribution route, not a grant; ESD's research licence form was never
  submitted ([docs/ESD_LICENCE_EMAIL.md](docs/ESD_LICENCE_EMAIL.md)), so this
  needs to be resolved before publishing beyond the course report;
* [x] Real ASV score producer ([scripts/producer_asv.py](scripts/producer_asv.py),
  ECAPA-TDNN embeddings + cosine) and SER producer
  ([scripts/producer_ser.py](scripts/producer_ser.py), wav2vec2-base-superb-ER →
  `emo_*` softmax columns) — both completed on the full manifest;
* [x] EmoSpoof-TTS downloads, checksum-gated single-stream rescue
  ([scripts/dataset_rescue.sh](scripts/dataset_rescue.sh)) after the segmented
  downloader proved unreliable on the Zenodo-hosted archive; all three
  generators are complete on disk (12,000 wavs each):
  * F5-TTS (1.09 GB) — md5-verified, extracted;
  * CosyVoice (2.67 GB) — md5-verified, extracted;
  * StyleTTS2 — delivered from the public Hugging Face parquet mirror
    ([scripts/hf_fetch.sh](scripts/hf_fetch.sh) +
    [scripts/extract_hf_corpora.py](scripts/extract_hf_corpora.py)) after Zenodo
    truncated the 2.9 GB archive twice; the mirror's `path` column preserves the
    original generator/speaker/emotion tree;
* [x] EmoSpoof manifest (`scripts/build_manifests.py emospoof`, 36,000 rows) and
  merged real manifest with ESD (`scripts/merge_manifests.py`, 53,900 rows);
  `easv audit` → 0 errors, 4 warnings (cross-speaker nontarget rows and all
  generators present in every split);
* [x] Offline CUDA torch for the audio venv: `torch 2.14.1+cu130` imports and
  `torch.cuda.is_available()` is `True` on the RTX 3050, after installing the
  NVIDIA runtime wheels (cublas, cudnn, cuda_runtime, nvrtc, nvjitlink, cufft,
  curand, cusolver, cusparse, nccl, nvtx, cupti, cufile, cusparselt, nvshmem);
* [x] StyleTTS2 generator present, so the three-generator design is complete and
  a leave-one-generator-out split (StyleTTS2 held out) is possible;
* [x] ASVspoof 2019 LA train corpus: 25,380 utterances fetched from the public
  parquet mirror and extracted with the official CM protocols
  ([scripts/extract_hf_corpora.py](scripts/extract_hf_corpora.py));
* [x] AASIST retrained on LA
  ([scripts/train_cm_aasist_la.py](scripts/train_cm_aasist_la.py)): checkpoint
  selected on a speaker-disjoint LA dev split (2.11 % EER on held-out LA
  speakers), then scored once on the EASV manifest (43.8 % test EER), which
  confirms that ASVspoof-2019 supervision does not describe these clones;
* [x] Fusion evaluation on real scores under four countermeasure regimes
  (in-domain, LA-pretrained AASIST, F5-held-out, StyleTTS2-held-out).

## Reproducing the real run

```bash
# corpora must exist: $DATA_ROOT/esd (ESD English) and $DATA_ROOT/emospoof
DATA_ROOT=/home/vct/data/easv CM=indomain bash scripts/run_real_pipeline.sh

# same, but scored by the bundled LA-trained AASIST checkpoint
DATA_ROOT=/home/vct/data/easv CM=aasist OUT=results/real/run_oodcm \
    bash scripts/run_real_pipeline.sh
```

Seven resumable stages: manifests → audit (hard stop on errors) → ASV → SER → CM
→ score combination → fusion evaluation and ASV summary. Results land in
`results/real/<run>/results.md`; measured numbers and caveats are summarised in
[docs/REAL_RESULTS.md](docs/REAL_RESULTS.md).

The **headline three-generator regime** adds one training step (the countermeasure
that never sees StyleTTS2), one score combination and one evaluation on top of that
pipeline; the exact commands are under "Reproduce" in
[docs/REAL_RESULTS.md](docs/REAL_RESULTS.md#reproduce). Evaluated reports and
per-trial scores land in `results/real/`, which is git-ignored because it encodes
corpus paths.

## Research-integrity rules encoded in the code

* No tuning on the test split: thresholds are chosen on `dev`; a unit test asserts
  that replacing test scores leaves every dev-selected threshold unchanged.
* Score direction is fixed and documented once; sign errors are caught by tests
  (including the EER complement behaviour under a score sign flip).
* Breakdowns only report quantities that are definable per group (no EER over
  groups that contain a single class).
* Synthetic results are labelled synthetic in both JSON and Markdown output.

## Environment

Two virtual environments, both real requirements for reproduction:

* `.venv` — Python ≥ 3.10 with `numpy`, `scipy`, `PyYAML`, `pytest` and `python-docx`
  for the analysis package, tests and the report updates. Fusion models train in
  seconds on CPU (L-BFGS for the logistic arms, Adam for the gated arm).
* `.venv-rm` — torch 2.14 + cu130, torchaudio, speechbrain and transformers for the
  audio score producers. `scripts/run_real_pipeline.sh` uses both. A 6 GB GPU was
  enough for every model used here, including AASIST training under bf16 autocast.

No audio, dataset archives or model weights are redistributed in this repository;
the corpora live outside it and the download routes are documented in
[docs/PROGRESS_LOG.md](docs/PROGRESS_LOG.md).

**Licence caveat:** the ESD bona fide audio was fetched from a public Hugging Face
mirror, which is a distribution route rather than a rights grant. The ESD research
licence form was never submitted
([draft](docs/ESD_LICENCE_EMAIL.md)), so this must be resolved before publishing
beyond the course report.
