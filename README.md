# Emotion-Conditioned Spoofing-Aware Speaker Verification (EASV)

Research codebase for the question: **does emotion-conditioned fusion of speaker and
anti-spoofing evidence improve spoofing-aware speaker verification under
cross-emotion and cross-generator evaluation, without unacceptably increasing
rejection of genuine speech?**

The plan this repo implements is [Emotion_Aware_Speaker_Verification_Research_Plan.md](Emotion_Aware_Speaker_Verification_Research_Plan.md);
the literature-review design is in [Emotion_Aware_Anti_Spoofing_Speaker_Verification-1.pdf](Emotion_Aware_Anti_Spoofing_Speaker_Verification-1.pdf).

> **Status: research infrastructure — real-data stage started, not finished.**
> The evaluation and fusion layer is implemented, tested, and now exercised on
> real data. Progress beyond the commit history is tracked in
> [docs/PROGRESS_LOG.md](docs/PROGRESS_LOG.md); the stage-by-stage plan is
> [docs/REMAINING_WORK_PLAN.md](docs/REMAINING_WORK_PLAN.md) and the roadmap is
> [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md). No claim in this
> repository about emotion-conditioned gains on real speech is supported yet.
>
> Verified so far: 66 tests, novelty gate closed (conditional GO, 4+1 papers read
> at recorded depth), a 41,900-row real merged manifest
> (17,900 ESD + 24,000 EmoSpoof, `easv audit` clean with 0 errors), real ASV
> (ECAPA-TDNN) and SER (wav2vec2-superb-ER) producers run over all of ESD, and a
> working CUDA torch stack (`2.14.1+cu130`, RTX 3050 6 GB) built from offline
> wheels.
>
> **First real results.** On the held-out test split the ECAPA SV front-end
> accepts 82.9 % of bona fide targets but also **55.2 % of EmoSpoof-TTS clones**
> presented as the cloned speaker, emotion-dependent (angry 64.6 %, neutral
> 61.7 %, sad 47.5 %, happy 47.2 %; per generator CosyVoice 59.0 %, F5-TTS
> 51.5 %). Standard fusion removes that vulnerability when the countermeasure
> works: SASV-EER falls from **33.1 % (ASV only) to 0.60 % (linear fusion)**.
> The LA-pretrained AASIST does **not** transfer to these clones (SPF-EER
> 50.8 %), and with an in-domain CM no emotion-conditioned arm beats plain
> fusion (gated arm is 0.14 pts worse, p=0.98). With the failing CM the emotion
> arms gain ~5.5 pts, but SER is confidently wrong on clones (30.7 % accuracy at
> 0.83 confidence), so that gain is most likely a spoof cue, not emotion
> conditioning. Full tables, CIs and caveats: [docs/REAL_RESULTS.md](docs/REAL_RESULTS.md).
>
> Still unfinished: the StyleTTS2 generator of EmoSpoof-TTS (re-fetch running),
> ASVspoof 2019 LA extraction (blocked on disk space, not on code), AASIST CM
> training, and the first real fusion evaluation — the latter two need CM scores,
> which need LA.
>
> **Stage status at a glance**
>
> | stage | state |
> |---|---|
> | Evaluation core (manifest/audit/fusion/metrics/CLI, 66 tests) | done |
> | ESD English subset + manifest (17,900 rows) | done |
> | EmoSpoof-TTS: F5-TTS + CosyVoice (24,000 wavs), manifest | done |
> | EmoSpoof-TTS: StyleTTS2 generator | missing on disk — re-fetch pending |
> | Merged real manifest + audit | done (0 errors, 4 warnings) |
> | ASV scores on ESD (17,660 trials) | done |
> | SER scores on ESD (17,660 trials) | done |
> | ASV scores on the merged (ESD + spoof) manifest | done — 41,900 trials |
> | Real ASV summary on the merged manifest | done — [scripts/summarize_real_scores.py](scripts/summarize_real_scores.py), 55.2 % spoof acceptance on test |
> | CM scores (AASIST-LA, in-domain CNN) | done — 41,900 trials each; AASIST 50.8 % (no transfer), in-domain CNN 0.09 % |
> | SER scores on the merged manifest | done — 41,900 trials |
> | **Real SASV fusion evaluation, 5 arms, both CM regimes** | done — [docs/REAL_RESULTS.md](docs/REAL_RESULTS.md) |
> | ASVspoof 2019 LA | blocked — inconsistent parts removed under ENOSPC; extraction needs ~7.5 GB free |
> | EmoSpoof StyleTTS2 generator | re-fetching (third generator, cross-generator test) |

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q                       # 65 tests, ~3 s
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
| `docs/` | implementation plan, protocol, novelty gate, progress log, remaining-work plan |
| `paper/`, `figures/` | placeholders for the manuscript and figures |

## What is implemented vs not

Implemented and tested:

* manifest schema, validation and split/leakage audits (speaker, generator, emotion);
* leakage-safe fitting: fusion parameters and scalers from `train`, thresholds and
  model selection from `dev`, test reported once, with a regression test that
  proves thresholds do not depend on test data;
* metric suite with explicit acceptance conventions and configurable cost models;
* the proposed gated fusion model plus its ablation arms, with JSON checkpoints;
* grouped and paired bootstrap statistics; per-emotion and per-attack breakdowns;
* CLI, synthetic demo, 65 tests.

Not implemented (deliberately, see the plan's novelty gate):

* WavLM/HuBERT front-ends and ECAPA-TDNN/AASIST training — the pipeline consumes
  scores, so encoders plug in as score producers without touching this code;
* real dataset ingestion (ASVspoof 2019/2021, SASV 2022, EmoSpoof-TTS, RAVDESS/CREMA-D);
* prior-art verification and the novelty matrix (partially done — 4+1 papers
  read and recorded at [docs/NOVELTY_MATRIX.md](docs/NOVELTY_MATRIX.md); conditional GO);
* the real-dataset stage of the pipeline (see [docs/PROGRESS_LOG.md](docs/PROGRESS_LOG.md));
* the manuscript.

### Real-data stage (in progress)

* [x] Offline wheel stack for the audio venv (`.venv-rm`, 69 wheels fetched +
  verified offline install with import checks) — see
  [scripts/fetch_wheels.sh](scripts/fetch_wheels.sh);
* [x] ESD English subset (speakers 0011–0020) fetched via public HF mirror,
  17,660-trial real manifest, `easv audit` clean;
* [x] Real ASV score producer ([scripts/producer_asv.py](scripts/producer_asv.py),
  ECAPA-TDNN embeddings + cosine) and SER producer
  ([scripts/producer_ser.py](scripts/producer_ser.py), wav2vec2-base-superb-ER →
  `emo_*` softmax columns) — both completed on the full manifest;
* [x] EmoSpoof-TTS downloads, checksum-gated single-stream rescue
  ([scripts/dataset_rescue.sh](scripts/dataset_rescue.sh)) after the segmented
  downloader proved unreliable on the Zenodo-hosted archive:
  * StyleTTS2 (2.93 GB) — md5-verified, CRC-valid, extracted; the extracted
    directory did **not** survive on disk and is missing (re-fetch pending);
  * F5-TTS (1.09 GB) — md5-verified, extracted, 12,000 wavs;
  * CosyVoice (2.67 GB) — md5-verified, extracted, 12,000 wavs;
* [x] EmoSpoof manifest (`scripts/build_manifests.py emospoof`, 24,000 rows) and
  merged real manifest with ESD (`scripts/merge_manifests.py`, 41,900 rows);
  `easv audit` → 0 errors, 4 warnings (cross-speaker nontarget rows and both
  available generators present in every split);
* [x] Offline CUDA torch for the audio venv: `torch 2.14.1+cu130` imports and
  `torch.cuda.is_available()` is `True` on the RTX 3050, after installing the
  NVIDIA runtime wheels (cublas, cudnn, cuda_runtime, nvrtc, nvjitlink, cufft,
  curand, cusolver, cusparse, nccl, nvtx, cupti, cufile, cusparselt, nvshmem);
* [ ] StyleTTS2 generator re-fetch to complete the three-generator design —
  running; the earlier tree was deleted by the rescue loop's unconditional
  `rm -rf` after *every* archive, now fixed and guarded by an extracted-tree
  check;
* [ ] ASVspoof 2019 LA download → selective train/dev/protocols extract
  (download 6.5/7.6 GB complete; extraction needs ~7.5 GB free and the volume has
  1.8 GB — disk, not code, is the blocker);
* [ ] AASIST CM training ([scripts/EASIST_LA.conf](scripts/EASIST_LA.conf)) once LA audio lands;
* [ ] fusion evaluation on real scores (needs the CM scores above).

## Research-integrity rules encoded in the code

* No tuning on the test split: thresholds are chosen on `dev`; a unit test asserts
  that replacing test scores leaves every dev-selected threshold unchanged.
* Score direction is fixed and documented once; sign errors are caught by tests
  (including the EER complement behaviour under a score sign flip).
* Breakdowns only report quantities that are definable per group (no EER over
  groups that contain a single class).
* Synthetic results are labelled synthetic in both JSON and Markdown output.

## Environment

Python ≥ 3.10 with `numpy`, `scipy`, `PyYAML` (and `pytest` for tests). Models are
trained with L-BFGS (logistic arms) and Adam (gated arm) on CPU; the demo runs in
seconds. No GPU or audio dependencies are required yet, and no dataset or model
weights are redistributed here.
