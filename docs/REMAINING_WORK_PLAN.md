# Work plan and completion status

This file used to be the project's to-do list, written when no audio entered the
repository. Stages 0–5 have since been executed; this version keeps the stage
structure of the original plan and records, per stage, what was actually done,
what the result was, and what remains. Measured numbers live in
[REAL_RESULTS.md](REAL_RESULTS.md); the execution history, including failures and
root causes, is in [PROGRESS_LOG.md](PROGRESS_LOG.md); the remaining methodological
gap is recorded in [PROTOCOL.md](PROTOCOL.md) section 6.

Last verified: 2026-10-26, 99 tests passing.

## Status at a glance

| stage | status | evidence |
|---|---|---|
| 0 — environment | **done** | CUDA torch `2.14.1+cu130` on the RTX 3050, offline wheels, `.venv` for analysis and `.venv-rm` for audio models |
| 1 — prior art / novelty gate | **closed on the mechanism, one item open** | four closest papers read, matrix filled; held-out-emotion item now done; published-baseline reproduction still open, declared a limitation |
| 2 — datasets, licences, manifests | **done, with one compliance gap** | 53,900-trial manifest, audit 0 errors / 4 expected warnings; ESD licence never granted |
| 3 — score producers | **done, one validation gap** | ASV, SER and four countermeasure regimes scored on every trial; no published-baseline reproduction |
| 4 — pre-register success criteria | **not done** | [PROTOCOL.md](PROTOCOL.md) §6 now records the rules applied and marks the missing pre-registration |
| 5 — real-data evaluation | **done** | four CM regimes, five arms, grouped and paired bootstrap, per-emotion and per-attack tables |
| 6 — ablations and depth | **done, except the items that need other data** | gate-weight introspection, group breakdowns, SER-confidence ablation, uncertainty-removed overlay, 1 %-FAR operating point and four held-out-emotion folds (two emotions × two countermeasure regimes) all measured on the real corpus; seed repeats done for the two decisive regimes. Not done because they need data the project does not have: a second speaker split, frozen-vs-fine-tuned SER, and a capacity-matched gate |
| 7 — paper writing | **complete draft; numbers still the pilot's** | [paper/manuscript.md](../paper/manuscript.md) with figures placed and citations verified; the course report [Course_Project_Report_EASV_Audited.docx](../Course_Project_Report_EASV_Audited.docx) is the submitted write-up; the journal version re-runs the pipeline on CREMA-D |

## Stage 0 — Environment

Done. The GPU was reported broken at plan time (userspace/kernel driver mismatch);
a reboot resolved it, and `torch 2.14.1+cu130` imports with `torch.cuda.is_available()
== True`. The CUDA runtime libraries had to be installed separately from the same
offline wheel directory (cublas, cudnn, cuda_runtime, nvrtc, nvjitlink, cufft,
curand, cusolver, cusparse, nccl, nvtx, cupti, cufile, cusparselt, nvshmem); the
detail is in [PROGRESS_LOG.md](PROGRESS_LOG.md). Two environments are used:
`.venv` (numpy/scipy/pyyaml/pytest/docx) for everything in `src/easv/`, and
`.venv-rm` (torch, speechbrain, transformers) for the audio producers. The split is
real and matters for reproduction: `scripts/run_real_pipeline.sh` calls both.

## Stage 1 — Prior-art verification and novelty gate

Conditionally closed on 2026-10-21. The four closest papers were read at the depth
recorded in [NOVELTY_MATRIX.md](NOVELTY_MATRIX.md) — Sensors 2025 (fixed-α emotion
score fusion), Interspeech 2025 GEM (emotion-gated ensemble of anti-spoofing models,
no speaker verification), ICASSP 2022 (SER features into a deepfake detector) and
GCCE 2023 (sequential emotion-then-speaker gate) — plus the multi-task spoofing-robust
ASV paper (Zhao et al. 2022) to check whether an integrated model already covers the
proposal. It does not: it is one network with task heads and no emotion signal.

Closed since: the held-out-emotion evaluation (2026-10-25 — `angry` and `neutral`
excluded from countermeasure training and from the fusion fit; ≤0.03 points change on
the held-out emotion, every paired CI spanning zero) and the SER-confidence ablation
(a single SER max-probability scalar reproduces the gated arm's Regime D edge), so the
mechanism claim is withdrawn rather than pending.

Still open, and it is verification rather than reading:

* reproduction of a strong published baseline under a matched protocol, which would
  also validate the metric code against reference numbers. For a workshop submission
  this is declared as a limitation instead of blocking the paper.

The proposed method's row in the matrix reports measured behaviour (it was executed
on 53,900 real trials), including the finding that its gate does not vary with
emotion. No "first" or "novel" wording is permitted anywhere in the project.

## Stage 2 — Datasets, licences, manifests

Done, with one compliance gap that is not technical.

Both origin servers capped this link at ~200 KB/s, so the corpora came from public
Hugging Face parquet mirrors instead, at 1.2–2 MB/s per connection and ~2 MB/s
aggregate with four parallel streams ([scripts/hf_fetch.sh](../scripts/hf_fetch.sh),
[scripts/extract_hf_corpora.py](../scripts/extract_hf_corpora.py)):

| corpus | obtained | notes |
|---|---|---|
| ESD English, speakers 0011–0020 | 17,900 trials | 17,500 targets (10 speakers × 5 emotions × 350) + 400 derived cross-speaker nontargets |
| EmoSpoof-TTS StyleTTS2 / F5-TTS / CosyVoice | 36,000 clones (12,000 each) | emotion label from the generator's folder tree; `path` column preserved by the mirror |
| ASVspoof 2019 LA | 25,380 train utterances + official CM protocols | train partition only; the evaluation partition was never fetched |

The merged manifest holds **53,900 trials** on a 4/2/4 speaker split (train 21,560 /
dev 10,780 / test 21,560). `easv audit` reports **0 errors and 4 warnings**: two
speaker-overlap and two generator-overlap findings, both expected consequences of
the merge policy (cross-speaker nontarget rows, and every generator present in every
split). The three-generator design is what makes a generator-held-out countermeasure
possible; the split discipline for it lives in the CM training script, not in the
manifest.

Two things were deliberately not obtained: the ASVspoof 2019 evaluation partition
(the EASV corpus is the target domain and its dev split selects everything) and the
SASV 2022 protocols/SASV challenge audio (no baseline reproduction). Both are listed
as open items, because they close different gaps.

**Compliance gap:** ESD is released under a research licence that requires a signed
form. Fetching the audio from a mirror does not grant rights to it. The request draft
exists at [ESD_LICENCE_EMAIL.md](ESD_LICENCE_EMAIL.md) and still contains bracketed
placeholders, so there is no evidence it was ever sent; the licence form and its terms
(including whether derived score files may be shared) must be settled before anything
beyond the course report is published. There is also no
corpus checksum/version table yet; the Zenodo md5s used during the download campaign
are recorded in [PROGRESS_LOG.md](PROGRESS_LOG.md), but the mirror shards are not
pinned.

## Stage 3 — Score producers

Done for four countermeasure regimes; the published-baseline check was not done.

| producer | model | output |
|---|---|---|
| [scripts/producer_asv.py](../scripts/producer_asv.py) | SpeechBrain `spkrec-ecapa-voxceleb`, cosine to the train-split centroid | `speaker_score` for 53,900 trials; SV-EER 3.12 % |
| [scripts/producer_ser.py](../scripts/producer_ser.py) | `superb/wav2vec2-base-superb-er`, 4-way softmax | `emo_*` for 53,900 trials; 41.2 % accuracy on bona fide, 32.8 % on clones |
| [scripts/producer_cm_aasist.py](../scripts/producer_cm_aasist.py) | published AASIST 2019-LA checkpoint | SPF-EER 50.8 % (no transfer) |
| [scripts/train_cm_indomain.py](../scripts/train_cm_indomain.py) | log-mel CNN, train split only, optional `--holdout-attack` | 0.09 % in-domain; 6.27 % on held-out F5-TTS; 2.50 % on held-out StyleTTS2 |
| [scripts/train_cm_aasist_la.py](../scripts/train_cm_aasist_la.py) | AASIST retrained on ASVspoof 2019 LA train | 2.11 % on held-out LA speakers, 43.8 % on EASV test |

Not done: score calibration (temperature fitted per producer on dev), SER
fine-tuning on the EmoSpoof emotion taxonomy, and any check that these scores
reproduce a published baseline. The last one is the item that would catch a metric
implementation error, and it is the main reason the novelty gate stays open.

## Stage 4 — Pre-registered success criteria

**Not done, and it cannot be recovered.** Metric definitions, split rules and the
comparison baseline were fixed before the first test report, but no numeric gain
margin, must-hold track or FRR margin was written down. [PROTOCOL.md](PROTOCOL.md) §6
now records the rules that were applied and labels the missing ones as not
pre-registered, including the uncomfortable consequence: the StyleTTS2 generalisation
track was emphasised after its numbers were known.

Any future write-up must freeze the decision table first, and must not describe the
existing results as confirming a pre-specified hypothesis.

## Stage 5 — Real-data evaluation

Done. The full ladder was evaluated under four countermeasure regimes (in-domain,
LA-pretrained AASIST, F5-held-out, StyleTTS2-held-out), with thresholds from dev only
and the test split reported once. Outputs are in `results/real/` (git-ignored, since
they encode corpus paths) and summarised in [REAL_RESULTS.md](REAL_RESULTS.md).

Headline outcomes:

* an off-the-shelf ECAPA ASV accepts 82.9 % of bona fide targets and 38.8 % of
  clones, generator-dependent (CosyVoice 59.0 %, F5-TTS 51.5 %, StyleTTS2 5.9 %);
* fusion cuts SASV-EER from 33.1 % to 0.60 % when the countermeasure works, and the
  proposed gated arm is 0.14 pts *worse* there (p = 0.98);
* with StyleTTS2 held out, the gated arm is the best model (1.54 % vs 2.27 %), and
  its +0.48 pts over the interaction control has an interval excluding zero — but its
  gate weights are flat across emotions, so the mechanism is not what produces the
  gain;
* ASVspoof-2019 supervision does not transfer, pretrained (50.8 %) or retrained
  in-domain (43.8 %), while a log-mel CM trained on two adjacent generators detects
  the third at 2.50 %.

Not done in this stage, after the follow-up runs: a second speaker split (needs
the CREMA-D re-anchoring). The two ablation overlays, the seed repeats and the
held-out-emotion folds against a failing countermeasure were added later — see
"What remains" below.

## Stage 6 — Ablations and depth

Done: per-emotion and per-attack breakdowns, gate-weight introspection by emotion,
paired arm-vs-arm contrasts, the four countermeasure regimes (themselves a
generalisation ablation), the SER-confidence ablation (`ablation_ser_confidence.yaml`,
regimes B and D), the `angry`/`neutral` held-out-emotion folds, both remaining
overlays on the real corpus (`ablation_no_emotion_uncertainty.yaml`,
`ablation_target_far_threshold.yaml`), and four-seed repeats of the two decisive
regimes.

Not done, and not cheap: a capacity-matched gate that would separate "extra
parameters" from "conditioning" (the interaction arm is the closest available
control), frozen vs fine-tuned SER, and an alternative emotion taxonomy. The
held-out-emotion fold against a failing countermeasure was added on 2026-10-25
(Regime B, `angry` and `neutral`, ≤0.15 points movement with every CI spanning
zero), so the fold result now holds in both regimes.

## Stage 7 — Paper writing and submission

A complete manuscript draft exists ([paper/manuscript.md](../paper/manuscript.md)) with
the five generated figures ([figures/README.md](../figures/README.md)) placed in the
text and every citation metadata-checked against publisher, ISCA, ACL Anthology,
NeurIPS and Crossref records. `figures/` is produced by
[scripts/make_figures.py](../scripts/make_figures.py) from stored results rather than
edited by hand; the manuscript text itself is still written by hand, so a changed
number has to be propagated to it deliberately — which is why every table carries its
source run. What remains is the CREMA-D re-run that replaces the pilot numbers — as
of 2026-10-26 its audio and bona fide manifest exist and its clone plan is costed,
but no clone has been synthesised; the course report is the submitted write-up.

## What remains, in priority order

| # | item | why it matters | rough cost |
|---|---|---|---|
| 1 | ~~Run the two ablation overlays on the 53,900-trial corpus~~ — **done 2026-10-25** (`results/real/abl_unc_{A,D}`, `abl_far_{A,D}`) | both plan-required ablations are now measured on real data: uncertainty features are inert (≤ 0.03 pts), and the FAR-budget operating point keeps the arm ordering | done (minutes) |
| 2 | ~~Held-out-emotion split~~ — **done 2026-10-25** (`angry`, `neutral`), **repeated against the failing Regime B CM the same day** (`run6_foldB_*`) | closed the "unseen emotion" objection in both countermeasure regimes; the null result is itself a finding | done (~1 day incl. two CM retrains; the Regime-B repeat took minutes) |
| 3 | ~~Repeat the fusion fits on seeds~~ — **done 2026-10-25** (four seeds, regimes A and D). A second speaker split still needs the CREMA-D re-anchoring (91 actors vs 10) | the ordering is stable across seeds; only the gated arm is seed-sensitive, because the other three arms are fitted by convex procedures | done (minutes per run) |
| 4 | Reproduce a published baseline under a matched protocol | validates metric code; closes the last novelty-gate item | days |
| 5 | ESD licence form + corpus checksum table | publication/compliance blocker, not technical | owner action |
| 6 | Longer LA training + the LA eval partition | the retrained CM is validated on 2 speakers, 3 epochs | hours + 7.8 GB |
| 7 | **Deliberately not run** (2026-10-25) — a fusion run for the LA-retrained CM | it would be a fifth copy of Regime B: that CM is at chance on these clones (43.8 % test EER, 49.4 % dev), so the ladder collapses to the failing-countermeasure case already reported. The per-split CM numbers that would be new are in [REAL_RESULTS.md](REAL_RESULTS.md) (2.11 % on held-out LA speakers, 49.4 % dev and 43.8 % test on EASV); a fusion run adds no information, and adding one would invite the reader to compare a chance-level arm against a working one | not done, by decision |
| 8 | ~~Measure per-generator speaker-identity recovery~~ — **done 2026-10-25** (`scripts/asv_identity_recovery.py`, `results/real/identity_recovery/`) | explains the StyleTTS2 anomaly instead of inferring it: 60.0 % of its clones still identify as the cloned speaker against 91–93 % for the other two generators, with mean true-speaker cosine +0.256 vs +0.430/+0.455; the identity ordering is the acceptance ordering | done (~4 min on the RTX 3050) |
| 9 | ~~Encode the headline regime in `run_real_pipeline.sh`; LICENSE file; CI~~ — **done 2026-10-25** | `CM=crossgen` now builds the regime-D countermeasure the paper reports; [LICENSE](../LICENSE) (MIT for code, corpus terms restated); [.github/workflows/tests.yml](../.github/workflows/tests.yml) runs the corpus-free suite on 3.10 and 3.12 | done |

## Decisions taken (were open at plan time)

| decision | taken | consequence |
|---|---|---|
| ASV baseline | pretrained ECAPA-TDNN, cosine to train centroid | strong SV on bona fide speech (3.12 % SV-EER); not the SASV 2022 baseline |
| Countermeasure | log-mel CNN trained in-domain, with leave-one-generator-out variants; AASIST for the transfer control | detects held-out generators; deliberately simple, so no claim about CM architecture |
| Emotion model | pretrained wav2vec2-superb-ER, 4 classes, no fine-tuning | confidently wrong on clones (32.8 % accuracy at 0.82 confidence), which becomes the project's main confound |
| Emotion taxonomy | 4 categorical classes from ESD/EmoSpoof folders | `surprise` exists in ESD but has no synthetic counterpart, so it is reported but never a spoof condition |
| Primary generalisation track | cross-generator, with StyleTTS2 held out | chose the track that stresses the countermeasure; recorded as post hoc in PROTOCOL §6 |
