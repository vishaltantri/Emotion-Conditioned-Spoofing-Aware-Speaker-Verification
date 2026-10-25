# Implementation plan and status

Derived from [Emotion_Aware_Speaker_Verification_Research_Plan.md](../Emotion_Aware_Speaker_Verification_Research_Plan.md)
(§7 phases, §8 timeline, §9 artifacts). This file states, for every phase, what was
implemented, what was executed, and what is still missing. Numbers are in
[REAL_RESULTS.md](REAL_RESULTS.md); the execution history is in
[PROGRESS_LOG.md](PROGRESS_LOG.md); the stage-by-stage work plan is
[REMAINING_WORK_PLAN.md](REMAINING_WORK_PLAN.md).

Last verified 2026-10-25: 88 tests passing, four countermeasure regimes, one
SER-confidence ablation and two held-out-emotion folds evaluated on 53,900 real
trials.

## Phase 1 — Prior-art verification and novelty gate — **conditionally closed**

Done: the four closest papers were read at the depth recorded in
[NOVELTY_MATRIX.md](NOVELTY_MATRIX.md) (Sensors 2025, Interspeech 2025 GEM, ICASSP
2022, GCCE 2023), the multi-task spoofing-robust ASV paper was verified as an
integrated model without emotion, the matrix is filled, and a conditional GO was
recorded on 2026-10-21 with the method-distinction paragraph.

Done since: the held-out-emotion evaluation (2026-10-25, folds on `angry` and
`neutral`; the emotion is excluded from CM training and from the fusion fit, and the
held-out test trials show ≤0.03 points change with every paired CI spanning zero).

Still open: reproduction of a strong published baseline under a matched protocol
(which would also validate the metric code against reference numbers). Until it
closes, nothing in this repository may be described as novel, first, or state of the
art. The proposed method's matrix row reports measured behaviour, including the
finding that its learned gate does not vary with emotion — and the held-out-emotion
fold now explains why: there was almost no emotion-specific structure to lose.

## Phase 2 — Dataset and protocol definition — **done, except licences and pre-registration**

Done: [PROTOCOL.md](PROTOCOL.md) fixes trial types, score conventions, split
discipline and metric definitions; the manifest schema and audit enforce them; the
real manifests exist for 53,900 trials (17,900 ESD bona fide + 36,000 EmoSpoof-TTS
clones from three generators) and 25,380 ASVspoof 2019 LA train utterances, with
speaker-disjoint splits and generator labels. `easv audit` reports 0 errors and 4
documented warnings.

Not done: ESD's research licence (the form was never submitted; a public mirror is a
distribution route, not a rights grant), a corpus checksum/version table, the
ASVspoof 2019 evaluation partition, and SASV 2022 protocols. The pre-registered
success-criterion table in PROTOCOL §6 was completed only after test results were
known, so it is recorded as post hoc.

## Phase 3 — Baseline reproduction — **score producers done, published baseline not reproduced**

Producers written and run over every trial: ECAPA-TDNN ASV
([producer_asv.py](../scripts/producer_asv.py)), wav2vec2-superb-ER SER
([producer_ser.py](../scripts/producer_ser.py)), the published AASIST 2019-LA
countermeasure ([producer_cm_aasist.py](../scripts/producer_cm_aasist.py)), an
in-domain log-mel countermeasure with leave-one-generator-out training
([train_cm_indomain.py](../scripts/train_cm_indomain.py)) and an AASIST retrained on
LA ([train_cm_aasist_la.py](../scripts/train_cm_aasist_la.py)). Measured component
quality: SV-EER 3.12 %, SER 41.2 % on bona fide / 32.8 % on clones, CM 0.09 %
in-domain, 6.27 % on held-out F5-TTS, 2.50 % on held-out StyleTTS2, 50.8 % for the
LA-pretrained AASIST and 43.8 % for the LA-retrained one on EASV.

Not done: reproducing a published benchmark (SASV 2022 or ASVspoof) under a matched
protocol. The metric implementations are validated by unit tests with known
closed-form cases, not against reference score files, so a systematic convention
error cannot be ruled out. Score calibration (temperature per producer on dev) and
SER fine-tuning were also skipped.

## Phase 4 — Proposed fusion method — **implemented and executed on real data**

`src/easv/fusion.py` implements the ladder:

* `linear` — conventional calibrated speaker + CM fusion, the primary baseline;
* `emotion_additive` — emotion features, no conditioning (capacity control);
* `emotion_interaction` — speaker/CM × emotion interactions (capacity control);
* `emotion_gated` — the proposal: `score = w_s(φ)·s + w_c(φ)·c + b(φ)` with `φ` the
  emotion probabilities, normalised entropy and confidence, non-negative gate
  weights, L2-regularised Adam, inner validation split for early stopping, and JSON
  checkpoints;
* `use_uncertainty: false` — the "remove emotion uncertainty" switch;
* gate-weight introspection by emotion (`model.gate_weights`), which is what showed
  the gate is flat across emotions.

Not done: frozen vs fine-tuned representation ablation, shared-SSL-encoder
comparison (RQ6), and a capacity-matched gate that would separate "extra parameters"
from "conditioning".

## Phase 5 — Evaluation and analysis — **done for the planned regimes except cross-emotion holdout**

Implemented and executed: in-domain, two-generator and three-generator evaluation;
cross-generator by holding one generator out of countermeasure training; per-emotion
and per-attack breakdowns with quantities that are well defined per group;
bona-fide acceptance at a dev-selected threshold; grouped and paired bootstrap
(1,000 resamples, speakers as the unit); arm-vs-arm contrasts
([compare_arms.py](../scripts/compare_arms.py)); persisted per-trial scores for error
analysis.

Not done: a held-out-emotion fold against a *failing* countermeasure (the two folds
that exist, `angry` and `neutral`, run in the working-CM regime and so stress the
fusion stage only), a cross-dataset track, repeated seeds, and an alternate speaker
split. The held-out-emotion splits themselves are done: `folds.holdout_emotion` in
the evaluation config plus `--holdout-emotion` in the CM trainer, with the test split
always left intact. The test split was evaluated once per regime, as required.

## Phase 6 — Paper writing — **not started**

`paper/` holds the section outline only. The figure set it needs is done:
five figures in [figures/](../figures/README.md) generated from stored results by
[scripts/make_figures.py](../scripts/make_figures.py). The course report
[Course_Project_Report_EASV_Audited.docx](../Course_Project_Report_EASV_Audited.docx)
is the current write-up and carries the real results, the four regimes, the
mechanism caveat and the documented limitations. The manuscript must not be drafted
as a contribution until Phase 1's remaining items close.

## Phase 7 — Submission readiness — **not started**

No target venue has been chosen, no citation/DOI audit has been done for a
submission, and the corpus licences are unresolved. See [paper/README.md](../paper/README.md).

## Decisions taken (these were open questions in the original plan)

| decision | taken | consequence |
|---|---|---|
| ASV baseline | pretrained ECAPA-TDNN, cosine to the train-split centroid | strong on bona fide speech (3.12 % SV-EER), not the SASV 2022 baseline, so numbers are not directly comparable to that challenge |
| CM baseline | log-mel CNN trained in-domain, with leave-one-generator-out variants; AASIST (pretrained and LA-retrained) as transfer controls | detects unseen generators at 2.50 % EER; the simple architecture means the project makes no claim about countermeasure design |
| Emotion model | pretrained wav2vec2-superb-ER, 4 classes, no fine-tuning | 32.8 % accuracy on clones at 0.82 confidence — confidently wrong, which is the project's main confound |
| Emotion taxonomy | 4 categorical classes shared by ESD and EmoSpoof | ESD's `surprise` has no synthetic counterpart and is reported as a bona fide-only condition |
| Primary generalisation track | cross-generator with StyleTTS2 held out | stresses the countermeasure; recorded as post hoc in PROTOCOL §6, so it cannot be called the pre-specified track |

## What to do next

1. Run the two ablation overlays on the real corpus (uncertainty removed; 1 % FAR
   budget) — the only plan-required ablations still missing.
2. ~~Build a held-out-emotion split and evaluate it~~ — done 2026-10-25 for `angry`
   and `neutral` (null result; see [REAL_RESULTS.md](REAL_RESULTS.md)). Optional
   extension: repeat a fold against the failing AASIST-LA countermeasure.
3. Repeat the fits across seeds and a second speaker split before describing the arm
   ordering as stable.
4. Reproduce one published baseline to validate the metric code and close the novelty
   gate.
5. Settle the ESD licence and pin corpus checksums before publishing anything beyond
   the course report.
