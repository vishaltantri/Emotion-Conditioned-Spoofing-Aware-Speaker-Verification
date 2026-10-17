# Implementation plan and status

Derived from [Emotion_Aware_Speaker_Verification_Research_Plan.md](../Emotion_Aware_Speaker_Verification_Research_Plan.md)
(§7 phases, §8 timeline, §9 artifacts). Status is deliberately blunt: this file is
the place where the project says what is *not* done.

## Phase 1 — Prior-art verification and novelty gate — **OPEN (blocking)**

The plan requires this before substantial implementation, and it is not finished.
Done: a research gate document with the search strings, the comparison matrix
schema and the go/no-go checklist ([NOVELTY_MATRIX.md](NOVELTY_MATRIX.md)).
Not done: reading the two 2025 papers in full, filling the matrix, writing the
method-distinction paragraph, and recording the go/no-go decision.

Consequence: nothing in this repository may be described as novel or first. The
code below is infrastructure that would be needed *if* the gate passes.

## Phase 2 — Dataset and protocol definition — **PARTIAL**

Done: protocol fixed in [PROTOCOL.md](PROTOCOL.md); manifest schema with required
trial types, emotion labels and attack systems; leakage audits for speaker,
generator and emotion; manifest/scores separation that keeps evaluation labels out
of the model inputs; documented thresholds and cost models.

Not done: dataset access checks and licences, the real label maps (bona fide /
spoof, speaker, attack, emotion), split manifests for ASVspoof 2019/2021, SASV 2022,
EmoSpoof-TTS and an emotion corpus, and the pre-registered success-criterion table
in §6 of the protocol.

Next actions: see [../data_manifests/README.md](../data_manifests/README.md) for
the ingestion checklist.

## Phase 3 — Baseline reproduction — **NOT STARTED (interface ready)**

The score consumer side exists: `speaker_score` + `cm_score` per trial, component
EERs, min t-DCF for raw CM, and threshold selection. Missing: the score *producers*
— ECAPA-TDNN or equivalent ASV embeddings, AASIST or a reproduced ASVspoof CM, and
validation that their scores reproduce published baselines on official partitions.

Because the pipeline is score-based, an encoder can be added as a separate script
that writes `scores.csv`; no change to `src/easv/` is required.

## Phase 4 — Proposed fusion method — **IMPLEMENTED (awaiting real scores)**

Implemented in `src/easv/fusion.py`:

* `linear` — conventional calibrated speaker + CM fusion, the primary baseline;
* `emotion_additive` — emotion features added with no conditioning (ablation 4);
* `emotion_interaction` — linear model with speaker/CM × emotion interactions
  (capacity control that tests whether extra parameters, not conditioning, explain
  any gain);
* `emotion_gated` — the proposed model: `score = w_s(φ)·s + w_c(φ)·c + b(φ)` with
  `φ` the emotion representation (probabilities, normalised entropy, confidence),
  non-negative weights from a tanh gate, L2-regularised Adam training, inner
  validation split for early stopping, and JSON checkpoints;
* `use_uncertainty: false` config switch — the "remove emotion uncertainty" ablation;
* gate-weight introspection by emotion (`model.gate_weights`) so the mechanism is
  inspectable rather than asserted.

Not done: frozen vs fine-tuned representation ablation, shared-SSL-encoder
comparison (RQ6), and any training on real data.

## Phase 5 — Evaluation and analysis — **IMPLEMENTED for score-level experiments**

Implemented: in-domain track, cross-generator and cross-emotion tracks via the
audit plus split naming, bona-fide acceptance reporting (FAR/FRR at the dev
threshold, per emotion), per-attack-family breakdowns using quantities that are
well defined per group (spoof-acceptance rate rather than a bogus EER), grouped and
paired bootstrap statistics, per-trial score dumps for error analysis.

Not done: cross-dataset track (Track D) and the real statistical comparison on
published baselines.

## Phase 6 — Paper writing — **NOT STARTED**

`paper/` holds a placeholder with the section outline from the literature-review
PDF. Nothing may be drafted as a result until Phase 1 closes and real scores exist.

## Phase 7 — Submission readiness — **NOT STARTED**

## Suggested immediate order of work

1. Close the novelty gate (Phase 1) — it can change what is worth implementing.
2. Confirm dataset access and build one real manifest with emotion labels and
   held-out generators (Phase 2).
3. Produce baseline scores from published checkpoints, no fine-tuning (Phase 3).
4. Only then run `easv evaluate` on real scores and write the success-criteria
   table *before* looking at the test split.

## Open decisions for the project owner

| decision | options | impact |
|---|---|---|
| ASV baseline | pretrained ECAPA-TDNN vs SSL-derived embeddings | comparability with SASV 2022 baselines |
| CM baseline | AASIST vs reproduced ASVspoof CM | attack coverage and t-DCF reference |
| Emotion model | pretrained SER on the score file vs ground-truth emotion | the latter leaks labels and is demo-only |
| Emotion taxonomy | 5–7 categorical vs dimensional (arousal/valence) | fusion features and gate design |
| Primary generalization track | cross-generator vs cross-emotion | which claim the paper can make |
