# Evaluation protocol

This document fixes the experimental protocol before any result is produced. It
is the contract the code in `src/easv/` implements; where a definition has a
source, the source is cited. Anything not cited here is a convention chosen by
this project and can be changed only by changing this document first.

## 1. Trials and labels

Every trial is one verification attempt with a claimed identity.

| `trial_type` | meaning | SASV ground truth |
|---|---|---|
| `target` | bona fide speech from the claimed speaker | accept (positive) |
| `nontarget` | bona fide speech from a different speaker | reject (negative) |
| `spoof` | spoofed speech presented as the claimed speaker (TTS/VC/replay) | reject (negative) |

`speaker_id` is the speaker who actually produced the audio (source speaker for
spoofed speech where known); `claimed_speaker` is the enrolled identity. The
manifest validator enforces `speaker_id == claimed_speaker` for `target` trials
and inequality for `nontarget` trials.

Spoof trials model the SASV threat: the attack imitates the enrolled speaker, so
the ASV branch alone cannot reject it. Replay (physical access) and
TTS/VC (logical access) attacks are distinguished by the `attack` column; claims
are only made for attack families present in the evaluation data.

## 2. Score conventions

* `speaker_score`: higher = more evidence for the claimed speaker.
* `cm_score`: higher = more evidence for bona fide. The spoof evidence used in
  fusion is `-cm_score`.
* Decision rule everywhere: accept when `score >= threshold`.
* Emotion: `emo_<label>` columns hold probabilities of the emotion model's output
  taxonomy. Ground-truth emotion lives in the manifest and is used only for split
  definition, breakdowns and error analysis, never as a model input.

## 3. Split discipline

1. Official benchmark partitions are used where the benchmark defines them.
2. Fusion parameters and all scalers are fitted on `train`.
3. Thresholds and model selection use `dev`.
4. `test` is evaluated once for reporting. `evaluate()` never reads test data
   while fitting or thresholding.
5. Speaker-disjoint evaluation is required unless a manifest explicitly declares
   an exception via `--allow-speaker-overlap`.
6. A trial lives in the split the corpus gave the **target rows** for its claimed
   speaker. Spoof and nontarget rows inherit it, and a generated clone inherits the
   split of the clip it was cloned from, never a fresh hash — a clone of a test
   speaker scored in train would fit the countermeasure on its own evaluation data.
   `merge_manifests.py` enforces the first rule; `build_manifests.py crema-clones`
   enforces the second.
7. Cross-generator claims require the test generators to be absent from training;
   `easv audit` reports generator overlap for exactly this check.
8. Cross-emotion claims require held-out emotion conditions: the emotion must be
   absent from both the countermeasure training and the fusion fit (`folds.holdout_emotion`
   plus `--holdout-emotion`), and the fold must leave the test split intact. The
   audit additionally reports emotions that appear in test but not in train.
9. Leakage audits (`easv audit`) are part of the protocol, not optional reporting.

## 4. Metrics

Let `FAR` be the false-alarm rate on negatives and `FRR` the false-rejection rate
on positives at a threshold; the EER is the value at the threshold where
`FAR = FRR`, obtained here by linear interpolation around the crossing point of
the two step functions.

### 4.1 SASV-EER (primary)

One binary EER over the pooled trial set: `target` trials are positive and **both**
`nontarget` and `spoof` trials are negative. The SASV 2022 evaluation plan states
that SASV-EER does not distinguish impostor from spoof attempts
([SASV 2022 evaluation plan](https://arxiv.org/html/2201.10283v2);
[challenge paper](https://www.isca-archive.org/interspeech_2022/jung22c_interspeech.html)).

Caveat to report: because the metric pools both negative types, its value depends
on the empirical mix of non-target and spoof trials in the evaluation set; this
prior dependence is discussed in the a-DCF paper
([arXiv:2403.01355](https://arxiv.org/html/2403.01355v1)).

### 4.2 Component EERs

* `SV-EER`: ASV score, target vs non-target, on bona fide trials only.
* `SPF-EER`: countermeasure score, bona fide vs spoof.

These are diagnostics; they never replace SASV-EER as the primary metric.

### 4.3 min a-DCF

Architecture-agnostic detection cost function (the name means *architecture
agnostic*, not "aggregated") with one threshold applied to one score:

```
a-DCF(t)   = C_miss·π_tar·P_miss(t) + C_fa,non·π_non·P_fa,non(t) + C_fa,spoof·π_spoof·P_fa,spoof(t)
a-DCF_def  = min( C_miss·π_tar , C_fa,non·π_non + C_fa,spoof·π_spoof )
a-DCF_norm = a-DCF(t) / a-DCF_def
min a-DCF  = min over t of a-DCF_norm(t)
```

There is no universal default: priors and costs are application assumptions. The
paper's illustrative settings are `π = (0.94, 0.01, 0.05)` and
`π = (0.98, 0.01, 0.01)` with `C_miss = 1`, `C_fa,non = C_fa,spoof = 10`
([arXiv:2403.01355](https://arxiv.org/html/2403.01355v1)). This project defaults
to the SASV/ASVspoof-style priors in `configs/*.yaml` and records them in
`results.json` so results are never compared across different cost assumptions.

### 4.4 min t-DCF

Tandem cost for an ASV + countermeasure pair, with priors and costs from the
ASVspoof 2019/2021 plans: `π_tar = 0.9405`, `π_non = 0.0095`, `π_spoof = 0.05`,
`C_miss = 1`, `C_fa = 10`, `C_fa,spoof = 10`
([ASVspoof 2021 evaluation plan, Table 1](https://arxiv.org/html/2109.00535v1)).

The ASV threshold is set at the EER point between target and non-target trials,
as in the ASVspoof protocol. With `P_miss^asv`, `P_fa^asv`, `P_fa,spoof^asv` the
ASV rates and `P_miss^cm(t)`, `P_fa^cm(t)` the countermeasure rates:

```
version "2021":   C0 = π_tar·C_miss·P_miss^asv + π_non·C_fa·P_fa^asv
                  C1 = π_tar·C_miss - C0
                  C2 = π_spoof·C_fa,spoof·P_fa,spoof^asv
                  min t-DCF = min_t (C0 + C1·P_miss^cm(t) + C2·P_fa^cm(t)) / (C0 + min(C1, C2))

version "2019":   C1 = π_tar·(C_miss - C_miss·P_miss^asv) - π_non·C_fa·P_fa^asv
                  C2 = π_spoof·C_fa,spoof·P_fa,spoof^asv
                  min t-DCF = min_t (C1·P_miss^cm(t) + C2·P_fa^cm(t)) / min(C1, C2)
```

The 2021 plan notes that the additive `C0` term was discarded in 2019 and retains
a legacy implementation
([ASVspoof 2021 metric code](https://raw.githubusercontent.com/asvspoof-challenge/2021/eval-package/eval_metrics.py)).
If the denominator degenerates (for example a perfect ASV branch, `C2 = 0`), the
code returns `NaN` and sets `note`, rather than silently reporting a number.

For a fusion model, t-DCF is computed with the ASV score as the ASV branch and the
fused score as the countermeasure score; the raw countermeasure is reported as a
reference. Both are labelled in `results.json`.

### 4.5 Threshold-dependent metrics

`FAR`, `FRR`, accuracy, precision, recall and F1 are reported at the threshold
selected on `dev`. Two policies are supported:

* `eer` (default): the dev EER point.
* `target_far`: the most permissive dev threshold whose FAR stays inside a
  security budget (`threshold.far_target`). If the budget cannot be met on dev,
  the threshold is `inf` and the report says so instead of guessing.

### 4.6 Calibration

Brier score and expected calibration error (10 equal-width bins) are reported for
the fused accept probability. Calibration is a secondary analysis: the primary
metrics are threshold-free (SASV-EER, min a-DCF, min t-DCF) and do not require
calibrated scores.

## 5. Statistical analysis

* Uncertainty is a **grouped** bootstrap that resamples whole claimed speakers
  with replacement (default 300–1000 resamples, percentile 95% interval),
  because trials from one speaker are correlated. The resampling unit is recorded
  in the report; changing it changes the interval and must be stated.
* Model comparisons use a **paired** bootstrap over the same resampled groups.
  The reported `mean_diff` is expressed as a SASV-EER *gain* (positive = the model
  is better), and `p_b_not_better` is the fraction of resamples in which the model
  was **not** better — a one-sided paired p-value. One comparison is reported
  against the configured baseline (`report.baseline_model`, default `linear`).
* Fixed seeds are recorded in the config; the demo and its tests are deterministic
  for a given seed.
* Absolute differences and intervals are reported, never only the best score, and
  gaps are described as observed rather than as statistically established when the
  interval includes zero.

## 6. Success criteria — recorded after the first real run (not pre-registered)

The research plan requires the decision framework to be written down *before*
looking at test results. **That did not happen.** Metric definitions, the split
policy and the comparison baseline were fixed before any test number existed, but
the numeric gain margin, the must-hold generalisation track and the acceptable
FRR increase were not. This table is therefore a record of the rules actually
applied, written on 2026-10-23 after test results were already known, and no
pre-registration claim may be built on it. The honest consequence: the
cross-generator StyleTTS2 track was emphasised after its numbers came in, so it
cannot be described as the pre-specified generalisation test; the two- and
three-generator results are reported side by side for that reason.

| criterion | value | status |
|---|---|---|
| primary metric | SASV-EER on the test split | fixed before the run |
| comparison baseline | `linear` (speaker + CM fusion) | fixed before the run |
| required gain over baseline | no numeric margin was fixed; results are reported as observed with paired bootstrap intervals | **not pre-registered** |
| must-hold generalization track | cross-generator (StyleTTS2 held out of CM training), added once the third generator landed | **post hoc** |
| acceptable bona-fide FRR increase | no margin was fixed on paper; target acceptance is reported at the dev-selected EER threshold (96.0 % gated vs 93.8 % linear in the three-generator run). The 1 % FAR-budget overlay has since been run on the real corpus (`results/real/abl_far_{A,D}`), and it keeps the arm ordering while widening the gaps (6.86 % FRR gated vs 10.83 % linear in regime D) — reported as a sensitivity check, not as a criterion that was fixed in advance | **not pre-registered** |
| statistical requirement | paired grouped-bootstrap interval, speakers as resampling unit, 1,000 resamples, seed 7 | fixed before the run |
| cross-emotion track | **run** (2026-10-25) for `angry` and `neutral`: the emotion is excluded from CM training *and* from the fusion fit's train/dev, then scored on its own untouched test trials. Result: ≤0.03 points change on the held-out emotion (5,040 trials each), every paired CI spanning zero — a null result that closes the objection without supporting the mechanism. Not run: folds against a failing countermeasure, and `surprise` (bona fide only, no spoof trials, so it cannot be folded for spoof detection) | done, with that caveat |

## 7. Reporting checklist

Every reported number must come with: the split it was computed on, the threshold
policy, the cost model for DCF-style metrics, the resampling unit and count for
intervals, the generator/emotion composition of the evaluation subset, and the
provenance of the scores (which encoders and which checkpoints produced them).
