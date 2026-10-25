# Results

Generated output lives here and is git-ignored except this file: the reports record
absolute corpus paths and per-trial audio identities, so they stay out of version
control. Every number quoted in [docs/REAL_RESULTS.md](../docs/REAL_RESULTS.md) and
in the course report comes from this directory.

Each evaluation run writes:

| file | contents |
|---|---|
| `results.json` | full machine-readable report: config snapshot, split counts, component baselines, per-model metrics, thresholds, bootstrap intervals, paired comparisons, breakdowns, gate weights, audit findings, caveats |
| `results.md` | human-readable tables rendered from the same payload |
| `test_trial_scores.csv` | per-trial test scores for every model, keyed by `trial_id` (error analysis and arm-vs-arm comparisons) |
| `models/<model>.json` | fitted fusion parameters, scalers and hyperparameters, loadable with `easv predict` |
| `audit.json`, `audit.md` | written by `easv audit` when `--out` is given |

## What is in here

| path | contents |
|---|---|
| `demo/`, `abl_far/`, `abl_layer/`, `abl_no_unc/`, `verify/` | synthetic runs (`provenance.kind = synthetic`). They validate the pipeline, the ablation switches and the threshold policies; their numbers are not research results. |
| `real/run1` | two-generator corpus, in-domain countermeasure (Regime A) |
| `real/run2_oodcm` | two-generator corpus, LA-pretrained AASIST (Regime B) |
| `real/run3_crossgen` | two-generator corpus, F5-TTS held out (Regime C) |
| `real/run4_3gen_crossgen` | **headline**: three-generator corpus, StyleTTS2 held out (Regime D), plus `paired_*.json` arm-vs-arm contrasts |
| `real/run2b_serconf` | SER-confidence ablation of Regime B: emotion vector replaced by one max-probability scalar (`configs/ablation_ser_confidence.yaml`), plus `paired_*_full.json` contrasts against the parent run |
| `real/run4b_serconf` | SER-confidence ablation of Regime D — the gated arm's edge reproduces with the scalar alone (1.49 % vs 1.54 %, indistinguishable) |
| `real/run5_holdout_{angry,neutral}_off` | held-out-emotion folds, fusion fitted on all five emotions (same CM as the `_on` run) |
| `real/run5_holdout_{angry,neutral}_on` | held-out-emotion folds, the emotion also excluded from the fusion fit (`--set folds.holdout_emotion=…`); the `off`/`on` pair is the measurement, and the contrast on the held-out emotion's own test trials is produced by `compare_arms.py --emotion` |
| `real/seed{A,D}_{11,23,37}` | seed repeats of regimes A and D (`--set seed=…`); the gated arm is the only seed-sensitive arm, and it stays best in D / worse than linear in A in every seed |
| `real/asv_summary.{json,md}` | ASV component summary: SV-EER and spoof acceptance by split, emotion and generator |
| `e2e/` | real manifests (`merged_manifest.csv`, 53,900 trials), LA manifest, and audit reports |

The corpora themselves are outside the repository (see
[data_manifests/README.md](../data_manifests/README.md)); the score CSVs
(`scores_asv_merged.csv`, `scores_ser_merged.csv`, `scores_combined_*.csv` and the
countermeasure score files) live next to them under `$DATA_ROOT/real/`.

## Reproducing

```bash
# pipeline smoke test (synthetic, seconds)
easv demo --out results/demo --config configs/demo.yaml

# real runs: corpora and score producers first, then
DATA_ROOT=/home/vct/data/easv CM=indomain bash scripts/run_real_pipeline.sh
```

The headline regime needs one extra step because it holds a generator out of
countermeasure training; the exact command sequence is in
[docs/REAL_RESULTS.md](../docs/REAL_RESULTS.md) under "Reproduce".
