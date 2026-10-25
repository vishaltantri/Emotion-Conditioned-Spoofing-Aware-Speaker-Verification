# Figures

The five manuscript figures are generated from stored evaluation output by
[scripts/make_figures.py](../scripts/make_figures.py). No number in them is typed by
hand: every value is read from `results/real/*/results.json` (metrics, intervals,
gate weights, ASV summary) or recomputed from `test_trial_scores.csv` against the
manifest — so a re-run of the evaluation updates the plots.

| file | shows | data source |
|---|---|---|
| `fig1_arms_by_regime` | SASV-EER with grouped-bootstrap 95 % intervals for the five arms, one panel per countermeasure regime; the sign flip between regimes is the point | `results.json` per run |
| `fig2_gate_weights` | the negative result in one image: gate-weight spread across emotions is named in the legend (speaker weight 0.00 %, 13.49 %, 4.04 %, 0.81 % for regimes A–D; countermeasure weight 0.00 %, 0.25 %, 1.68 %, 0.40 %). Deviations are plotted against each regime's own mean so the four regimes share one scale; the largest excursion is the failing-CM regime, and its weights still do not follow emotion order | `results.json` → `models[].gate_weights` |
| `fig3_vulnerability` | ASV spoof acceptance by generator (pooled 38.8 % hides 5.9–59.0 %) and the emotion × generator matrix, which separates StyleTTS2 instead of averaging it away | `asv_summary.json`; matrix recomputed from the raw ASV scores at the dev threshold |
| `fig4_det_curves` | DET curves for `linear` vs `emotion_gated` per regime on probit axes, open circles at the equal-error point; the crossing reproduces the reported SASV-EER to the second decimal | `test_trial_scores.csv` + manifest |
| `fig5_asv_scores` | per-generator ECAPA score distributions against the dev-selected threshold, showing why a single pooled spoof-acceptance rate misleads | raw ASV scores |

Each figure is written twice, `.png` (200 dpi) for reading and `.pdf` (vector) for
submission.

## Regenerating

```bash
pip install -e '.[figures]'        # or: .venv/bin/python -m pip install matplotlib
.venv/bin/python scripts/make_figures.py
```

Optional paths, when the corpora live elsewhere:

```bash
.venv/bin/python scripts/make_figures.py \
    --manifest results/e2e/merged_manifest.csv \
    --results-root results/real \
    --out figures \
    --asv-scores $DATA_ROOT/real/scores_asv_merged.csv \
    --asv-summary results/real/asv_summary.json
```

`fig3`'s heatmap and `fig5` need the raw ASV score file, which lives with the
corpora rather than in the repository; without it they fall back to the stored
marginals or are skipped with a notice, so a clean checkout still runs.

## What is deliberately not drawn

* The system diagram in the course report is a schematic, not a data figure; it is
  not regenerated here.
* No figure is produced from synthetic demo runs — their numbers validate the
  pipeline and are not research results.
* Held-out-emotion folds are reported as tables in
  [docs/REAL_RESULTS.md](../docs/REAL_RESULTS.md) rather than as a sixth figure: the
  finding is a null result (≤0.03 points with a working countermeasure, ≤0.15 with
  a failing one), and a bar chart of indistinguishable pairs would overstate it.
