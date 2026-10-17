# Results

Generated output lives here and is git-ignored except this file. Each run writes:

| file | contents |
|---|---|
| `results.json` | full machine-readable report: config snapshot, split counts, component baselines, per-model metrics, thresholds, bootstrap intervals, paired comparisons, breakdowns, gate weights, audit findings, caveats |
| `results.md` | human-readable tables rendered from the same payload |
| `test_trial_scores.csv` | per-trial test scores for every model (error analysis) |
| `models/<model>.json` | fitted fusion parameters, scalers and hyperparameters, loadable with `easv predict` |
| `audit.json`, `audit.md` | written by `easv audit` when `--out` is given |

Reproduce the demo:

```bash
easv demo --out results/demo --config configs/demo.yaml
```

Synthetic runs are labelled twice — `provenance.kind = synthetic` in the JSON and a
`Caveat` line in the Markdown — because the demo's numbers validate the pipeline,
not the research hypothesis. Real runs must record dataset versions, licences and
how emotion labels were obtained (`--set provenance.kind=real --set provenance.note=...`).
