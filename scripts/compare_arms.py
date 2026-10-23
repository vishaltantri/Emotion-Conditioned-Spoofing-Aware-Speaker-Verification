"""Paired grouped bootstrap between two fusion arms in an existing evaluation run.

The evaluation CLI only tests each arm against the configured baseline (``linear``).
That leaves the contrast that matters for the paper's claim untested: the proposed
gated arm against the *simpler emotion-conditioned control* (``emotion_interaction``).
If gating beats linear but not the interaction model, the gate adds parameters, not
capability.

This script reuses the persisted per-trial test scores
(``test_trial_scores.csv``) and the same resampling scheme as the CLI: groups are
claimed speakers, resamples are drawn with replacement, and the default seed and
resample count match ``configs/real_run.yaml``.

Sign convention matches the CLI: ``mean_gain`` is ``EER(b) - EER(a)``, so a
positive value means arm ``a`` is better, and ``p_b_not_worse`` is the
one-sided paired p-value for the null "a is not better than b".

Usage:
    .venv/bin/python scripts/compare_arms.py \
        --results results/real/run4_3gen_crossgen \
        --manifest results/e2e/merged_manifest.csv \
        --a emotion_gated --b emotion_interaction

Two different evaluation runs can also be compared (e.g. the SER-confidence
ablation against the full-emotion run), as long as both scored the same test
split. Pass the second directory with ``--results-b``; arm ``a`` is read from
``--results``, arm ``b`` from ``--results-b``, and the trial lists must match.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from easv.metrics import compute_eer, paired_grouped_bootstrap  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=Path,
                    help="evaluation output directory containing results.json")
    ap.add_argument("--results-b", type=Path, default=None,
                    help="optional second run directory; arm b is read from there "
                         "instead of --results (both must share the same test split)")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--a", required=True, help="arm under test")
    ap.add_argument("--b", required=True, help="control arm")
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional JSON output path")
    args = ap.parse_args()

    def read_arm(results_dir: Path, arm: str) -> tuple[list[str], list[float]]:
        with open(results_dir / "test_trial_scores.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        column = f"{arm}_score"
        if column not in rows[0]:
            raise SystemExit(f"arm {arm!r} not in {sorted(rows[0])} ({results_dir})")
        return [r["trial_id"] for r in rows], [float(r[column]) for r in rows]

    trial_ids, scores_a = read_arm(args.results, args.a)
    if args.results_b is not None:
        trial_ids_b, scores_b = read_arm(args.results_b, args.b)
        if trial_ids_b != trial_ids:
            raise SystemExit(
                "the two runs scored different test trials "
                f"({len(trial_ids)} vs {len(trial_ids_b)} rows, or different order); "
                "a paired comparison needs the identical split and ordering"
            )
    else:
        _, scores_b = read_arm(args.results, args.b)

    with open(args.manifest, encoding="utf-8") as fh:
        manifest = {r["trial_id"]: r for r in csv.DictReader(fh)}
    labels = np.array([manifest[t]["trial_type"] == "target" for t in trial_ids])
    groups = np.array([manifest[t]["claimed_speaker"] for t in trial_ids], dtype=object)

    # Keys are positional, not the arm names: the same arm name can legitimately
    # appear in both runs (e.g. an ablation run vs its parent).
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    eer_a = compute_eer(a, labels).eer
    eer_b = compute_eer(b, labels).eer

    # metric_a - metric_b is negative when a is better; flip so positive = gain.
    paired = paired_grouped_bootstrap(
        lambda idx: compute_eer(b[idx], labels[idx]).eer,
        lambda idx: compute_eer(a[idx], labels[idx]).eer,
        groups,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    result = {
        "results": str(args.results),
        "results_b": str(args.results_b) if args.results_b else None,
        "arm_a": args.a, "arm_b": args.b,
        "n_test": int(len(labels)),
        "eer_a": eer_a, "eer_b": eer_b,
        "mean_gain": paired["mean_diff"],
        "ci_lo": paired["lo"], "ci_hi": paired["hi"],
        "p_b_not_worse": paired["p_b_not_better"],
        "n_resamples": paired["n_resamples"],
        "note": "positive mean_gain = arm_a has lower SASV-EER than arm_b",
    }
    print(f"{args.a}: SASV-EER {100 * eer_a:.2f}%  vs  {args.b}: {100 * eer_b:.2f}%  "
          f"(n={len(labels)})")
    print(f"paired gain {100 * result['mean_gain']:+.3f} pts "
          f"[{100 * result['ci_lo']:+.3f}, {100 * result['ci_hi']:+.3f}] "
          f"p(not worse)={result['p_b_not_worse']:.3f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
