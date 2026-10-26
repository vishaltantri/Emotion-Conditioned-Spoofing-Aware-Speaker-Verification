"""Gate the generated clone corpus before any fusion time is spent.

Regenerating the spoof side on a new bona fide corpus is the one step that can be
wasted: if the clones are too clean (the countermeasure spots them instantly and
the ASV accepts none) or too degraded (the speaker front-end rejects everything for
the wrong reason), the SASV ladder measures nothing and no amount of fusion tuning
repairs it. This script reads the score files that already exist and answers the
three questions in minutes:

1. **Does the ASV find the clones convincing?** Per-generator spoof-acceptance at
   the dev-selected threshold, and the spread between generators. The ESD pilot's
   finding was a 5.9 / 51.5 / 59.0 % spread; a flat or inverted profile means the
   regenerated clones do not reproduce the phenomenon.
2. **Is the emotion front-end still confidently wrong on clones?** Accuracy against
   the intended label and mean max-probability, bona fide versus clones. On the
   pilot, clones were 32.8 % accurate at 0.82 confidence. If the new clones are
   recognised correctly, the "SER as an implicit spoof cue" mechanism has no room
   to operate and the paper's analysis would change.
3. **Are the clones detectable by a countermeasure at all?** Per-generator
   SPF-EER. A countermeasure that already separates them at 0.1 % EER makes the
   fusion comparison trivial and the corpus uninteresting.

Thresholds are explicit and overridable; the script exits non-zero when a criterion
fails, so a bad corpus blocks the pipeline instead of quietly becoming a table.

Usage:
    .venv/bin/python scripts/verify_clone_quality.py \
        --manifest results/e2e/m_crema_merged.csv \
        --scores $DATA/real/scores_combined_crema.csv \
        --out results/crema/clone_gate
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from easv.metrics import compute_eer  # noqa: E402

SER_CLASSES = ("angry", "happy", "neutral", "sad")


def load_rows(manifest_csv: Path, scores_csv: Path) -> list[dict]:
    """Join manifest truth with the score columns the gate needs."""
    with open(manifest_csv, encoding="utf-8") as fh:
        manifest = {r["trial_id"]: r for r in csv.DictReader(fh)}
    rows: list[dict] = []
    with open(scores_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            trial = manifest.get(row["trial_id"])
            if trial is None:
                continue
            emo_cols = {k: float(v) for k, v in row.items()
                        if k.startswith("emo_") and v not in ("", None)}
            rows.append(dict(
                trial_id=row["trial_id"],
                split=trial["split"],
                trial_type=trial["trial_type"],
                emotion=trial["emotion"],
                generator=trial["attack"],
                speaker_score=float(row["speaker_score"]),
                cm_score=float(row["cm_score"]) if row.get("cm_score") else None,
                emotion_probs=emo_cols,
            ))
    if not rows:
        raise SystemExit(f"no overlapping trial ids between {manifest_csv} and {scores_csv}")
    return rows


def dev_threshold(rows: list[dict]) -> tuple[float, float]:
    """EER threshold on the dev split's bona fide trials (target = positive)."""
    dev = [r for r in rows if r["split"] == "dev"
           and r["trial_type"] in ("target", "nontarget")]
    if not dev:
        raise SystemExit("no dev bona fide trials; cannot set a threshold without "
                         "touching test")
    scores = np.array([r["speaker_score"] for r in dev])
    labels = np.array([r["trial_type"] == "target" for r in dev])
    if labels.all() or not labels.any():
        raise SystemExit("dev bona fide trials need both targets and nontargets")
    result = compute_eer(scores, labels)
    return result.threshold, result.eer


def ser_view(rows: list[dict], clone: bool) -> dict:
    """SER accuracy and confidence restricted to the classes the front-end knows."""
    picked = [r for r in rows
              if (r["trial_type"] == "spoof") == clone
              and r["trial_type"] in ("target", "nontarget", "spoof")
              and r["emotion"] in SER_CLASSES
              and r["emotion_probs"]]
    if not picked:
        return {"n": 0, "accuracy": None, "mean_confidence": None}
    correct = 0
    confidences = []
    for row in picked:
        probs = row["emotion_probs"]
        prediction = max(probs, key=probs.get).replace("emo_", "")
        confidences.append(max(probs.values()))
        correct += int(prediction == row["emotion"])
    return {"n": len(picked), "accuracy": correct / len(picked),
            "mean_confidence": float(np.mean(confidences))}


def summarise(rows: list[dict], threshold: float) -> dict:
    """Per-generator vulnerability, SER behaviour and countermeasure detectability."""
    by_generator: dict[str, dict] = {}
    clones = [r for r in rows if r["trial_type"] == "spoof"]
    generators = sorted({r["generator"] for r in clones})
    bona = [r for r in rows if r["trial_type"] in ("target", "nontarget")]
    for generator in generators:
        subset = [r for r in clones if r["generator"] == generator]
        entry: dict = {
            "n": len(subset),
            "accept_rate": float(np.mean([r["speaker_score"] >= threshold
                                         for r in subset])),
            "mean_speaker_score": float(np.mean([r["speaker_score"] for r in subset])),
        }
        if subset and subset[0]["cm_score"] is not None and bona:
            scores = np.array([r["cm_score"] for r in bona]
                              + [r["cm_score"] for r in subset])
            labels = np.array([True] * len(bona) + [False] * len(subset))
            entry["cm_eer"] = compute_eer(scores, labels).eer
        by_generator[generator] = entry
    acceptance = [v["accept_rate"] for v in by_generator.values()]
    return {
        "threshold": threshold,
        "generators": by_generator,
        "accept_spread_pts": (max(acceptance) - min(acceptance)) * 100 if acceptance else 0.0,
        "ser_bona_fide": ser_view(rows, clone=False),
        "ser_clones": ser_view(rows, clone=True),
        "n_bona_fide": len(bona),
        "n_clones": len(clones),
    }


def gate(summary: dict, min_spread_pts: float, max_ser_accuracy: float,
         min_ser_confidence: float, max_cm_eer: float) -> list[dict]:
    """The explicit pass/fail criteria, each with the number that decided it."""
    checks = []
    checks.append({
        "criterion": "generator spread in spoof acceptance",
        "value": summary["accept_spread_pts"],
        "rule": f">= {min_spread_pts:g} points",
        "pass": summary["accept_spread_pts"] >= min_spread_pts,
    })
    ser = summary["ser_clones"]
    if ser["n"] == 0:
        checks.append({"criterion": "SER confidently wrong on clones", "value": None,
                       "rule": "needs emo_* scores", "pass": False})
    else:
        checks.append({
            "criterion": "SER accuracy on clones",
            "value": ser["accuracy"],
            "rule": f"<= {max_ser_accuracy:g}",
            "pass": ser["accuracy"] <= max_ser_accuracy,
        })
        checks.append({
            "criterion": "SER mean confidence on clones",
            "value": ser["mean_confidence"],
            "rule": f">= {min_ser_confidence:g}",
            "pass": ser["mean_confidence"] >= min_ser_confidence,
        })
    eers = [v["cm_eer"] for v in summary["generators"].values() if "cm_eer" in v]
    if not eers:
        checks.append({"criterion": "countermeasure separates clones", "value": None,
                       "rule": "needs cm_score", "pass": False})
    else:
        worst = max(eers)
        checks.append({
            "criterion": "worst per-generator CM EER",
            "value": worst,
            "rule": f"<= {max_cm_eer:g}",
            "pass": worst <= max_cm_eer,
        })
    return checks


def render_markdown(summary: dict, checks: list[dict]) -> str:
    lines = ["# Clone-corpus quality gate", "",
             f"Trials joined: {summary['n_bona_fide']} bona fide, "
             f"{summary['n_clones']} clones. ASV threshold {summary['threshold']:+.4f} "
             "(dev-selected).", "",
             "| generator | n | spoof acceptance | mean speaker score | CM-EER |",
             "|---|---:|---:|---:|---:|"]
    for name, entry in sorted(summary["generators"].items()):
        cm = f"{100 * entry['cm_eer']:.2f} %" if "cm_eer" in entry else "n/a"
        lines.append(f"| {name} | {entry['n']} | {100 * entry['accept_rate']:.1f} % "
                     f"| {entry['mean_speaker_score']:+.4f} | {cm} |")
    lines += ["", f"Acceptance spread across generators: "
                  f"**{summary['accept_spread_pts']:.1f} points**", "",
              "| SER view | n | accuracy | mean confidence |", "|---|---:|---:|---:|"]
    for label, view in (("bona fide", summary["ser_bona_fide"]),
                        ("clones", summary["ser_clones"])):
        acc = "n/a" if view["accuracy"] is None else f"{100 * view['accuracy']:.1f} %"
        conf = "n/a" if view["mean_confidence"] is None else f"{view['mean_confidence']:.3f}"
        lines.append(f"| {label} | {view['n']} | {acc} | {conf} |")
    lines += ["", "| criterion | value | rule | result |", "|---|---:|---|---|"]
    for check in checks:
        value = "n/a" if check["value"] is None else (
            f"{check['value']:.3f}" if check["criterion"].startswith(("SER", "worst"))
            else f"{check['value']:.1f}")
        lines.append(f"| {check['criterion']} | {value} | {check['rule']} | "
                     f"{'pass' if check['pass'] else '**fail**'} |")
    overall = all(c["pass"] for c in checks)
    lines += ["", f"**Verdict: {'PASS — the corpus reproduces the pilot phenomenon' if overall else 'FAIL — do not spend fusion time on this corpus'}**"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/crema/clone_gate"))
    ap.add_argument("--threshold", type=float, default=None,
                    help="ASV acceptance threshold (default: EER threshold on dev)")
    ap.add_argument("--min-spread-pts", type=float, default=10.0)
    ap.add_argument("--max-ser-accuracy", type=float, default=0.50)
    ap.add_argument("--min-ser-confidence", type=float, default=0.60)
    ap.add_argument("--max-cm-eer", type=float, default=0.05)
    ap.add_argument("--no-fail", action="store_true",
                    help="report but exit 0 even when a criterion fails")
    args = ap.parse_args()

    # This gate is run at a checkpoint, and the most likely first run has no score
    # files yet. Say which one is missing and what produces it instead of letting a
    # FileNotFoundError traceback end the run.
    hints = {
        "manifest": (args.manifest,
                     "build it with 'build_manifests.py crema', then merge the "
                     "clone rows with 'merge_manifests.py --extra'"),
        "combined score file": (args.scores,
                                "produce ASV/SER/CM scores with producer_asv.py, "
                                "producer_ser.py and producer_cm.py, then join them "
                                "with combine_scores.py"),
    }
    for what, (path, hint) in hints.items():
        if not path.exists():
            raise SystemExit(f"{what} not found: {path}\n  -> {hint}")

    rows = load_rows(args.manifest, args.scores)
    if args.threshold is None:
        threshold, dev_eer = dev_threshold(rows)
        print(f"dev SV-EER {100 * dev_eer:.2f} %, threshold {threshold:+.4f}")
    else:
        threshold = args.threshold
    summary = summarise(rows, threshold)
    checks = gate(summary, args.min_spread_pts, args.max_ser_accuracy,
                  args.min_ser_confidence, args.max_cm_eer)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"threshold": threshold, "summary": summary, "checks": checks,
               "pass": all(c["pass"] for c in checks)}
    (args.out / "clone_gate.json").write_text(json.dumps(payload, indent=2),
                                              encoding="utf-8")
    markdown = render_markdown(summary, checks)
    (args.out / "clone_gate.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {args.out / 'clone_gate.json'} and {args.out / 'clone_gate.md'}")

    if not payload["pass"] and not args.no_fail:
        raise SystemExit("clone-corpus gate failed")


if __name__ == "__main__":
    main()
