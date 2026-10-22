"""Summarise real ASV scores: bona fide separation and spoof acceptance.

The merged real manifest joins three kinds of trials onto one score file:

* ``target``    — bona fide speech presented as its own speaker (ESD),
* ``nontarget`` — bona fide speech presented as another speaker (derived),
* ``spoof``     — EmoSpoof-TTS clones presented as the cloned speaker.

Without countermeasure scores a full SASV-EER is not definable, but two
quantities are, and they are the ones that motivate a countermeasure at all:

1. **SV-EER** over target vs nontarget bona fide trials — how well the speaker
   front-end separates genuine speakers on emotional speech.
2. **Spoof acceptance rate** at the SV threshold chosen on dev at equal error —
   the share of cloned emotional utterances the ASV accepts as the cloned
   speaker. Broken down per emotion and per generator, this is the measured
   vulnerability the fusion stage has to fix.

Score direction is the project convention: `speaker_score`, higher = more like
the claimed speaker. Thresholds come from dev only; test is reported, never used
to choose anything.

Usage:
    .venv/bin/python scripts/summarize_real_scores.py \
        --manifest results/e2e/merged_manifest.csv \
        --scores   /home/vct/data/easv/real/scores_asv_merged.csv \
        --out      results/real/asv_summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from easv.metrics import compute_eer  # noqa: E402


def load_manifest(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        return {r["trial_id"]: r for r in csv.DictReader(fh)}


def load_scores(path: Path) -> dict[str, float]:
    with open(path, encoding="utf-8") as fh:
        return {r["trial_id"]: float(r["speaker_score"]) for r in csv.DictReader(fh)}


def eer_threshold(rows: list[tuple[float, bool]]) -> tuple[float, float]:
    scores = [s for s, _ in rows]
    labels = [lab for _, lab in rows]
    res = compute_eer(scores, labels)
    return res.eer, res.threshold


def main() -> None:
    ap = argparse.ArgumentParser(__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    scores = load_scores(args.scores)

    joined: list[dict] = []
    missing = 0
    for tid, row in manifest.items():
        s = scores.get(tid)
        if s is None:
            missing += 1
            continue
        joined.append({**row, "speaker_score": s})
    if not joined:
        raise SystemExit("no trials joined between manifest and scores")

    summary: dict = {"joined": len(joined), "missing_scores": missing,
                     "per_split": {}, "spoof_acceptance": {}}

    # --- 1) SV-EER on bona fide trials, dev threshold -----------------------
    dev_rows = [(r["speaker_score"], r["trial_type"] == "target")
                for r in joined if r["split"] == "dev"
                and r["trial_type"] in ("target", "nontarget")]
    dev_eer, dev_thr = eer_threshold(dev_rows)
    summary["dev_sv_eer"] = dev_eer
    summary["dev_sv_threshold"] = dev_thr

    def accept_rate(rows: list[dict], thr: float) -> float | None:
        if not rows:
            return None
        return sum(1 for r in rows if r["speaker_score"] >= thr) / len(rows)

    for split in ("train", "dev", "test"):
        split_rows = [r for r in joined if r["split"] == split]
        bona = [(r["speaker_score"], r["trial_type"] == "target")
                for r in split_rows if r["trial_type"] in ("target", "nontarget")]
        entry: dict = {"n": len(split_rows)}
        if bona and any(lab for _, lab in bona) and not all(lab for _, lab in bona):
            entry["sv_eer"], entry["sv_eer_threshold"] = eer_threshold(bona)
        spoof_rows = [r for r in split_rows if r["trial_type"] == "spoof"]
        entry["spoof_n"] = len(spoof_rows)
        entry["spoof_accept_at_dev_thr"] = accept_rate(spoof_rows, dev_thr)
        targets = [r for r in split_rows if r["trial_type"] == "target"]
        entry["target_accept_at_dev_thr"] = accept_rate(targets, dev_thr)
        summary["per_split"][split] = entry

    # --- 2) spoof acceptance by emotion / generator (held-out test) ----------
    test_spoof = [r for r in joined
                  if r["split"] == "test" and r["trial_type"] == "spoof"]
    by_emo: dict[str, list[dict]] = defaultdict(list)
    by_gen: dict[str, list[dict]] = defaultdict(list)
    for r in test_spoof:
        by_emo[r["emotion"]].append(r)
        by_gen[r["attack"]].append(r)

    summary["spoof_acceptance"]["overall_test"] = accept_rate(test_spoof, dev_thr)
    summary["spoof_acceptance"]["by_emotion_test"] = {
        k: {"n": len(v), "accept_rate": accept_rate(v, dev_thr),
            "mean_score": sum(r["speaker_score"] for r in v) / len(v)}
        for k, v in sorted(by_emo.items())
    }
    summary["spoof_acceptance"]["by_generator_test"] = {
        k: {"n": len(v), "accept_rate": accept_rate(v, dev_thr),
            "mean_score": sum(r["speaker_score"] for r in v) / len(v)}
        for k, v in sorted(by_gen.items())
    }

    # --- 3) bona fide reference on the same split ---------------------------
    test_targets = [r for r in joined
                    if r["split"] == "test" and r["trial_type"] == "target"]
    summary["bona_fide_test"] = {
        "n": len(test_targets),
        "accept_rate": accept_rate(test_targets, dev_thr),
        "mean_score": (sum(r["speaker_score"] for r in test_targets)
                       / len(test_targets)) if test_targets else None,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.with_suffix(".json")).write_text(
        json.dumps({"provenance": "real", **summary}, indent=2), encoding="utf-8")

    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    lines = [
        "# Real ASV summary (provenance: real)",
        "",
        f"Trials joined: {summary['joined']} (unscored: {missing}).",
        f"Dev SV-EER on bona fide trials: {100 * dev_eer:.2f}% "
        f"at threshold {dev_thr:+.4f} (chosen on dev only).",
        "",
        "| split | trials | SV-EER | spoof n | spoof accepted @dev thr | target accepted |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, e in summary["per_split"].items():
        eer = f"{100 * e['sv_eer']:.2f}%" if "sv_eer" in e else "n/a"
        lines.append(f"| {split} | {e['n']} | {eer} | {e['spoof_n']} | "
                     f"{pct(e['spoof_accept_at_dev_thr'])} | "
                     f"{pct(e['target_accept_at_dev_thr'])} |")
    lines += ["", "## Test-split spoof acceptance by emotion", "",
              "| emotion | n | accepted | mean score |", "|---|---:|---:|---:|"]
    for emo, e in summary["spoof_acceptance"]["by_emotion_test"].items():
        lines.append(f"| {emo} | {e['n']} | {pct(e['accept_rate'])} | "
                     f"{e['mean_score']:+.4f} |")
    lines += ["", "## Test-split spoof acceptance by generator", "",
              "| generator | n | accepted | mean score |", "|---|---:|---:|---:|"]
    for gen, e in summary["spoof_acceptance"]["by_generator_test"].items():
        lines.append(f"| {gen} | {e['n']} | {pct(e['accept_rate'])} | "
                     f"{e['mean_score']:+.4f} |")
    bf = summary["bona_fide_test"]
    lines += ["", f"Bona fide test targets accepted: {pct(bf['accept_rate'])} "
                  f"of {bf['n']} (mean score {bf['mean_score']:+.4f}).",
              "", "Caveat: no countermeasure scores exist yet, so this is the ASV "
              "vulnerability that motivates emotion-conditioned fusion — not a "
              "SASV-EER result.",
              ""]
    args.out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
