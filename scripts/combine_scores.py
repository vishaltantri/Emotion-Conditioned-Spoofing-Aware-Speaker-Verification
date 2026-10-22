"""Combine per-producer score files into the wide score table `easv` consumes.

Each producer writes one narrow file (`trial_id,speaker_score` / `cm_score` /
`emo_*` columns). The evaluation stage expects a single file with at least
`trial_id,speaker_score,cm_score` plus optional `emo_<label>` columns, so this
joins them on `trial_id` with a left join driven by the ASV file (the speaker
score is the one column no arm can do without).

Rows missing a required column are dropped and counted, never filled with a
placeholder: a fabricated CM score would silently change every SASV number.

Usage:
    .venv/bin/python scripts/combine_scores.py \
        --speaker real/scores_asv_merged.csv \
        --cm      real/scores_cm_merged.csv \
        --emotion real/scores_ser_merged.csv \
        --out     real/scores_combined.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser(__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", required=True, type=Path)
    ap.add_argument("--cm", required=True, type=Path)
    ap.add_argument("--emotion", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    base = read_rows(args.speaker)
    cm = {r["trial_id"]: r["cm_score"] for r in read_rows(args.cm)}
    emo_cols: list[str] = []
    emo: dict[str, dict[str, str]] = {}
    if args.emotion:
        emo_rows = read_rows(args.emotion)
        if emo_rows:
            emo_cols = [c for c in emo_rows[0] if c.startswith("emo_")]
            emo = {r["trial_id"]: r for r in emo_rows}

    header = ["trial_id", "speaker_score", "cm_score", *emo_cols]
    kept, no_cm, no_emo = 0, 0, 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for r in base:
            tid = r["trial_id"]
            if tid not in cm:
                no_cm += 1
                continue
            row = {"trial_id": tid, "speaker_score": r["speaker_score"],
                   "cm_score": cm[tid]}
            if emo_cols:
                src = emo.get(tid)
                if src is None:
                    no_emo += 1
                else:
                    for c in emo_cols:
                        row[c] = src.get(c, "")
            writer.writerow(row)
            kept += 1

    print(f"wrote {kept} rows -> {args.out}")
    print(f"dropped: missing cm={no_cm}, missing emotion={no_emo}")
    if emo_cols:
        print(f"emotion columns: {', '.join(emo_cols)}")
    if no_emo:
        print("warning: some trials have no emotion columns; emotion-conditioned "
              "arms will treat them as unknown rather than guess")


if __name__ == "__main__":
    main()
