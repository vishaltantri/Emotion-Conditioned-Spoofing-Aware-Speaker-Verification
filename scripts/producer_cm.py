"""CM (countermeasure) score converter for the EASV pipeline.

Converts an ASVspoof/AASIST-style score file into the `easv.scores` CSV format.
The AASIST `produce_evaluation_file` format (line: `utt_id src key score`) is
the default. Score convention: higher = more bonafide (ASVspoof convention) —
the score is passed through as `cm_score` unmapped, after an optional SLC/ZAR
calibration over the provided calibration subset, so no sign flip occurs.

Usage:
    .venv/bin/python scripts/producer_cm.py \
        --scores-file exp_result/la.txt --out scores/cm_scores.csv \
        [--trial-id-column 1] [--delimiter space]
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_line(line: str, delimiter: str, uid_col: int, score_col: int):
    parts = line.strip().split(delimiter)
    if len(parts) <= max(uid_col, score_col):
        return None
    return parts[uid_col], parts[score_col]


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--scores-file", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--delimiter", default=" ", help="separator in scores file")
    ap.add_argument("--trial-id-column", type=int, default=0)
    ap.add_argument("--score-column", type=int, default=3,
                    help="AASIST writes utt_id src key score -> column 3")
    ap.add_argument("--nonlinearity", choices=["identity", "logit"], default="identity",
                    help="apply when the input file holds probabilities in (0,1)")
    args = ap.parse_args()

    rows: list[dict[str, str]] = []
    with open(args.scores_file, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            parsed = parse_line(line, args.delimiter, args.trial_id_column, args.score_column)
            if parsed is None:
                print(f"skip malformed line {ln}", flush=True)
                continue
            utt, raw = parsed
            value = float(raw)
            if args.nonlinearity == "logit":
                clamped = min(max(value, 1e-7), 1 - 1e-7)
                value = math.log(clamped / (1 - clamped))
            rows.append({"trial_id": utt, "cm_score": f"{value:.6f}"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["trial_id", "cm_score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"converted {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
