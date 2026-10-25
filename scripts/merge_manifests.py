"""Merge the three corpus manifests into one SASV manifest.

Split reconciliation: Stages 2-3 builders assign splits inconsistently
(ASVspoof rows follow LA partitions; EmoSpoof rows follow generators; ESD rows
follow speakers). The merged manifest must keep each claimed speaker inside
exactly one split, otherwise `easv audit` reports split/leakage findings and
the fusion semantics get murky.

Policy:
- The **target rows are authoritative**: whichever split a corpus put a speaker's
  real recordings in, that is where every trial claiming that speaker belongs
  (ESD's 4/2/4, CREMA-D's actor-disjoint split, or any future corpus). Spoof and
  nontarget rows inherit it.
- ASVspoof LA rows keep their own LA partition and are never demoted: its
  speakers are a separate namespace.
- Generator-based split conflicts are demoted: recorded in `notes`, so a
  cross-generator analysis can still be run by regrouping on `attack`.
- Emotion: ESD/EmoSpoof/CREMA-D emotion labels win; ASVspoof rows stay `unknown`.

Usage:
    python scripts/merge_manifests.py --asvspoof a.csv --esd e.csv \
        --emospoof m.csv --out merged.csv

    # CREMA-D journal corpus: bona fide rows plus generated clones
    python scripts/merge_manifests.py --esd $D/m_crema.csv \
        --extra $D/m_crema_clones.csv --out $D/m_crema_merged.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

FIELDS_OUT = [
    "trial_id", "split", "speaker_id", "claimed_speaker", "trial_type",
    "emotion", "attack", "dataset", "path", "source_trial_id", "notes",
]


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def merge_rows(rows: list[dict]) -> list[dict]:
    """Reconcile splits, de-duplicate trial ids, normalise fields.

    Policy: the ESD speaker split (GEM-style 4/2/4) is authoritative for rows
    whose claimed speaker belongs to the ESD speaker universe — that covers the
    EmoSpoof spoof rows, whose generator-based split would otherwise place a
    test speaker's trials inside the train split. ASVspoof LA rows keep their
    own partition (its speakers are a separate namespace in the real data).
    """
    speaker_split: dict[str, str] = {}
    split_conflicts: Counter[str] = Counter()

    # Target rows define where a speaker lives; everything else follows them.
    # Deriving this from the corpus name instead would tie the merge to ESD.
    for r in rows:
        if r["trial_type"] == "target":
            speaker_split.setdefault(r["claimed_speaker"], r["split"])

    out_rows: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        r = dict(r)
        trial_id = r["trial_id"]
        if trial_id in seen:
            # emospoof spoof rows use raw stem ids that can collide across
            # corpora; prefix unifies without touching easv semantics
            r["trial_id"] = trial_id = f"{r['dataset']}_{trial_id}"
            if trial_id in seen:
                split_conflicts[trial_id] += 1
                continue
        seen.add(trial_id)
        target = speaker_split.get(r["claimed_speaker"])
        if target and target != r["split"] and r["dataset"] != "asvspoof2019_la":
            r["notes"] = (f"{r['notes']}; split demoted from {r['split']}"
                          f" to speaker split {target}").strip("; ")
            r["split"] = target
        out_rows.append(r)
    if split_conflicts:
        print(f"dropped {sum(split_conflicts.values())} duplicate trial_ids: "
              f"{dict(list(split_conflicts.items())[:5])}...")
    return out_rows


def coverage_report(rows: list[dict]) -> str:
    by_split: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_split[r["split"]][r["trial_type"]] += 1
    lines = []
    for split, counts in sorted(by_split.items()):
        lines.append(f"{split}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    by_attack_speaker = defaultdict(set)
    for r in rows:
        if r["trial_type"] == "spoof":
            by_attack_speaker[r["attack"]].add(r["claimed_speaker"])
    for attack, speakers in sorted(by_attack_speaker.items()):
        lines.append(f"attack {attack}: speakers {sorted(speakers)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--asvspoof", type=Path)
    ap.add_argument("--esd", type=Path)
    ap.add_argument("--emospoof", type=Path)
    ap.add_argument("--extra", type=Path, action="append", default=[],
                    help="additional manifest(s) to merge, e.g. generated CREMA-D clones; "
                         "repeatable")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows: list[dict] = []
    for opt in ("asvspoof", "esd", "emospoof"):
        path = getattr(args, opt)
        if path and path.exists():
            rows.extend(read_rows(path))
    for path in args.extra:
        if not path.exists():
            raise SystemExit(f"--extra manifest not found: {path}")
        rows.extend(read_rows(path))
    if not rows:
        raise SystemExit("no input manifests found")
    merged = merge_rows(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS_OUT)
        writer.writeheader()
        writer.writerows(merged)

    print(coverage_report(merged))


if __name__ == "__main__":
    main()
