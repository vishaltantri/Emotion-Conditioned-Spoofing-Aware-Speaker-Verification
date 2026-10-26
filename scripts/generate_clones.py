"""Generate the cloned-speech side of the CREMA-D corpus.

The journal corpus pairs real CREMA-D recordings with clones of the same actors,
so that a spoof trial claims the speaker it actually imitates (see
docs/CORPUS_STRATEGY.md). This script is the generation half: it decides *what*
to synthesise, runs an external text-to-speech command for each item, and records
what landed so an interrupted run resumes instead of restarting.

Design decisions that matter:

* **Deterministic planning.** Per (actor, emotion) cell the first clips in sorted
  order become the cloning references (they must be long enough to clone from) and
  the rest become synthesis targets. No sampling, no seeds to remember. A cell with
  no clip at the preferred reference length falls back to its longest clip and says
  so, because CREMA-D is a short-clip corpus (median 2.5 s) and dropping such cells
  would remove particular emotions for particular actors.
* **No duplicated text.** Each target clip is synthesised once per system. Adding
  more files by repeating the same transcript with different references inflates
  the corpus without adding information; the speaker count is what carries the
  evidence.
* **Resumable.** Every completed file is appended to a manifest CSV immediately, so
  a crash loses at most one generation.
* **Backend-agnostic.** The TTS systems disagree about flags, so the runner takes a
  command template rather than importing three heavyweight packages. Example
  templates (adapt to the installed versions):

      # StyleTTS2 (MIT code; README requires disclosing synthetic samples)
      --cmd "python -m styletts2.inference --ref {ref} --text {text} --out {out}"

      # F5-TTS (code MIT, weights CC BY-NC 4.0 -> non-commercial research only)
      --cmd "f5-tts_infer-cli --ref_audio {ref} --gen_text {text} --output_file {out}"

      # CosyVoice (Apache-2.0)
      --cmd "python -m cosyvoice.cli.cosyvoice --ref {ref} --text {text} --out {out}"

Nothing here downloads audio or a model. `--dry-run` writes the plan and prints
the estimated generation count so the cost is visible before any GPU time is
spent.

Usage:
    # 1) build the bona fide manifest first
    .venv/bin/python scripts/build_manifests.py crema \
        --crema-root $DATA/crema/AudioWAV --out results/e2e/m_crema.csv

    # 2) see the plan (no synthesis)
    .venv/bin/python scripts/generate_clones.py plan \
        --manifest results/e2e/m_crema.csv \
        --transcripts data_manifests/crema_sentences.csv \
        --system styletts2 --out $DATA/crema/clones --dry-run

    # 3) generate (one system per run; re-run the same command to resume)
    .venv/bin/python scripts/generate_clones.py plan \
        --manifest results/e2e/m_crema.csv \
        --transcripts data_manifests/crema_sentences.csv \
        --system styletts2 --out $DATA/crema/clones \
        --cmd "python -m styletts2.inference --ref {ref} --text {text} --out {out}"
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_manifests import _wav_seconds as wav_seconds  # noqa: E402
from build_manifests import parse_crema_name  # noqa: E402

PLAN_HEADER = [
    "system", "actor", "emotion", "sentence", "text",
    "ref_path", "source_path", "out_path", "trial_id",
]


def load_transcripts(path: Path) -> dict[str, str]:
    """``sentence,text`` CSV -> {sentence_code: text}.

    CREMA-D's twelve prompt sentences are not guessed here: the mapping is taken
    from the corpus release, because a wrong transcript would be cloned into the
    target and would silently change how natural the clone sounds.
    """
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "sentence" not in rows[0] or "text" not in rows[0]:
        raise SystemExit(f"{path}: expected columns 'sentence,text'")
    mapping = {}
    for row in rows:
        code = (row["sentence"] or "").strip()
        text = (row["text"] or "").strip()
        if not code or not text:
            raise SystemExit(f"{path}: empty sentence or text in row {row!r}")
        mapping[code] = text
    return mapping


def read_crema_targets(manifest_csv: Path) -> list[dict]:
    """Bona fide CREMA-D rows from a manifest built by ``build_manifests.py crema``."""
    with open(manifest_csv, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("dataset") == "crema-d" and r.get("trial_type") == "target"]
    if not rows:
        raise SystemExit(f"{manifest_csv}: no CREMA-D target rows; run "
                         "'build_manifests.py crema' first")
    clips = []
    for row in rows:
        parsed = parse_crema_name(Path(row["path"]).stem)
        if parsed is None:
            raise SystemExit(f"cannot parse CREMA-D filename from {row['path']!r}")
        actor, sentence, emotion, level = parsed
        clips.append(dict(actor=actor, sentence=sentence, emotion=emotion,
                          level=level, path=Path(row["path"]),
                          trial_id=row["trial_id"]))
    return clips


def plan_generation(clips: list[dict], transcripts: dict[str, str], *,
                    system: str, out_root: Path,
                    refs_per_cell: int = 2,
                    targets_per_cell: int = 10,
                    min_ref_seconds: float = 3.0,
                    min_ref_floor: float = 2.0,
                    strict_refs: bool = False) -> tuple[list[dict], dict]:
    """Decide references and synthesis targets cell by cell.

    One row per generation, sorted deterministically. Raises when a sentence has no
    transcript — that is a corpus problem that would silently change every clone
    built from it, and it should stop the run before any GPU time is spent.

    **Short-reference fallback.** ``min_ref_seconds`` is a preference, not a rule.
    CREMA-D clips run 1.27–5.00 s with a median of 2.50 s, so on the real corpus 181
    of 546 (actor, emotion) cells contain no clip that reaches 3 s at all. Aborting
    there — the original behaviour — made the planner unusable on the corpus it was
    written for. Dropping those cells instead would remove an emotion for particular
    actors, which biases the corpus along exactly the axis the study analyses. So a
    cell without a long-enough clip falls back to its **longest** clip, recorded in
    the returned report so the reference quality is visible rather than assumed.

    ``min_ref_floor`` is the hard limit: a cell whose longest clip is still shorter
    than this is reported as unusable and skipped (or made fatal by
    ``strict_refs``). ``strict_refs`` also turns the fallback itself into an error,
    for anyone who wants the old all-or-nothing behaviour.

    Returns ``(rows, report)`` where report counts cells, lists the fallback cells
    with the reference length each used, the shortest reference the plan relies on,
    and any cells that produced no target or no usable reference.
    """
    cells: dict[tuple[str, str], list[dict]] = {}
    for clip in clips:
        cells.setdefault((clip["actor"], clip["emotion"]), []).append(clip)

    missing_sentences: set[str] = set()
    rows: list[dict] = []
    report: dict = {"cells": len(cells), "fallback_cells": [], "skipped_cells": [],
                    "cells_without_targets": [], "min_ref_used": None}

    def ref_key(pair: tuple[dict, float]):
        """Longest first, unreadable header last, then name — deterministic."""
        clip, seconds = pair
        ok = seconds == seconds
        return (not ok, -(seconds if ok else 0.0), clip["path"].name)

    for (actor, emotion), cell in sorted(cells.items()):
        cell = sorted(cell, key=lambda c: c["path"].name)
        usable = [(c, wav_seconds(c["path"])) for c in cell]
        long_enough = [c for c, seconds in usable
                       if seconds == seconds and seconds >= min_ref_seconds]
        if long_enough:
            refs = long_enough[:refs_per_cell]
        else:
            if strict_refs:
                # The pre-fallback behaviour, kept for callers who would rather
                # stop than synthesise from a short reference.
                raise SystemExit(
                    f"cell {actor}/{emotion}: no clip reaches {min_ref_seconds:g} s, "
                    "so there is nothing to clone from")
            best, best_seconds = sorted(usable, key=ref_key)[0]
            if not (best_seconds == best_seconds and best_seconds >= min_ref_floor):
                report["skipped_cells"].append(
                    f"{actor}/{emotion} (longest {best_seconds:.2f} s)")
                continue
            refs = [best]
            report["fallback_cells"].append((f"{actor}/{emotion}", round(best_seconds, 2)))
        for ref in refs:
            used = wav_seconds(ref["path"])
            if used == used and (report["min_ref_used"] is None
                                 or used < report["min_ref_used"]):
                report["min_ref_used"] = round(used, 2)
        ref = refs[0]
        targets = [c for c in cell if c["trial_id"] not in {r["trial_id"] for r in refs}]
        if not targets:
            report["cells_without_targets"].append(f"{actor}/{emotion}")
        for clip in targets[:targets_per_cell]:
            text = transcripts.get(clip["sentence"])
            if text is None:
                missing_sentences.add(clip["sentence"])
                continue
            rows.append(dict(
                system=system, actor=actor, emotion=emotion,
                sentence=clip["sentence"], text=text,
                ref_path=str(ref["path"]), source_path=str(clip["path"]),
                out_path=str(Path(out_root) / system / actor / Path(clip["path"]).name),
                trial_id=f"crema_clone_{Path(clip['path']).stem}_{system}",
            ))
    if missing_sentences:
        raise SystemExit(
            "no transcript for sentence code(s) "
            f"{', '.join(sorted(missing_sentences))}; fill them in the transcripts CSV")
    return rows, report


def completed(path: Path) -> set[str]:
    """Output paths already recorded as generated (the resume set)."""
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as fh:
        return {r["out_path"] for r in csv.DictReader(fh)}


def run(rows: list[dict], command: str, done_csv: Path,
        backend=None, verbose: bool = True) -> tuple[int, int]:
    """Execute the plan. Returns ``(generated, skipped)``.

    ``backend`` exists so tests can drive the loop without a TTS install; when it
    is None the shell template in ``command`` is used, one process per item.
    """
    done_csv.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    new_file = not done_csv.exists()
    with open(done_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PLAN_HEADER)
        if new_file:
            writer.writeheader()
        for row in rows:
            out = Path(row["out_path"])
            # The filesystem is the truth for resumption, not the log: a file
            # deleted after a crash must be regenerated, and a zero-byte file left
            # by an interrupted write must not count as done.
            if out.exists() and out.stat().st_size > 0:
                skipped += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            if backend is not None:
                backend(row)
            else:
                argv = [part.format(text=row["text"], ref=row["ref_path"], out=row["out_path"])
                        for part in shlex.split(command)]
                result = subprocess.run(argv, capture_output=True, text=True)
                if result.returncode != 0:
                    raise SystemExit(
                        f"generation failed for {row['trial_id']} (exit "
                        f"{result.returncode}):\n{result.stderr[-2000:]}")
            writer.writerow(row)
            fh.flush()
            written += 1
            if verbose and written % 100 == 0:
                print(f"  {written} generated, {skipped} skipped", flush=True)
    return written, skipped


def main() -> None:
    ap = argparse.ArgumentParser(__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="plan (and optionally run) one system's generations")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--transcripts", required=True, type=Path,
                   help="CSV with columns sentence,text (from the CREMA-D release)")
    p.add_argument("--system", required=True,
                   help="generator label; becomes the attack id and the output subdirectory")
    p.add_argument("--out", required=True, type=Path,
                   help="root for generated audio; files land in <out>/<system>/<actor>/")
    p.add_argument("--reflist", type=Path, default=None,
                   help="write the full plan to this CSV (default: <out>/<system>/plan.csv)")
    p.add_argument("--refs-per-cell", type=int, default=2)
    p.add_argument("--targets-per-cell", type=int, default=10)
    p.add_argument("--min-ref-seconds", type=float, default=3.0,
                   help="preferred reference length; a cell without one falls back "
                        "to its longest clip (see --min-ref-floor)")
    p.add_argument("--min-ref-floor", type=float, default=2.0,
                   help="hard floor: a cell whose longest clip is shorter is skipped")
    p.add_argument("--strict-refs", action="store_true",
                   help="fail instead of falling back or skipping when a cell has no "
                        "clip at --min-ref-seconds")
    p.add_argument("--cmd", default=None,
                   help="TTS command template with {ref}, {text} and {out} placeholders")
    p.add_argument("--dry-run", action="store_true",
                   help="write the plan, print the cost estimate, generate nothing")
    args = ap.parse_args()

    # Name the missing input and the command that produces it: the two paths a
    # first run is most likely to get wrong, and a traceback does not say either.
    inputs = (
        (args.manifest, "manifest",
         "build it first: build_manifests.py crema --crema-root <AudioWAV> --out <csv>"),
        (args.transcripts, "transcripts CSV",
         "expected columns 'sentence,text'; the twelve CREMA-D prompts are in "
         "data_manifests/crema_sentences.csv"),
    )
    for path, what, hint in inputs:
        if not path.exists():
            raise SystemExit(f"{what} not found: {path}\n  -> {hint}")

    clips = read_crema_targets(args.manifest)
    transcripts = load_transcripts(args.transcripts)
    rows, report = plan_generation(clips, transcripts, system=args.system,
                                   out_root=args.out,
                                   refs_per_cell=args.refs_per_cell,
                                   targets_per_cell=args.targets_per_cell,
                                   min_ref_seconds=args.min_ref_seconds,
                                   min_ref_floor=args.min_ref_floor,
                                   strict_refs=args.strict_refs)

    plan_csv = args.reflist or (args.out / args.system / "plan.csv")
    plan_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PLAN_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    cells = {(r["actor"], r["emotion"]) for r in rows}
    print(f"plan: {len(rows)} generations over {len(cells)} of {report['cells']} "
          "(actor, emotion) cells")
    if report["fallback_cells"]:
        shortest = min(seconds for _cell, seconds in report["fallback_cells"])
        print(f"  {len(report['fallback_cells'])} cell(s) had no clip >= "
              f"{args.min_ref_seconds:g} s and used their longest clip instead "
              f"(shortest such reference {shortest:.2f} s)")
    if report["min_ref_used"] is not None:
        print(f"  shortest reference the plan relies on: {report['min_ref_used']:.2f} s")
    if report["skipped_cells"]:
        print(f"  WARNING: skipped {len(report['skipped_cells'])} unusable cell(s) "
              f"(longest clip below {args.min_ref_floor:g} s): "
              f"{', '.join(report['skipped_cells'])}")
    if report["cells_without_targets"]:
        print(f"  WARNING: {len(report['cells_without_targets'])} cell(s) produced no "
              f"target once the reference was excluded: "
              f"{', '.join(report['cells_without_targets'])}")
    print(f"wrote {plan_csv}")
    print(f"GPU estimate at ~2.5 s per item (vendor docs, not measured here): "
          f"{len(rows) * 2.5 / 3600:.1f} h")

    if args.dry_run:
        return
    if not args.cmd:
        raise SystemExit("--cmd is required unless --dry-run is given")
    done_csv = args.out / args.system / "generated.csv"
    written, skipped = run(rows, args.cmd, done_csv)
    print(f"generated {written}, skipped {skipped} (already present) -> {done_csv}")


if __name__ == "__main__":
    main()
