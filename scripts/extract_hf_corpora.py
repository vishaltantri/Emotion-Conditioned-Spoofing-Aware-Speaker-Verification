"""Convert the Hugging Face parquet mirrors into the corpus layouts the pipeline reads.

Why these mirrors exist here: Edinburgh DataShare and Zenodo cap this machine at
~200 KB/s, while the HF CDN delivers the same corpora at >1 MB/s per connection,
so the mirrors are how the pending corpora actually arrive. Both are conversions
of the original distributions, not the originals; the provenance is recorded in
docs/PROGRESS_LOG.md and in the report.

``emospoof``
    The mirror ships 36,000 spoof utterances and keeps the original relative tree
    in its ``path`` column (``StyleTTS2/0011/Angry/0011_000351t1.wav``), with a
    JSON ``notes`` column carrying utterance_id/speaker/emotion/tts_model. This
    writes one generator's audio back to that tree under ``--out`` so
    ``build_manifests.py emospoof`` reads it unchanged.

``asvspoof``
    Writes each LA utterance as FLAC named by its utterance id, so the official
    CM protocol files index the directory directly.

Idempotent: existing output files are skipped, so an interrupted conversion
resumes where it stopped.

Usage:
    .venv/bin/python scripts/extract_hf_corpora.py emospoof \
        --parquet emospoof_hf/shard0.parquet emospoof_hf/shard1.parquet \
        --out emospoof --generator StyleTTS2
    .venv/bin/python scripts/extract_hf_corpora.py asvspoof \
        --parquet asvspoof2019LA/la_train.parquet --out .../flac
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path


def audio_bytes(cell) -> bytes:
    """Pull raw encoded audio out of an HF audio cell (struct or plain bytes)."""
    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            return cell["bytes"]
        if cell.get("path"):
            return Path(cell["path"]).read_bytes()
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise TypeError(f"unsupported audio cell: {type(cell)}")


def notes_field(cell) -> dict:
    if not cell:
        return {}
    if isinstance(cell, dict):
        return cell
    try:
        return json.loads(cell)
    except (TypeError, ValueError):
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["emospoof", "asvspoof"])
    ap.add_argument("--parquet", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--generator", default="StyleTTS2",
                    help="emospoof only: which generator subtree to write")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import soundfile as sf

    written = skipped = filtered = 0
    for shard in args.parquet:
        columns = ["path", "audio", "notes", "audio_file_name"]
        available = set(pq.ParquetFile(shard).schema_arrow.names)
        cols = [c for c in columns if c in available]
        table = pq.read_table(shard, columns=cols)
        names = table.column_names
        print(f"{shard.name}: {table.num_rows} rows, columns {names}", flush=True)

        paths = table.column("path").to_pylist() if "path" in names else None
        file_names = (table.column("audio_file_name").to_pylist()
                      if "audio_file_name" in names else None)
        audio = table.column("audio").to_pylist() if "audio" in names else None
        notes = table.column("notes").to_pylist() if "notes" in names else [None] * table.num_rows
        if audio is None:
            raise SystemExit(f"{shard}: no audio column (have {names})")

        for i, cell in enumerate(audio):
            meta = notes_field(notes[i])
            if args.kind == "emospoof":
                rel = paths[i] if paths else None
                if rel is None:
                    raise SystemExit(f"{shard}: emospoof needs the 'path' column")
                if rel.split("/")[0].lower() != args.generator.lower():
                    filtered += 1
                    continue
                dest = args.out / rel
            else:
                # the LA mirror names utterances in its own column, and the audio
                # struct repeats the name as a path; either identifies the trial
                utt = file_names[i] if file_names else None
                if not utt and isinstance(cell, dict) and cell.get("path"):
                    utt = Path(cell["path"]).stem
                if not utt:
                    raise SystemExit(f"{shard}: cannot determine utterance id (notes={meta})")
                utt = str(utt)
                if utt.endswith(".flac"):
                    utt = utt[:-5]
                dest = args.out / f"{utt}.flac"

            if dest.exists() and dest.stat().st_size > 0:
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = audio_bytes(cell)
            wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            sf.write(str(dest), wav[:, 0] if wav.shape[1] == 1 else wav, sr,
                     format="FLAC" if dest.suffix == ".flac" else "WAV")
            written += 1

        print(f"  written={written} skipped={skipped} filtered={filtered}", flush=True)

    print(f"done: written={written} skipped={skipped} filtered={filtered} -> {args.out}")


if __name__ == "__main__":
    main()
