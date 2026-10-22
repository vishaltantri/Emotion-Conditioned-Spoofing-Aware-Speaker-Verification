"""CM (countermeasure) score producer: pretrained AASIST over real audio.

Runs the official ASVspoof-2019-LA AASIST checkpoint (``AASIST.pth``, bundled
with the AASIST release) over the audio files referenced by a manifest and writes
``trial_id,cm_score`` rows for :mod:`easv.scores`. Direction follows the project
convention and AASIST's own evaluation script: the score is the bonafide logit
``batch_out[:, 1]``, so **higher = more bona fide**.

The pretrained model is the standard SASV-2022-style CM front-end, which is why
it can score our EmoSpoof clones without retraining on LA here; training an
in-domain AASIST is a separate stage that needs the LA corpus extracted.

Audio handling mirrors ``data_utils.Dataset_ASVspoof2019_eval``: 16 kHz mono,
``nb_samp = 64600`` samples (4.04 s), centre crop when longer, tile when shorter.

Usage:
    .venv/bin/python scripts/producer_cm_aasist.py \
        --manifest results/e2e/merged_manifest.csv \
        --out      /home/vct/data/easv/real/scores_cm_merged.csv \
        --aasist-root /home/vct/data/easv/aasist \
        --device cuda [--limit 100]

Resumable: trial ids already in the output file are skipped, and the file is
flushed after every batch.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

BATCH = 16
NB_SAMP = 64600
SR = 16000


def load_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as fh:
        return {r["trial_id"] for r in csv.DictReader(fh)}


def prepare(x: np.ndarray) -> np.ndarray:
    """Mono, 64600 samples: centre crop when long, tile when short."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.shape[0] > NB_SAMP:
        start = (x.shape[0] - NB_SAMP) // 2
        x = x[start:start + NB_SAMP]
    elif x.shape[0] < NB_SAMP:
        reps = int(np.ceil(NB_SAMP / max(1, x.shape[0])))
        x = np.tile(x, reps)[:NB_SAMP]
    return x.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--aasist-root", required=True, type=Path)
    ap.add_argument("--config", default="config/AASIST.conf")
    ap.add_argument("--weights", default="models/weights/AASIST.pth")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--limit", type=int, default=0, help="score at most N trials (smoke test)")
    args = ap.parse_args()

    root = args.aasist_root.resolve()
    sys.path.insert(0, str(root))

    import json

    import soundfile as sf
    import torch
    from importlib import import_module

    with open(root / args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    model_cfg = cfg["model_config"]
    if model_cfg["nb_samp"] != NB_SAMP:
        raise SystemExit(f"unexpected nb_samp in config: {model_cfg['nb_samp']}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    module = import_module(f"models.{model_cfg['architecture']}")
    model = module.Model(model_cfg).to(device)
    state = torch.load(root / args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AASIST loaded: {n_params} params on {device}", flush=True)

    rows = load_manifest(args.manifest)
    todo = [r for r in rows if r.get("path") and Path(r["path"]).exists()]
    done = load_done(args.out)
    todo = [r for r in todo if r["trial_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"scoring {len(todo)} trials ({len(rows) - len(todo) - len(done)} without audio, "
          f"{len(done)} already scored)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists()
    out = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=["trial_id", "cm_score"])
    if write_header:
        writer.writeheader()

    scored = 0
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start:start + args.batch_size]
        xs, keep = [], []
        for r in batch:
            try:
                wav, sr = sf.read(r["path"], dtype="float32", always_2d=True)
            except Exception as exc:  # unreadable file -> skip, keep going
                print(f"skip {r['trial_id']}: {exc}", file=sys.stderr, flush=True)
                continue
            if sr != SR:
                idx = (np.arange(int(wav.shape[0] * SR / sr)) * sr / SR).astype(np.int64)
                wav = wav[idx]
            xs.append(prepare(wav))
            keep.append(r)
        if not xs:
            continue
        x = torch.from_numpy(np.stack(xs)).to(device)
        with torch.no_grad():
            _, logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            scores = logits[:, 1].detach().cpu().numpy().ravel()
        for r, s in zip(keep, scores):
            writer.writerow({"trial_id": r["trial_id"], "cm_score": f"{float(s):.6f}"})
        scored += len(keep)
        out.flush()
        if (start // args.batch_size) % 20 == 0:
            print(f"{scored}/{len(todo)} scored", flush=True)
    out.close()
    print(f"scored={scored} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
