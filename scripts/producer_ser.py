"""SER (speech emotion recognition) score producer for the EASV pipeline.

Runs Hugging Face pipeline/model `superb/wav2vec2-base-superb-er` over the
audio files referenced by a manifest and writes `scores.csv`:

    trial_id,emo_angry,emo_happy,emo_neutral,emo_sad

superb-ER natively predicts exactly {angry, happy, neutral, sad}, which happens
to match the project's chosen 4-class EmoSpoof taxonomy, so classes map 1:1 —
no collapsing rule is needed for this checkpoint (kept explicit below in case
another backbone is swapped in).

Ground-truth manifest emotion is NEVER read or written here: the score file
contains model outputs only (leakage rule from data_manifests/README.md).

Resumable and batched; requires .venv-rm (torch + transformers + soundfile).
Usage:

    .venv-rm/bin/python scripts/producer_ser.py \
        --manifest <manifest.csv> --out scores/ser_scores.csv --device cuda
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BATCH_FRAMES = 8   # sequential frames; parallel batches on CPU are memory-heavy

CLASS_MAP = {      # superb-ER id2label -> project taxonomy, explicit on purpose
    "ang": "angry",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
}
OUT_CLASSES = ("angry", "happy", "neutral", "sad")


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--model", default="superb/wav2vec2-base-superb-er")
    parser.add_argument("--score-splits", default="train,dev,test")
    parser.add_argument("--max-seconds", type=float, default=8.0,
                        help="truncate utterances at this length to bound memory")
    args = parser.parse_args()

    import soundfile
    import torch
    import torchaudio
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

    device = torch.device(args.device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    # resolve model label ordering -> project taxonomy column order
    id2label = model.config.id2label
    cols = [CLASS_MAP.get(id2label[i].strip().lower(), None) for i in range(len(id2label))]
    if any(c is None for c in cols) or set(cols) != set(OUT_CLASSES):
        sys.exit(f"unexpected model taxonomy {id2label}; extend CLASS_MAP")

    with open(args.manifest, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["split"] in args.score_splits.split(",") and Path(r["path"]).exists()]

    def read_audio(path: Path) -> "torch.Tensor":
        # soundfile instead of torchaudio.load (torchcodec not available offline)
        wav_np, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np.T)  # [channels, T]
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.mean(dim=0)
        limit = int(args.max_seconds * 16000)
        return wav[: min(wav.numel(), limit)][None, :]

    fieldnames = ["trial_id"] + [f"emo_{c}" for c in OUT_CLASSES]
    done: set[str] = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as fh:
            done = {r["trial_id"] for r in csv.DictReader(fh)}
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)

    out = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    if not done:
        writer.writeheader()

    todo = [r for r in rows if r["trial_id"] not in done]
    if not todo:
        out.close()
        print(f"nothing to do (all {len(done)} already scored) -> {args.out}")
        return
    for i, r in enumerate(todo):
        try:
            with torch.no_grad():
                inputs = feature_extractor(
                    read_audio(Path(r["path"])).squeeze(0),
                    sampling_rate=16000,
                    return_tensors="pt",
                ).to(device)
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0)
            row_out = {"trial_id": r["trial_id"]}
            for tid, c in enumerate(cols):
                row_out[f"emo_{c}"] = f"{float(probs[tid]):.6f}"
            writer.writerow(row_out)
        except Exception as exc:
            print(f"skip {r['trial_id']}: {exc}", file=sys.stderr, flush=True)
        if (i + 1) % 50 == 0:
            out.flush()
    out.close()
    print(f"scored {i + 1} -> {args.out}")


if __name__ == "__main__":
    main()
