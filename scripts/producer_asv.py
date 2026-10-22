"""ASV score producer for the EASV pipeline (Stage 3.1).

Runs SpeechBrain's pretrained `speechbrain/spkrec-ecapa-voxceleb` over the audio
files referenced by a manifest, builds one enrollment centroid per claimed
speaker from the train-split target rows, and writes `scores.csv` in the exact
schema that `easv.scores` consumes:

    trial_id,speaker_score

Score direction: cosine similarity, higher = more like the claimed speaker.

Resumable: trials already present in the output file are skipped, and the file
is flushed every batch. Requires the .venv-rm environment (torch + speechbrain
+ soundfile), NOT the lightweight .venv. Usage:

    .venv-rm/bin/python scripts/producer_asv.py \
        --manifest results/real/data_manifest_asvspoof_esd_emospoof.csv \
        --out     scores/asv_scores.csv \
        --device  cuda            # or cpu; RTX 3050 6GB fits comfortably
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BATCH = 8          # audio tensors per forward pass
CAP_ENROLL = 20    # max enrollment utterances per claimed speaker


def load_manifest_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    with open(out_path, encoding="utf-8") as fh:
        return {r["trial_id"] for r in csv.DictReader(fh)}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--enroll-split", default="train",
                        help="manifest split whose target rows form enrollment centroids")
    parser.add_argument("--score-splits", default="train,dev,test",
                        help="comma-separated manifest splits to score")
    parser.add_argument("--cap-enroll", type=int, default=CAP_ENROLL)
    args = parser.parse_args()

    import soundfile
    import torch
    import torchaudio
    from speechbrain.inference.speaker import SpeakerRecognition

    device = torch.device(args.device)
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
    )
    model.eval()

    rows = load_manifest_rows(args.manifest)
    score_rows = [r for r in rows
                  if Path(r["path"]).exists()
                  and r["split"] in args.score_splits.split(",")]

    # --- enrollment centroids from --enroll-split targets -------------------
    enroll_rows = [r for r in rows
                   if r["split"] == args.enroll_split and r["trial_type"] == "target"
                   and Path(r["path"]).exists()]
    by_spk_utt: dict[str, list[Path]] = {}
    for r in enroll_rows:
        by_spk_utt.setdefault(r["claimed_speaker"], []).append(Path(r["path"]))

    def read_audio(path: Path) -> "torch.Tensor":
        # soundfile instead of torchaudio.load: torchaudio 2.11 dropped its
        # native file backends (torchcodec needs FFMPEG; not installable offline)
        wav_np, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np.T)  # [channels, T]
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.mean(dim=0).unsqueeze(0)  # mono [1, T]

    centroids: dict[str, torch.Tensor] = {}
    for spk, paths in by_spk_utt.items():
        chosen = paths[: args.cap_enroll]
        with torch.no_grad():
            embs = []
            for p in chosen:
                emb = model.encode_batch(read_audio(p))
                embs.append(emb.squeeze(0).squeeze(0))
            centroids[spk] = torch.stack(embs).mean(dim=0)
    print(f"enrollment centroids for {len(centroids)} speakers", flush=True)

    # --- leave-one-out fallback for speakers with no train targets ----------
    # A claimed speaker whose only target rows live in dev/test cannot be
    # enrolled from train. For those, enroll from up to cap_enroll target
    # utterances of that speaker in the trial's own split (other splits
    # first if available), EXCLUDING the probe utterance itself.
    all_targets = [r for r in rows if r["trial_type"] == "target"
                   and Path(r["path"]).exists()]
    pool_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    def pool_embs(spk: str, split: str) -> dict[str, torch.Tensor]:
        key = (spk, split)
        if key in pool_cache:
            return pool_cache[key]
        local = [r for r in all_targets
                 if r["claimed_speaker"] == spk and r["split"] == split]
        if len(local) < 2:
            local = [r for r in all_targets if r["claimed_speaker"] == spk]
        chosen = local[: args.cap_enroll]
        with torch.no_grad():
            embs = [model.encode_batch(read_audio(Path(r["path"])))
                    .squeeze(0).squeeze(0) for r in chosen]
        pool_cache[key] = {r["trial_id"]: e for r, e in zip(chosen, embs)}
        return pool_cache[key]

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)))

    # --- stream trials, resumable -------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(args.out)
    write_header = args.out.exists() is False
    out = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=["trial_id", "speaker_score"])
    if write_header:
        writer.writeheader()

    todo = [r for r in score_rows if r["trial_id"] not in done]
    scored, missing_centroid = 0, 0
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        with torch.no_grad():
            for r in batch:
                key = r["claimed_speaker"]
                try:
                    emb = model.encode_batch(read_audio(Path(r["path"])))
                    emb = emb.squeeze(0).squeeze(0)
                except Exception as exc:  # corrupt/unreadable audio -> skip, log
                    print(f"skip {r['trial_id']}: {exc}", file=sys.stderr, flush=True)
                    continue
                if key in centroids:
                    c = centroids[key]
                else:
                    pool = pool_embs(key, r["split"])
                    others = [e for tid, e in pool.items() if tid != r["trial_id"]]
                    if not others:
                        missing_centroid += 1
                        continue
                    c = torch.stack(others).mean(dim=0)
                s = cosine(emb.view(-1), c.view(-1))
                writer.writerow({"trial_id": r["trial_id"],
                                 "speaker_score": f"{s:.6f}"})
            scored += len(batch)
        out.flush()
    out.close()
    print(f"scored={scored} missing_centroid={missing_centroid} -> {args.out}")


if __name__ == "__main__":
    main()
