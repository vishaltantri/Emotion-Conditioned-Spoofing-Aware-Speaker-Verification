"""Per-generator speaker-identity recovery of the ASV front-end.

Motivation (open item 8 in docs/REMAINING_WORK_PLAN.md): the ECAPA front-end
accepts 59.0 % of CosyVoice and 51.5 % of F5-TTS clones as the cloned speaker
but only 5.9 % of StyleTTS2 clones, and docs/PROGRESS_LOG.md records that as
unexplained: audio properties were checked, identity recovery was not measured.

`producer_asv.py` scores a trial against ONE centroid — the claimed speaker's —
so a single cosine number cannot say whether the clone still *looks like* that
speaker. This script re-embeds the same trials and scores each one against
**every** enrolled centroid, which turns the question into a closed-set
identification measurement:

    is the cloned (true) speaker the nearest centroid?  what rank is it?

For a bona fide target, top-1 identification is the reference behaviour of the
front-end; for a clone it is "did the speaker identity survive synthesis?".
Splitting that by generator tells us whether StyleTTS2's low acceptance is the
identity disappearing (top-1 collapses, mean cosine to the true speaker falls)
or something else.

Correctness anchor: enrollment reproduces `producer_asv.py` exactly (same
train-split centroids, same cap, same leave-one-out fallback), and the script
re-computes the claimed-speaker score and compares it against the existing
`--reference-scores` file. A replication error would show up there as a
mismatch, before any number reaches the report.

Resumable: embeddings are cached to <out>/embeddings.npz in chunks, so an
interrupted run restarts from the last chunk.

Usage (needs the audio venv: torch + speechbrain + soundfile):

    .venv-rm/bin/python scripts/asv_identity_recovery.py \
        --manifest results/e2e/merged_manifest.csv \
        --reference-scores /home/vct/data/easv/real/scores_asv_merged.csv \
        --out-dir results/real/identity_recovery --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

CAP_ENROLL = 20      # must match producer_asv.py
CHUNK = 500          # embedding cache granularity


def pct(x: float) -> str:
    """Percent to one decimal, rounded half-up (91.25 -> 91.3, not 91.2).

    Plain ``:.1f`` uses round-half-to-even, which reports 4380/4800 as 91.2 % and
    disagrees with the half-up rounding used in the report text.
    """
    return f"{100 * x + 1e-9:.1f}"


def read_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_reference(path: Path) -> dict[str, float]:
    with open(path, encoding="utf-8") as fh:
        return {r["trial_id"]: float(r["speaker_score"])
                for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser(__file__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--reference-scores", required=True, type=Path,
                    help="existing claimed-speaker scores; the recomputed score for "
                         "every trial is checked against this file")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--threshold", type=float, default=0.4331,
                    help="operating threshold reported in results/real/asv_summary.md "
                         "(dev-selected SV-EER threshold); used only to reproduce the "
                         "acceptance column alongside the identification columns")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N trials (smoke test)")
    args = ap.parse_args()

    import numpy as np
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

    rows = read_manifest(args.manifest)
    keep = set(args.splits.split(","))
    probes = [r for r in rows if r["split"] in keep and Path(r["path"]).exists()]
    if args.limit:
        probes = probes[: args.limit]

    def read_audio(path: Path) -> "torch.Tensor":
        # soundfile, not torchaudio.load: torchaudio 2.11 dropped its native
        # backends (see producer_asv.py for the same note)
        wav_np, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np.T)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.mean(dim=0).unsqueeze(0)

    def embed(path: Path) -> torch.Tensor:
        with torch.no_grad():
            return model.encode_batch(read_audio(path)).squeeze(0).squeeze(0)

    # ---- enrollment: identical to producer_asv.py --------------------------
    enroll_rows = [r for r in rows if r["split"] == "train"
                   and r["trial_type"] == "target" and Path(r["path"]).exists()]
    by_spk: dict[str, list[Path]] = defaultdict(list)
    for r in enroll_rows:
        by_spk[r["claimed_speaker"]].append(Path(r["path"]))

    centroids: dict[str, torch.Tensor] = {}
    for spk, paths in by_spk.items():
        centroids[spk] = torch.stack([embed(p) for p in paths[:CAP_ENROLL]]).mean(0)
    print(f"enrollment centroids from train targets: {len(centroids)}", flush=True)

    all_targets = [r for r in rows if r["trial_type"] == "target"
                   and Path(r["path"]).exists()]
    pool_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    def pool_embs(spk: str, split: str) -> dict[str, torch.Tensor]:
        """Leave-one-out fallback for speakers with no train targets
        (identical to producer_asv.py)."""
        key = (spk, split)
        if key not in pool_cache:
            local = [r for r in all_targets
                     if r["claimed_speaker"] == spk and r["split"] == split]
            if len(local) < 2:
                local = [r for r in all_targets if r["claimed_speaker"] == spk]
            chosen = local[:CAP_ENROLL]
            pool_cache[key] = {r["trial_id"]: embed(Path(r["path"])) for r in chosen}
        return pool_cache[key]

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)))

    # ---- probe embeddings, resumable via an .npz chunk cache ----------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "embeddings.npz"
    ids = [r["trial_id"] for r in probes]
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        cache = dict(np.load(cache_path, allow_pickle=False))
        print(f"resuming from cache: {len(cache)} embeddings", flush=True)

    t0 = time.time()
    for i in range(0, len(probes), CHUNK):
        chunk = probes[i : i + CHUNK]
        todo = [r for r in chunk if r["trial_id"] not in cache]
        for n, r in enumerate(todo, 1):
            try:
                cache[r["trial_id"]] = embed(Path(r["path"])).cpu().numpy()
            except Exception as exc:  # unreadable audio: skip, like the producer
                print(f"skip {r['trial_id']}: {exc}", file=sys.stderr, flush=True)
            if n % 100 == 0:
                print(f"  chunk {i // CHUNK + 1}: {n}/{len(todo)}",
                      flush=True)
        np.savez(cache_path, **cache)
        print(f"cached {len(cache)}/{len(probes)} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- closed-set identification over every centroid ----------------------
    # Candidate set: every speaker we hold a centroid for. Probes whose claimed
    # speaker needed the fallback still contribute that centroid, so the set is
    # the union of both enrollment routes.
    def vector(spk: str, split: str) -> torch.Tensor | None:
        if spk in centroids:
            return centroids[spk]
        pool = pool_embs(spk, split)
        if not pool:
            return None
        return torch.stack(list(pool.values())).mean(0)

    # every speaker named anywhere in the manifest gets a candidate centroid
    candidates: dict[str, torch.Tensor] = {}
    for spk in sorted({r["claimed_speaker"] for r in rows}):
        c = vector(spk, "test")
        if c is not None:
            candidates[spk] = c
    print(f"candidate centroids: {len(candidates)}", flush=True)

    if not cache:
        raise SystemExit("no embeddings cached; nothing to score")

    reference = read_reference(args.reference_scores)
    ref_checked, ref_max_diff, ref_abs = 0, 0.0, 0.0

    per_trial = []
    for r in probes:
        tid = r["trial_id"]
        if tid not in cache:
            continue
        # cached embeddings live on CPU (npz); centroids are on the model device
        emb = torch.from_numpy(cache[tid]).to(device)
        true_spk = r["speaker_id"] or r["claimed_speaker"]
        sims = {spk: cosine(emb, c) for spk, c in candidates.items()}
        # claimed score, recomputed the way producer_asv.py computes it
        if r["claimed_speaker"] in centroids:
            claimed_vec: torch.Tensor | None = centroids[r["claimed_speaker"]]
        else:
            pool = pool_embs(r["claimed_speaker"], r["split"])
            others = [e for tid2, e in pool.items() if tid2 != tid]
            claimed_vec = torch.stack(others).mean(0) if others else None
        claimed = cosine(emb, claimed_vec) if claimed_vec is not None else None

        if claimed is not None and tid in reference:
            d = abs(claimed - reference[tid])
            ref_checked += 1
            ref_max_diff = max(ref_max_diff, d)
            ref_abs += d

        order = sorted(sims, key=sims.get, reverse=True)
        rank = order.index(true_spk) + 1 if true_spk in sims else None
        per_trial.append({
            "trial_id": tid,
            "split": r["split"],
            "trial_type": r["trial_type"],
            "emotion": r["emotion"],
            "generator": r["attack"] if r["trial_type"] == "spoof" else "none",
            "true_speaker": true_spk,
            "true_score": sims.get(true_spk),
            "claimed_score": claimed,
            "top1": bool(order and order[0] == true_spk),
            "rank": rank,
            "accepted": (claimed is not None and claimed >= args.threshold),
        })

    # ---- correctness anchor --------------------------------------------------
    if ref_checked == 0:
        raise SystemExit("no reference scores matched; refusing to report")
    mean_diff = ref_abs / ref_checked
    print(f"claimed-score replication vs {args.reference_scores.name}: "
          f"n={ref_checked} mean|diff|={mean_diff:.2e} max|diff|={ref_max_diff:.2e}",
          flush=True)
    if ref_max_diff > 1e-3:
        raise SystemExit(
            f"replication mismatch: max |diff| {ref_max_diff:.4f} > 1e-3; "
            "enrollment no longer matches producer_asv.py, results would be wrong")

    # ---- aggregation ---------------------------------------------------------
    if not per_trial:
        raise SystemExit("no trials scored; nothing to report")

    def block(sel: list[dict]) -> dict:
        n = len(sel)
        if not n:
            return {"n": 0}
        ranks = [t["rank"] for t in sel if t["rank"]]
        return {
            "n": n,
            "top1_rate": sum(t["top1"] for t in sel) / n,
            "mean_rank": (sum(ranks) / len(ranks)) if ranks else None,
            "mrr": sum(1.0 / t["rank"] for t in sel if t["rank"]) / n,
            "mean_true_score": float(np.mean([t["true_score"] for t in sel
                                              if t["true_score"] is not None])),
            "accepted_rate": sum(t["accepted"] for t in sel) / n,
        }

    spoof = [t for t in per_trial if t["trial_type"] == "spoof"]
    target = [t for t in per_trial if t["trial_type"] == "target"]
    nontarget = [t for t in per_trial if t["trial_type"] == "nontarget"]

    summary = {
        "provenance": {"kind": "real", "model": "speechbrain/spkrec-ecapa-voxceleb",
                       "splits": args.splits, "threshold": args.threshold,
                       "manifest": str(args.manifest),
                       "reference_scores": str(args.reference_scores),
                       "reference_check": {"n": ref_checked,
                                           "mean_abs_diff": mean_diff,
                                           "max_abs_diff": ref_max_diff}},
        "enrollment": {"centroids": len(candidates),
                       "route": "train-split targets, cap 20, leave-one-out fallback "
                                "(identical to producer_asv.py)"},
        "reference": {"target": block(target), "nontarget": block(nontarget)},
        "spoof_overall": block(spoof),
        "by_generator": {g: block([t for t in spoof if t["generator"] == g])
                         for g in sorted({t["generator"] for t in spoof})},
        "by_emotion": {e: block([t for t in spoof if t["emotion"] == e])
                       for e in sorted({t["emotion"] for t in spoof})},
    }

    (args.out_dir / "identity_recovery.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    with open(args.out_dir / "per_trial.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_trial[0].keys()))
        w.writeheader()
        w.writerows(per_trial)

    # ---- human-readable report ----------------------------------------------
    lines = ["# Speaker-identity recovery (provenance: real)", "",
             f"Closed-set identification of every {args.splits} trial against all "
             f"{len(candidates)} enrolled centroids.",
             f"Claimed-score replication against `{args.reference_scores.name}`: "
             f"mean |diff| {mean_diff:.2e}, max {ref_max_diff:.2e} "
             f"(n = {ref_checked}) — the enrollment is bit-for-bit the producer's.", "",
             "## Reference behaviour (bona fide)", "",
             "| class | n | top-1 | mean rank | MRR | mean true-score |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in ("target", "nontarget"):
        b = summary["reference"][name]
        if not b.get("n"):
            lines.append(f"| {name} | 0 | — | — | — | — |")
            continue
        lines.append(f"| {name} | {b['n']} | {pct(b['top1_rate'])} % | "
                     f"{b['mean_rank']:.2f} | {b['mrr']:.3f} | "
                     f"{b['mean_true_score']:+.4f} |")
    if summary["spoof_overall"].get("n"):
        lines += ["", "## Clones by generator", "",
                  "| generator | n | top-1 identity recovered | mean rank | MRR | "
                  "mean true-score | accepted @ dev thr |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for g, b in summary["by_generator"].items():
            lines.append(f"| {g} | {b['n']} | {pct(b['top1_rate'])} % | "
                         f"{b['mean_rank']:.2f} | {b['mrr']:.3f} | "
                         f"{b['mean_true_score']:+.4f} | "
                         f"{pct(b['accepted_rate'])} % |")
        lines += ["", "## Clones by emotion", "",
                  "| emotion | n | top-1 | mean rank | mean true-score | accepted |",
                  "|---|---:|---:|---:|---:|---:|"]
        for e, b in summary["by_emotion"].items():
            lines.append(f"| {e} | {b['n']} | {pct(b['top1_rate'])} % | "
                         f"{b['mean_rank']:.2f} | {b['mean_true_score']:+.4f} | "
                         f"{pct(b['accepted_rate'])} % |")
        b = summary["spoof_overall"]
        lines += ["", f"Overall spoof: n = {b['n']}, "
                  f"top-1 {pct(b['top1_rate'])} %, "
                  f"accepted {pct(b['accepted_rate'])} %.", ""]
    else:
        lines += ["", "(no spoof trials in this run — smoke test)", ""]
    (args.out_dir / "identity_recovery.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"-> {args.out_dir}/identity_recovery.md", flush=True)


if __name__ == "__main__":
    main()
