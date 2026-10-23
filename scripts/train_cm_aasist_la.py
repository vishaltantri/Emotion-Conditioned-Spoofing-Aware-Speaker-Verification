"""Train AASIST in-domain on ASVspoof 2019 LA and score the EASV manifest.The released AASIST checkpoint does not transfer to EmoSpoof-TTS clones
(SPF-EER 50.8 %, recorded in docs/REAL_RESULTS.md), so the missing experiment is
the one the SASV protocol actually prescribes: train the AASIST architecture on
LA train audio and then measure what that trained countermeasure does on the
EASV corpus. It reaches 2.11 % EER on held-out LA speakers and 43.8 % on EASV
test, i.e. the negative transfer is a property of the ASVspoof-2019 supervision.

Design decisions, all traceable to the protocol document:

* LA rows come from the official CM protocols (``build_manifests.py asvspoof``);
  only rows whose audio exists are used, so extracting LA train alone is enough.
* Checkpoint selection uses a **speaker-disjoint LA dev split**, i.e. in-domain
  performance, because that is the only way to show the trained countermeasure
  actually discriminates LA spoofs. Selecting on EASV dev was tried first and
  collapsed to epoch 1 (EASV dev EER 39.8 %), which measures the domain gap, not
  the countermeasure. EASV dev/test are then scored once as *transfer* numbers.
* Threshold selection and fusion stay on the EASV dev split, as everywhere else
  in the project; no EASV test trial influences training or selection.
* EmoSpoof/ESD audio is 16 kHz; LA is 16 kHz too, so one 4.04 s crop
  (``nb_samp = 64600``) serves both, matching AASIST's own preprocessing.
* The architecture and its config are read from the AASIST checkout, so the model
  is the published one rather than a lookalike.

Usage:
    .venv/bin/python scripts/train_cm_aasist_la.py \
        --la-manifest results/e2e/m_asvspoof_la.csv \
        --easv-manifest results/e2e/merged_manifest.csv \
        --aasist-root /home/vct/data/easv/aasist \
        --out-dir /home/vct/data/easv/real/cm_aasist_la \
        --device cuda --epochs 8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np

NB_SAMP = 64600
SR = 16000


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_wav(path: str) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        idx = (np.arange(int(len(wav) * SR / sr)) * sr / SR).astype(np.int64)
        wav = wav[idx]
    if len(wav) > NB_SAMP:
        start = (len(wav) - NB_SAMP) // 2
        wav = wav[start:start + NB_SAMP]
    elif len(wav) < NB_SAMP:
        wav = np.tile(wav, int(np.ceil(NB_SAMP / max(1, len(wav)))))[:NB_SAMP]
    return wav.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--la-manifest", required=True, type=Path)
    ap.add_argument("--easv-manifest", required=True, type=Path)
    ap.add_argument("--aasist-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--config", default="config/AASIST.conf")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-train", type=int, default=0,
                    help="cap LA training utterances after the speaker split (0 = all)")
    ap.add_argument("--la-dev-frac", type=float, default=0.12,
                    help="fraction of LA speakers held out to select the checkpoint")
    ap.add_argument("--la-eval-n", type=int, default=4000,
                    help="LA training rows sampled for the in-domain train EER")
    ap.add_argument("--amp", default="off", choices=["off", "bf16", "fp16"],
                    help="mixed precision; bf16 needs no gradient scaler")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    root = args.aasist_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from easv.metrics import compute_eer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    la_rows = [r for r in load_rows(args.la_manifest) if Path(r["path"]).exists()]
    ea_rows = [r for r in load_rows(args.easv_manifest) if Path(r["path"]).exists()]
    print(f"LA rows with audio: {len(la_rows)}; EASV rows: {len(ea_rows)}", flush=True)
    if not la_rows:
        raise SystemExit("no LA audio found: extract ASVspoof2019_LA_train first")

    rng = np.random.default_rng(args.seed)
    la_speakers = sorted({r["speaker_id"] for r in la_rows})
    shuffled = list(la_speakers)
    rng.shuffle(shuffled)
    n_dev_spk = max(1, int(round(len(shuffled) * args.la_dev_frac)))
    la_dev_speakers = set(shuffled[:n_dev_spk])
    la_dev_rows = [r for r in la_rows if r["speaker_id"] in la_dev_speakers]
    la_train_rows = [r for r in la_rows if r["speaker_id"] not in la_dev_speakers]
    if args.max_train and len(la_train_rows) > args.max_train:
        la_train_rows = list(rng.choice(la_train_rows, args.max_train, replace=False))
        print(f"capped LA training set to {len(la_train_rows)} utterances", flush=True)
    print(f"LA split: {len(la_train_rows)} train / {len(la_dev_rows)} dev "
          f"({len(la_dev_speakers)} of {len(la_speakers)} speakers held out)", flush=True)
    la_rows = la_train_rows

    class AudioDS:
        """Picklable dataset: raw waveform + bona-fide label."""

        def __init__(self, rows: list[dict]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            import torch

            r = self.rows[i]
            wav = read_wav(r["path"])
            label = 0 if r["trial_type"] == "spoof" else 1
            return torch.from_numpy(wav), torch.tensor(label), i

    torch.multiprocessing.set_start_method("fork", force=True)

    with open(root / args.config, encoding="utf-8") as fh:
        model_cfg = json.load(fh)["model_config"]
    module = import_module(f"models.{model_cfg['architecture']}")
    model = module.Model(model_cfg).to(device)
    print(f"AASIST params: {sum(p.numel() for p in model.parameters())}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    amp_dtype = {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")

    def loader(rows, shuffle):
        return DataLoader(AudioDS(rows), batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=4, drop_last=shuffle)

    train_loader = loader(la_train_rows, True)
    la_dev_loader = loader(la_dev_rows, False)
    dev_rows = [r for r in ea_rows if r["split"] == "dev"]
    dev_loader = loader(dev_rows, False)

    def score(loader_) -> dict[int, float]:
        model.eval()
        out: dict[int, float] = {}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype,
                                             enabled=amp_dtype is not None):
            for x, _, idx in loader_:
                _, logits = model(x.to(device))
                if isinstance(logits, tuple):
                    logits = logits[0]
                # autocast may hand back bf16/fp16 logits, which numpy cannot hold
                vals = logits[:, 1].float().detach().cpu().numpy()
                for j, i in enumerate(idx.tolist()):
                    out[i] = float(vals[j])
        return out

    def eer_on(rows: list[dict], loader_) -> float:
        s = score(loader_)
        vals = [s[i] for i in range(len(rows))]
        labs = [r["trial_type"] != "spoof" for r in rows]
        return compute_eer(vals, labs).eer

    best_eer, best_state, best_epoch = 1.0, None, -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                loss = loss_fn(model(x)[1], y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach()) * len(y)
            seen += len(y)
        eer = eer_on(la_dev_rows, la_dev_loader)
        # Transfer EER is deliberately not measured per epoch: it is not used for
        # selection, and scoring the 10.8k-trial EASV dev split every epoch would
        # double the run time for a number that cannot influence anything.
        print(f"epoch {epoch}: loss={total / max(1, seen):.4f} "
              f"la_dev_CM_EER={100 * eer:.2f}% ({time.time() - t0:.0f}s)", flush=True)
        if eer < best_eer:
            best_eer, best_epoch = eer, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise SystemExit("training never improved on dev")
    model.load_state_dict(best_state)
    model.to(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.out_dir / "cm_aasist_la.pt")

    results = {"provenance": "real", "model": "AASIST",
               "trained_on": "asvspoof2019_la_train",
               "best_epoch": best_epoch, "selection": "speaker-disjoint LA dev CM-EER",
               "la_dev_cm_eer": best_eer,
               "n_la_train": len(la_train_rows), "n_la_dev": len(la_dev_rows),
               "la_dev_speakers": len(la_dev_speakers), "n_la_speakers": len(la_speakers),
               "note": (
                   "AASIST retrained on ASVspoof 2019 LA train audio; checkpoint "
                   "selected on held-out LA speakers only. EASV dev/test numbers are "
                   "transfer measurements and were not used for selection")}

    # In-domain sanity: the trained model must separate LA spoofs, otherwise the
    # transfer number below says nothing about domain shift.
    subset = la_train_rows
    if args.la_eval_n and len(subset) > args.la_eval_n:
        pick = rng.choice(len(subset), args.la_eval_n, replace=False)
        subset = [subset[i] for i in sorted(pick.tolist())]
    results["la_train_cm_eer_sampled"] = float(
        eer_on(subset, loader(subset, False)))
    results["la_train_sample_n"] = len(subset)
    print(f"LA train sample ({len(subset)}): CM-EER = "
          f"{100 * results['la_train_cm_eer_sampled']:.2f}%", flush=True)
    print(f"LA dev: CM-EER = {100 * best_eer:.2f}%", flush=True)

    all_rows: list[dict] = []
    scores: dict[str, float] = {}
    # Only the splits used downstream: dev chooses the fusion threshold, test is
    # reported once. The EASV train split needs no CM score.
    for split in ("dev", "test"):
        rs = [r for r in ea_rows if r["split"] == split]
        if not rs:
            continue
        s = score(loader(rs, False))
        vals = np.array([s[i] for i in range(len(rs))])
        labs = np.array([r["trial_type"] != "spoof" for r in rs])
        results[f"easv_{split}_cm_eer"] = float(compute_eer(vals, labs).eer)
        results[f"easv_{split}_n"] = len(rs)
        print(f"EASV {split}: CM-EER = {100 * results[f'easv_{split}_cm_eer']:.2f}%", flush=True)
        for i, r in enumerate(rs):
            scores[r["trial_id"]] = float(vals[i])
        all_rows.extend(rs)

    out_csv = args.out_dir / "scores_cm_aasist_la.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["trial_id", "cm_score"])
        writer.writeheader()
        for r in all_rows:
            writer.writerow({"trial_id": r["trial_id"], "cm_score": f"{scores[r['trial_id']]:.6f}"})
    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_csv} and {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
