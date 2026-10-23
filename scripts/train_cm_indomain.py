"""In-domain CM (countermeasure) trained on the real manifest's train split.

The LA-trained AASIST checkpoint transfers poorly to EmoSpoof-TTS clones
(CM-EER 50.8 %, see docs/REAL_RESULTS.md). This script trains a small spectrogram
CNN countermeasure **on the corpus at hand** so the fusion stage has a usable CM:

* bona fide  = ESD English targets/nontargets (train-split speakers only),
* spoof      = EmoSpoof-TTS clones (same split, so speakers stay disjoint),
* model      = 4-block log-mel CNN, global mean+max pooling, one logit,
* selection  = early stopping on dev CM-EER, never on test.

Consequences, stated in the output and the report: without ``--holdout-attack``
the CM is **in-domain** (it knows every generator in the manifest), so its EER is
optimistic and says nothing about unseen attack systems. With
``--holdout-attack <generator>`` that generator is excluded from training and
from dev-based early stopping, which turns the run into the cross-generator
generalisation test used for the headline results (0.09 % in-domain, 6.27 % on
held-out F5-TTS, 2.50 % on held-out StyleTTS2).

Outputs `trial_id,cm_score` for every trial (higher = more bona fide), a JSON
metrics file and the model state dict.

Usage:
    .venv/bin/python scripts/train_cm_indomain.py \
        --manifest results/e2e/merged_manifest.csv \
        --out-dir  /home/vct/data/easv/real/cm_indomain \
        --device cuda --epochs 12
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def load_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_wav(path: str, sr_out: int, seconds: float) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    n = int(sr_out * seconds)
    if sr != sr_out:
        idx = (np.arange(int(len(wav) * sr_out / sr)) * sr / sr_out).astype(np.int64)
        wav = wav[idx]
    if len(wav) > n:
        start = (len(wav) - n) // 2
        wav = wav[start:start + n]
    elif len(wav) < n:
        wav = np.tile(wav, int(math.ceil(n / max(1, len(wav)))))[:n]
    return wav.astype(np.float32)


class MelFrontend:
    """Log-mel spectrogram, computed in torch so the dataset stays dependency-free."""

    def __init__(self, sr: int = 16000, n_fft: int = 512, hop: int = 160,
                 n_mels: int = 64, fmin: float = 20.0, fmax: float = 7600.0):
        import torch

        self.torch = torch
        self.n_fft, self.hop, self.sr = n_fft, hop, sr
        self.window = torch.hann_window(n_fft)
        self.mel = self._mel_filterbank(n_mels, n_fft, sr, fmin, fmax)

    @staticmethod
    def _mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float, fmax: float):
        import torch

        def hz_to_mel(f):
            return 2595.0 * math.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10 ** (m / 2595.0) - 1.0)

        mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
        hz = mel_to_hz(mels)
        freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
        fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
        for i in range(n_mels):
            lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
            left = (freqs - lo) / max(1e-9, mid - lo)
            right = (hi - freqs) / max(1e-9, hi - mid)
            fb[i] = np.clip(np.minimum(left, right), 0, None)
        return torch.from_numpy(fb)

    def __call__(self, wav: np.ndarray):
        torch = self.torch
        x = torch.from_numpy(wav)
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                          window=self.window, center=True, return_complex=True)
        power = spec.real ** 2 + spec.imag ** 2
        mel = self.mel @ power
        logmel = torch.log(mel + 1e-6)
        # per-utterance standardisation keeps level/recording differences out
        return (logmel - logmel.mean()) / (logmel.std() + 1e-5)


class TrialDS:
    """Manifest rows -> (log-mel, bona-fide label, index). Picklable by design."""

    def __init__(self, rows: list[dict], seconds: float):
        self.rows = rows
        self.seconds = seconds
        self._frontend = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import torch

        if self._frontend is None:
            self._frontend = MelFrontend(sr=16000)
        r = self.rows[i]
        wav = read_wav(r["path"], 16000, self.seconds)
        feat = self._frontend(wav).unsqueeze(0)  # [1, n_mels, T]
        label = 0.0 if r["trial_type"] == "spoof" else 1.0
        return feat, torch.tensor(label), i


def build_model(n_mels: int = 64):
    import torch.nn as nn

    class CMNet(nn.Module):
        def __init__(self):
            super().__init__()

            def block(cin, cout):
                return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                    nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                    nn.MaxPool2d(2),
                )

            self.net = nn.Sequential(block(1, 32), block(32, 64), block(64, 96), block(96, 128))
            self.head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

        def forward(self, x):  # x: [B, 1, n_mels, T]
            h = self.net(x)
            h = h.mean(dim=(2, 3)), h.amax(dim=(2, 3))
            import torch
            h = torch.cat(h, dim=1)
            return self.head(h).squeeze(1)

    return CMNet()


def main() -> None:
    ap = argparse.ArgumentParser(__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--holdout-attack", default=None,
                    help="generator excluded from training entirely; its rows are only "
                         "scored, which turns the run into a cross-generator test")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from easv.metrics import compute_eer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    frontend = MelFrontend(sr=16000)

    rows = [r for r in load_manifest(args.manifest) if Path(r["path"]).exists()]
    # CM labels: spoof rows are attacks, everything else is bona fide.
    by_split: dict[str, list[dict]] = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)
    for split, rs in by_split.items():
        n_spoof = sum(1 for r in rs if r["trial_type"] == "spoof")
        print(f"{split}: {len(rs)} trials ({n_spoof} spoof)", flush=True)

    holdout = args.holdout_attack
    if holdout:
        held = sum(1 for r in rows if r["attack"] == holdout)
        if held == 0:
            raise SystemExit(f"holdout attack '{holdout}' not present in the manifest")
        print(f"cross-generator run: '{holdout}' ({held} spoof trials) excluded from "
              f"training and reported separately", flush=True)

        # Python 3.14 defaults to the forkserver start method, which refuses to
    # pickle locally defined classes; fork is fine for a read-only dataset.
    torch.multiprocessing.set_start_method("fork", force=True)

    def make_loader(rs, shuffle):
        return DataLoader(TrialDS(rs, args.seconds), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=4, persistent_workers=True,
                          drop_last=shuffle)

    def without_holdout(rs):
        return [r for r in rs if r["attack"] != holdout] if holdout else rs

    train_loader = make_loader(without_holdout(by_split["train"]), True)
    dev_loader = make_loader(without_holdout(by_split["dev"]), False)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_eer, best_state, best_epoch = 1.0, None, -1

    def score_loader(loader):
        model.eval()
        out = {}
        with torch.no_grad():
            for x, _, idx in loader:
                logits = model(x.to(device)).detach().cpu().numpy()
                for j, i in enumerate(idx.tolist()):
                    out[i] = float(logits[j])
        return out

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss) * len(y)
            seen += len(y)
        dev_scores = score_loader(dev_loader)
        dev_rows = without_holdout(by_split["dev"])
        s = [dev_scores[i] for i in range(len(dev_rows))]
        lab = [r["trial_type"] != "spoof" for r in dev_rows]
        dev_eer = compute_eer(s, lab).eer
        print(f"epoch {epoch}: loss={total / max(1, seen):.4f} dev_CM_EER={100 * dev_eer:.2f}% "
              f"({time.time() - t0:.0f}s)", flush=True)
        if dev_eer < best_eer:
            best_eer, best_epoch = dev_eer, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise SystemExit("training did not improve on dev")
    model.load_state_dict(best_state)
    model.to(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.out_dir / "cm_indomain.pt")

    # --- score every split with the dev-selected model -----------------------
    scores: dict[str, float] = {}
    metrics = {"provenance": "real", "model": "logmel_cnn", "best_epoch": best_epoch,
               "dev_cm_eer": best_eer, "holdout_attack": holdout,
               "note": ("in-domain CM: trained on ESD bona fide vs EmoSpoof clones" if not holdout
                        else f"cross-generator CM: '{holdout}' excluded from training and dev "
                             f"selection, scored for generalisation")}
    for split, rs in by_split.items():
        loader = make_loader(rs, False)
        raw = score_loader(loader)
        s = np.array([raw[i] for i in range(len(rs))])
        lab = np.array([r["trial_type"] != "spoof" for r in rs])
        eer = compute_eer(s, lab).eer
        metrics[f"{split}_cm_eer"] = float(eer)
        metrics[f"{split}_n"] = len(rs)
        print(f"{split}: CM-EER = {100 * eer:.2f}%", flush=True)
        if holdout:
            # held-out rows are all spoof, so the comparison set must add the
            # bona fide trials of the same split; EER inside the spoof-only
            # subset is undefined (single class).
            idx = [i for i, r in enumerate(rs)
                   if r["attack"] == holdout or r["trial_type"] != "spoof"]
            if idx and any(lab[i] for i in idx) and not all(lab[i] for i in idx):
                sub_eer = compute_eer(s[idx], lab[idx]).eer
                metrics[f"{split}_heldout_{holdout}_cm_eer"] = float(sub_eer)
                print(f"{split}: held-out generator {holdout} CM-EER = "
                      f"{100 * sub_eer:.2f}% (n={len(idx)})", flush=True)
        for i, r in enumerate(rs):
            scores[r["trial_id"]] = float(s[i])

    out_csv = args.out_dir / "scores_cm_indomain.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["trial_id", "cm_score"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"trial_id": r["trial_id"],
                             "cm_score": f"{scores[r['trial_id']]:.6f}"})
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {out_csv} and {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
