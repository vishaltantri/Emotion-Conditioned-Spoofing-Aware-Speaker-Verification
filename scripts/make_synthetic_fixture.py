"""Generate a tiny synthetic multi-corpus fixture for the EASV end-to-end test.

Creates speech-like formant audio (not real speech!) in three layouts:

1. ASVspoof 2019 LA layout — `LA/ASVspoof2019_LA_train/flac/*.flac` +
   `LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt` where
   spoof files are spectrally-distorted clones of bona-fide synthesis
   (enough structure for AASIST to *learn something* in a few epochs on GPU).
2. ESD layout — `ESD/English/<speaker>/<Emotion>/*.wav` (speakers 0011..0016).
3. EmoSpoof layout — `emospoof/<Generator>/<speaker>/<Emotion>/*.wav`.

Emotion acoustic profile: base F0 and level vary per class so the SER model
receives a learnable (though crude) signal; entropy of outputs then varies as
intended for uncertainty conditioning.

All outputs labelled synthetic; nothing here approximates real corpora.
Usage:
    python scripts/make_synthetic_fixture.py --root ~/data/easv/fixture
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
DUR = 1.0
EMO_F0 = {"angry": 220.0, "happy": 190.0, "sad": 110.0, "neutral": 150.0}
EMO_LEVEL = {"angry": 0.9, "happy": 0.8, "sad": 0.45, "neutral": 0.6}


def synth_wave(emo: str, spoof: bool = False, seed: int = 0, seconds: float = DUR) -> np.ndarray:
    """Formant-ish tonal speech placeholder; spoofs = tremolo + spectral tilt."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    f0 = EMO_F0[emo] * (1.0 + 0.05 * np.sin(2 * np.pi * rng.uniform(1, 3, size=len(t)) / 10))
    wave = np.zeros_like(t)
    for k, weight in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)):
        wave += weight * np.sin(2 * np.pi * k * f0 * t)
    # breathy noise floor
    wave += 0.02 * rng.standard_normal(len(t))
    if spoof:
        # spectral tilt + tremolo: what an imperfect clone leaves behind
        wave *= (1.0 + 0.05 * np.sin(2 * np.pi * 7 * t))
        wave = np.convolve(wave, np.ones(9) / 9, mode="same")  # tilt down
    level = EMO_LEVEL[emo] * (0.5 if spoof else 1.0)
    peak = np.max(np.abs(wave)) or 1.0
    return (wave / peak * level).astype(np.float32)


def write_corpus(root: Path) -> dict:
    """Create the three corpus layouts; returns a summary of file counts."""
    import hashlib

    summary: dict[str, int] = {}
    speakers = [f"00{i:02d}" for i in range(11, 21)]  # 10 speakers: 4/2/4 splits
    # LA corpus uses its own speaker namespace (LA_00xx), like the real data
    la_speakers = [f"LA_00{i:02d}" for i in range(11, 21)]
    generators = ["CosyVoice", "F5TTS", "StyleTTS2"]
    emotions = list(EMO_F0)

    # ---- 1) LA layout (CM training fixture) --------------------------------
    la = root / "LA" / "ASVspoof2019_LA_train" / "flac"
    la_dev_dir = root / "LA" / "ASVspoof2019_LA_dev" / "flac"
    proto_path = root / "LA" / "ASVspoof2019_LA_cm_protocols"
    proto_path.mkdir(parents=True, exist_ok=True)
    la_flac = la
    la_flac.mkdir(parents=True, exist_ok=True)
    la_dev_dir.mkdir(parents=True, exist_ok=True)
    proto_lines = []
    n = 0
    for spk in la_speakers:
        for emo in emotions:
            for i in range(2):
                utt = f"{spk}_{emo[:3]}_{i:03d}"
                wav = synth_wave(emo, spoof=False, seed=n)
                sf.write(la_flac / f"{utt}.flac", wav, SR)
                proto_lines.append(f"{spk} - {utt} bonafide -\n")
                # spoof clone of the same utterance under a fake generator id
                utt_s = f"{utt}gen"
                proto_lines.append(f"{spk} - {utt_s} spoof G{n % 3 + 1:02d}\n")
                n += 1
                wav_s = synth_wave(emo, spoof=True, seed=10_000 + n)
                sf.write(la_flac / f"{utt_s}.flac", wav_s, SR)
    (proto_path / "ASVspoof2019.LA.cm.train.trl.txt").write_text(
        "".join(proto_lines), encoding="utf-8")

    # small dev protocol so the manifest builder can build both splits
    dev_lines = []
    m = 0
    for spk in la_speakers[5:]:
        for emo in emotions[:2]:
            utt = f"{spk}_dev_{m:03d}"
            sf.write(la_dev_dir / f"{utt}.flac", synth_wave(emo, seed=7_000 + m), SR)
            dev_lines.append(f"{spk} - {utt} bonafide -\n")
            m += 1
    (proto_path / "ASVspoof2019.LA.cm.dev.trl.txt").write_text(
        "".join(dev_lines), encoding="utf-8")
    summary["la_files"] = n * 2 + m

    # ---- 2) ESD layout ------------------------------------------------------
    n = 0
    for spk in speakers:
        for emo in emotions:
            d = root / "ESD" / "English" / spk / emo.capitalize()
            d.mkdir(parents=True, exist_ok=True)
            sf.write(d / f"{emo[:3]}_{n:03d}.wav", synth_wave(emo, seed=n), SR)
            n += 1
    summary["esd_files"] = n

    # ---- 3) EmoSpoof layout -------------------------------------------------
    n = 0
    for gen in generators:
        for spk in speakers:
            for emo in emotions:
                d = root / "emospoof" / gen / spk / emo
                d.mkdir(parents=True, exist_ok=True)
                sf.write(d / f"{spk}_{emo}_{n:03d}.wav",
                         synth_wave(emo, spoof=True, seed=num(n, gen)), SR)
                n += 1
    summary["emospoof_files"] = n
    return summary


def num(n: int, gen: str) -> int:
    return hash((n, gen)) % 100_000


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    summary = write_corpus(args.root)
    print(summary)


if __name__ == "__main__":
    main()
