"""Build real-data manifests for the EASV project.

Three builders that emit CSVs matching the `src/easv/manifest.py` contract
(required columns: trial_id, split, speaker_id, claimed_speaker, trial_type,
emotion, attack; optional: path, duration_sec, dataset, source_trial_id, notes).

- build_manifest_asvspoof: ASVspoof 2019 LA CM protocol files -> spoof rows
  (attack ids A01..A19) plus bona fide target rows and derived nontarget rows.
- build_manifest_esd: ESD English recordings (speakers 0011-0020) -> target and
  nontarget trials; speakers are split GEM-style (4 train / 2 dev / 4 test).
- build_manifest_emospoof: EmoSpoof-TTS generator subsets -> spoof rows with
  emotion ground truth, attack id = generator name.

Emotion taxonomy: angry / happy / sad / neutral (+ surprise for ESD only).
This script never reads audio; paths are metadata only.
"""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

REQUIRED_HEADER = [
    "trial_id",
    "split",
    "speaker_id",
    "claimed_speaker",
    "trial_type",
    "emotion",
    "attack",
    "dataset",
    "path",
    "source_trial_id",
    "notes",
]

ESD_EMOTION_MAP = {
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "suprise": "surprise",  # note ESD's folder spelling
    "surprise": "surprise",
    "neutral": "neutral",
}

EMOSPOOF_CLASSES = ("angry", "happy", "sad", "neutral")

SPEAKER_SPLIT = {
    # GEM-style design: 4 train / 2 dev / 4 test speakers (ESD English 0011-0020)
    "0011": "train", "0012": "train", "0013": "train", "0014": "train",
    "0015": "dev", "0016": "dev",
    "0017": "test", "0018": "test", "0019": "test", "0020": "test",
}


def _write(rows: list[dict], out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _stable_split(speaker: str) -> str:
    """Split by speaker with a deterministic fallback for unknown speakers."""
    if speaker in SPEAKER_SPLIT:
        return SPEAKER_SPLIT[speaker]
    h = int(hashlib.sha1(speaker.encode()).hexdigest(), 16) % 10
    return "train" if h < 4 else ("dev" if h < 6 else "test")


# --------------------------------------------------------------------------
# 1) ASVspoof 2019 LA
# --------------------------------------------------------------------------
def parse_cm_protocol_line(line: str):
    """Return (speaker_id, utterance_id, label, attack_id) or None.

    The distributed 2019 LA CM protocol is ``speaker utt - system label``, e.g.
    ``LA_0079 LA_T_1138215 - - bonafide`` or
    ``LA_0022 LA_T_1202073 - A01 spoof``. Older docs in this project assumed
    ``speaker - utt label attack``, which silently produced ``'-'`` utterance ids;
    the parser therefore locates the label field rather than trusting a position.
    Tabs or runs of spaces are both accepted.
    """
    parts = line.replace("\t", " ").split()
    if len(parts) < 4:
        return None
    lower = [p.lower() for p in parts]
    label_idx = next((i for i, p in enumerate(lower)
                      if p.startswith("bona") or p.startswith("spoof")), None)
    if label_idx is None or label_idx < 2:
        return None
    speaker, utt = parts[0], parts[1]
    label = "bonafide" if lower[label_idx].startswith("bona") else "spoof"
    attack_parts = [p for p in parts[2:label_idx] if p not in ("-", "")]
    attack = attack_parts[0] if attack_parts else "-"
    return speaker, utt, label, attack


def build_manifest_asvspoof(la_root: Path, out_csv: Path,
                            nontarget_per_speaker: int = 10,
                            seed: int = 1234) -> int:
    """ASVspoof 2019 LA train/dev partitions -> manifest rows.

    Bona fide utterances become `target` rows; a capped number of
    cross-speaker presentations becomes `nontarget` rows. Spoof rows keep the
    protocol's claimed speaker as both speaker ids — the LA protocol does not
    disclose the source voice, and the audit rules forbid inferring it.
    """
    from collections import defaultdict
    la_root = Path(la_root)
    rows: list[dict] = []
    rng = random.Random(seed)

    for split_name in ("train", "dev"):
        proto_dir = la_root / "ASVspoof2019_LA_cm_protocols"
        # the distribution names the training protocol .trn and the trial lists
        # .trl; accept either rather than hard-coding one spelling
        proto = next((p for p in (
            proto_dir / f"ASVspoof2019.LA.cm.{split_name}.trn.txt",
            proto_dir / f"ASVspoof2019.LA.cm.{split_name}.trl.txt",
            proto_dir / f"ASVspoof2019.LA.cm.{split_name}.trl",
        ) if p.exists()), None)
        if proto is None:
            print(f"skip split '{split_name}': no CM protocol under {proto_dir}")
            continue
        flac_dir = la_root / f"ASVspoof2019_LA_{split_name}" / "flac"
        bona: dict[str, str] = defaultdict(str)   # utt -> true speaker
        bona_by_spk: dict[str, list[str]] = defaultdict(list)
        with open(proto, encoding="utf-8") as fh:
            for line in fh:
                parsed = parse_cm_protocol_line(line.rstrip("\n"))
                if parsed is None:
                    continue
                speaker, utt, label, attack = parsed
                path = flac_dir / f"{utt}.flac"
                if label == "bonafide":
                    rows.append(dict(trial_id=utt, split=split_name,
                                     speaker_id=speaker, claimed_speaker=speaker,
                                     trial_type="target", emotion="unknown",
                                     attack="none", dataset="asvspoof2019_la",
                                     path=str(path), source_trial_id="", notes=""))
                    bona[utt] = speaker
                    bona_by_spk[speaker].append(utt)
                else:
                    rows.append(dict(trial_id=utt, split=split_name,
                                     speaker_id=speaker, claimed_speaker=speaker,
                                     trial_type="spoof", emotion="unknown",
                                     attack=attack, dataset="asvspoof2019_la",
                                     path=str(path), source_trial_id="",
                                     notes="source voice undisclosed in LA protocol"))
        # derived nontargets: utterance by src != claimed, presented as claimed
        speakers = sorted(bona_by_spk)
        for s in speakers:
            pool = [u for u in bona if bona[u] != s]
            take = rng.sample(pool, min(nontarget_per_speaker, len(pool)))
            for utt in take:
                src_speaker = bona[utt]
                rows.append(dict(trial_id=f"{utt}_nt_as_{s}", split=split_name,
                                 speaker_id=src_speaker, claimed_speaker=s,
                                 trial_type="nontarget", emotion="unknown",
                                 attack="none", dataset="asvspoof2019_la",
                                 path=str(la_root / f"ASVspoof2019_LA_{split_name}"
                                          / "flac" / f"{utt}.flac"),
                                 source_trial_id=utt,
                                 notes="derived cross-speaker pair"))
    return _write(rows, out_csv)


# --------------------------------------------------------------------------
# 2) ESD English bona fide
# --------------------------------------------------------------------------
def build_manifest_esd(esd_root: Path, out_csv: Path,
                       nontarget_per_speaker_emo: int = 10,
                       seed: int = 1234) -> int:
    """ESD English speakers 0011-0020 -> target + nontarget rows.

    Splits follow SPEAKER_SPLIT so that speaker-disjointness holds across
    train/dev/test, mirroring the GEM EmoSpoof design (4/2/4 speakers).

    ``esd_root`` may itself BE the directory holding speaker folders (when the
    caller stages an ``English`` view with symlinks) or a parent containing one.
    """
    esd_root = Path(esd_root)
    rows: list[dict] = []
    rng = random.Random(seed)
    english = next((p for p in esd_root.rglob("*English*") if p.is_dir()),
                   esd_root if any(d.is_dir() for d in esd_root.iterdir()) else None)
    if english is None:
        raise FileNotFoundError(f"no 'English' directory under {esd_root}")
    by_speaker_emo: dict[str, dict[str, list[Path]]] = {}
    for spk_dir in sorted(d for d in english.iterdir() if d.is_dir()):
        for emo_dir in sorted(d for d in spk_dir.iterdir() if d.is_dir()):
            emo_norm = ESD_EMOTION_MAP.get(emo_dir.name.strip().lower())
            if emo_norm is None:
                continue
            wavs = sorted(emo_dir.glob("*.wav"))
            by_speaker_emo.setdefault(spk_dir.name, {})[emo_norm] = wavs
    for spk, emo_map in sorted(by_speaker_emo.items()):
        split_name = _stable_split(spk)
        for emo, wavs in sorted(emo_map.items()):
            for wav in wavs:
                rows.append(dict(trial_id=f"esd_{wav.stem}", split=split_name,
                                 speaker_id=spk, claimed_speaker=spk,
                                 trial_type="target", emotion=emo, attack="none",
                                 dataset="esd_english", path=str(wav),
                                 source_trial_id="",
                                 notes="" if emo in EMOSPOOF_CLASSES else "not an EmoSpoof class"))
            if emo not in EMOSPOOF_CLASSES:
                continue
            others = {o: {e: ws for e, ws in m2.items() if e == emo}
                      for o, m2 in by_speaker_emo.items() if o != spk}
            pool = [(o, oemo, ow) for o, om in sorted(others.items())
                    for oemo, ws in om.items() for ow in ws]
            take = rng.sample(pool, min(nontarget_per_speaker_emo, len(pool)))
            for o, oemo, ow in take:
                rows.append(dict(trial_id=f"esd_nt_{ow.stem}_{o}_as_{spk}",
                                 split=split_name, speaker_id=o, claimed_speaker=spk,
                                 trial_type="nontarget", emotion=emo, attack="none",
                                 dataset="esd_english", path=str(ow),
                                 source_trial_id=f"esd_{ow.stem}",
                                 notes="derived cross-speaker pair"))
    return _write(rows, out_csv)


# --------------------------------------------------------------------------
# 3) EmoSpoof-TTS spoof rows
# --------------------------------------------------------------------------
def build_manifest_emospoof(emo_root: Path, out_csv: Path,
                            generator_splits: dict | None = None) -> int:
    """EmoSpoof-TTS generator archives -> spoof rows with emotion ground truth.

    Layout discovered from extracted archives: each top-level generator folder
    (CosyVoice / F5TTS / StyleTTS2) with per-speaker/emotion structure.
    Default split mapping follows GEM: CosyVoice->train, F5TTS->dev, StyleTTS2->test.
    """
    emo_root = Path(emo_root)
    if generator_splits is None:
        generator_splits = {
            "cosyvoice": (["cosyvoice", "cosy_voice"], "train"),
            "f5_tts": (["f5tts", "f5-tts", "f5_tts"], "dev"),
            "styletts2": (["styletts2", "style_tts2", "styletts"], "test"),
        }
    rows: list[dict] = []
    for gen_key, (alt_names, split_name) in generator_splits.items():
        norm = lambda name: name.lower().replace("-", "").replace("_", "").replace(" ", "")
        alt_norm = {norm(a) for a in alt_names}
        gen_dir = next((p for p in emo_root.iterdir() if p.is_dir() and norm(p.name) in alt_norm), None)
        if gen_dir is None:
            continue
        for wav in sorted(gen_dir.rglob("*.wav")):
            emo, spk = "unknown", "unknown"
            parts_lower = [p.strip().lower() for p in wav.parts[:-1]]
            for p in parts_lower:
                if p in EMOSPOOF_CLASSES:
                    emo = p
                    break
            if wav.stem and wav.stem.split("_")[0].isdigit():
                spk = wav.stem.split("_")[0]
            rows.append(dict(trial_id=wav.stem, split=split_name,
                             speaker_id=spk, claimed_speaker=spk,
                             trial_type="spoof", emotion=emo, attack=gen_key,
                             dataset="emospoof_tts", path=str(wav),
                             source_trial_id="", notes=f"generator={gen_key}"))
    return _write(rows, out_csv)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("asvspoof")
    p.add_argument("--la-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--nontarget-per-speaker", type=int, default=10)

    p = sub.add_parser("esd")
    p.add_argument("--esd-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--nontarget-per-speaker-emo", type=int, default=10)

    p = sub.add_parser("emospoof")
    p.add_argument("--emo-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)

    args = parser.parse_args()
    if args.cmd == "asvspoof":
        n = build_manifest_asvspoof(args.la_root, args.out, args.nontarget_per_speaker)
    elif args.cmd == "esd":
        n = build_manifest_esd(args.esd_root, args.out, args.nontarget_per_speaker_emo)
    else:
        n = build_manifest_emospoof(args.emo_root, args.out)
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
