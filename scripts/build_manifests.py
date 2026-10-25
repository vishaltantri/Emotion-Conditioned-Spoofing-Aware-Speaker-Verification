"""Build real-data manifests for the EASV project.

Five builders that emit CSVs matching the `src/easv/manifest.py` contract
(required columns: trial_id, split, speaker_id, claimed_speaker, trial_type,
emotion, attack; optional: path, duration_sec, dataset, source_trial_id, notes).

- build_manifest_asvspoof: ASVspoof 2019 LA CM protocol files -> spoof rows
  (attack ids A01..A19) plus bona fide target rows and derived nontarget rows.
- build_manifest_esd: ESD English recordings (speakers 0011-0020) -> target and
  nontarget trials; speakers are split GEM-style (4 train / 2 dev / 4 test).
- build_manifest_crema: CREMA-D recordings (91 actors, ODbL) -> target and
  nontarget trials; actors are split 61/15/15 and impostors stay in their own
  partition. This is the journal corpus (docs/CORPUS_STRATEGY.md).
- build_manifest_crema_clones: the generation log from scripts/generate_clones.py
  -> spoof rows whose split is inherited from the source clip.
- build_manifest_emospoof: EmoSpoof-TTS generator subsets -> spoof rows with
  emotion ground truth, attack id = generator name.

Emotion taxonomy: angry / happy / sad / neutral (+ surprise for ESD only,
+ disgust and fear for CREMA-D, which the SER front-end does not predict).
Apart from reading WAV headers for the CREMA-D duration filter, this script never
opens audio; paths are metadata only.
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
# 2b) CREMA-D bona fide rows (journal re-anchor: ODbL 1.0 + DbCL 1.0)
# --------------------------------------------------------------------------
CREMA_EMOTION_MAP = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}


def _wav_seconds(path: Path) -> float:
    """Duration from the WAV header only; no sample data is read."""
    import wave
    try:
        with wave.open(str(path), "rb") as fh:
            rate = fh.getframerate() or 1
            return fh.getnframes() / rate
    except (wave.Error, EOFError, OSError):
        return float("nan")


def parse_crema_name(stem: str):
    """``1001_TAI_HAP_XX`` -> (actor, sentence, emotion, level), or None.

    CREMA-D encodes the target emotion and the intended intensity in the filename;
    a fourth token carries the sentence code, which is not needed for trials but is
    kept so a caller can group by prompt.
    """
    parts = stem.split("_")
    if len(parts) != 4:
        return None
    actor, sentence, emo, level = parts
    if not (len(actor) == 4 and actor.isdigit()):
        return None
    emotion = CREMA_EMOTION_MAP.get(emo.upper())
    if emotion is None:
        return None
    return actor, sentence, emotion, level.upper()


def build_manifest_crema(crema_root: Path, out_csv: Path,
                         nontarget_per_speaker_emo: int = 10,
                         min_seconds: float = 2.0,
                         dev_fraction: float = 0.17,
                         test_fraction: float = 0.17,
                         seed: int = 1234) -> int:
    """CREMA-D ``<actor>_<sentence>_<EMO>_<LEVEL>.wav`` -> target + nontarget rows.

    Replaces ESD as the bona fide side so the corpus rests on a real licence
    (ODbL 1.0 / DbCL 1.0, no request needed) instead of a pending grant — see
    docs/CORPUS_STRATEGY.md.

    Splits are **actor-disjoint** and deterministic: actors are sorted, shuffled
    with ``seed``, and cut into train/dev/test, so the audit's speaker-overlap check
    passes instead of warning. ``min_seconds`` drops clips too short for a stable
    score (CREMA-D runs from 1.3 s); the duration comes from the WAV header, so no
    audio is decoded. Clips whose header cannot be parsed are kept with a note
    rather than silently dropped.
    """
    crema_root = Path(crema_root)
    by_actor: dict[str, dict[str, list[tuple[Path, float]]]] = {}
    for wav in sorted(crema_root.rglob("*.wav")):
        parsed = parse_crema_name(wav.stem)
        if parsed is None:
            continue
        actor, _sentence, emotion, _level = parsed
        by_actor.setdefault(actor, {}).setdefault(emotion, []).append(
            (wav, _wav_seconds(wav)))
    if not by_actor:
        raise FileNotFoundError(
            f"no CREMA-D-style '<actor>_<sentence>_<EMO>_<LEVEL>.wav' files under {crema_root}")

    actors = sorted(by_actor)
    rng = random.Random(seed)
    rng.shuffle(actors)
    n = len(actors)
    n_test = max(1, round(test_fraction * n)) if n >= 3 else 0
    n_dev = max(1, round(dev_fraction * n)) if n >= 3 else 0
    split_of = {}
    for i, actor in enumerate(actors):
        split_of[actor] = "test" if i < n_test else ("dev" if i < n_test + n_dev else "train")

    kept = {a: {e: [w for w, d in clips if not (d == d) or d >= min_seconds]
                for e, clips in emo_map.items()}
             for a, emo_map in by_actor.items()}
    dropped_short = sum(1 for a in by_actor for e in by_actor[a]
                        for _w, d in by_actor[a][e] if d == d and d < min_seconds)

    rows: list[dict] = []
    pair_rng = random.Random(seed + 1)
    for actor in sorted(kept):
        split_name = split_of[actor]
        for emotion, clips in sorted(kept[actor].items()):
            for wav in clips:
                rows.append(dict(
                    trial_id=f"crema_{wav.stem}", split=split_name,
                    speaker_id=actor, claimed_speaker=actor, trial_type="target",
                    emotion=emotion, attack="none", dataset="crema-d",
                    path=str(wav), source_trial_id="",
                    notes="" if emotion in EMOSPOOF_CLASSES
                    else "not a wav2vec2-superb-er class"))
            if emotion not in EMOSPOOF_CLASSES:
                continue
            # Impostor utterances come from the SAME partition, as in the SASV
            # protocol: a nontarget trial pairs a test utterance with a model of
            # another evaluation speaker, so its audio must not be drawn from a
            # speaker the fusion or the countermeasure was fitted on.
            pool = [w for other, emo_map in kept.items()
                    if other != actor and split_of[other] == split_name
                    for w in emo_map.get(emotion, [])]
            take = pair_rng.sample(pool, min(nontarget_per_speaker_emo, len(pool)))
            for other_wav in take:
                rows.append(dict(
                    trial_id=f"crema_nt_{other_wav.stem}_as_{actor}",
                    split=split_name,
                    speaker_id=parse_crema_name(other_wav.stem)[0],
                    claimed_speaker=actor, trial_type="nontarget", emotion=emotion,
                    attack="none", dataset="crema-d", path=str(other_wav),
                    source_trial_id=f"crema_{other_wav.stem}",
                    notes="derived cross-speaker pair"))

    written = _write(rows, out_csv)
    if dropped_short:
        print(f"dropped {dropped_short} clip(s) shorter than {min_seconds:g} s")
    splits = {s: len({r['speaker_id'] for r in rows if r['split'] == s})
              for s in ("train", "dev", "test")}
    print(f"actors per split: {splits}")
    return written


def build_manifest_crema_clones(generated_csv: Path, bona_fide_csv: Path,
                                out_csv: Path) -> int:
    """``generate_clones.py`` output -> spoof rows for the CREMA-D corpus.

    The split is **inherited from the source clip**, never computed here: a clone
    of a test actor's utterance must be scored in the test split, otherwise the
    countermeasure is fitted on speakers it is later evaluated against. That is
    also why the bona fide manifest is required rather than optional.

    Rows whose source clip is absent from the bona fide manifest (for example a
    clip dropped by the minimum-duration filter) are skipped and counted; writing
    them with an invented split would be the silent version of the bug above.

    ``generated.csv`` is append-only, so the same output can appear twice after a
    manual re-run; duplicates are collapsed on ``out_path``.
    """
    bona = {r["trial_id"]: r for r in csv.DictReader(open(bona_fide_csv, encoding="utf-8"))
            if r.get("dataset") == "crema-d" and r.get("trial_type") == "target"}
    if not bona:
        raise SystemExit(f"{bona_fide_csv}: no CREMA-D target rows to inherit splits from")

    rows: list[dict] = []
    seen_out: set[str] = set()
    duplicates = unknown = 0
    with open(generated_csv, encoding="utf-8") as fh:
        for gen in csv.DictReader(fh):
            out_path = gen["out_path"]
            if out_path in seen_out:
                duplicates += 1
                continue
            seen_out.add(out_path)
            source_stem = Path(gen["source_path"]).stem
            source = bona.get(f"crema_{source_stem}")
            if source is None:
                unknown += 1
                continue
            system = gen["system"].strip().lower().replace(" ", "_")
            rows.append(dict(
                trial_id=gen["trial_id"], split=source["split"],
                speaker_id=source["speaker_id"], claimed_speaker=source["speaker_id"],
                trial_type="spoof", emotion=gen["emotion"], attack=system,
                dataset="crema-clones", path=out_path,
                source_trial_id=source["trial_id"],
                notes=f"generated by {system}"))
    if not rows:
        raise SystemExit(f"{generated_csv}: no usable generations (check --out paths)")
    print(f"kept {len(rows)} clones; skipped {duplicates} duplicate(s), "
          f"{unknown} with no source clip in the bona fide manifest")
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

    p = sub.add_parser("crema")
    p.add_argument("--crema-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--nontarget-per-speaker-emo", type=int, default=10)
    p.add_argument("--min-seconds", type=float, default=2.0)

    p = sub.add_parser("crema-clones")
    p.add_argument("--generated", required=True, type=Path,
                   help="generated.csv written by generate_clones.py")
    p.add_argument("--bona-fide", required=True, type=Path,
                   help="manifest from 'build_manifests.py crema'; splits are inherited")
    p.add_argument("--out", required=True, type=Path)

    args = parser.parse_args()
    if args.cmd == "asvspoof":
        n = build_manifest_asvspoof(args.la_root, args.out, args.nontarget_per_speaker)
    elif args.cmd == "esd":
        n = build_manifest_esd(args.esd_root, args.out, args.nontarget_per_speaker_emo)
    elif args.cmd == "crema":
        n = build_manifest_crema(args.crema_root, args.out,
                                 args.nontarget_per_speaker_emo, args.min_seconds)
    elif args.cmd == "crema-clones":
        n = build_manifest_crema_clones(args.generated, args.bona_fide, args.out)
    else:
        n = build_manifest_emospoof(args.emo_root, args.out)
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
