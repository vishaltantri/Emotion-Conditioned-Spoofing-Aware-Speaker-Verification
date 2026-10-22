"""Tests for scripts/build_manifests.py using tiny synthetic corpus fixtures.

These fixtures mimic the real corpus layouts (ASVspoof 2019 LA protocols, ESD
English folders, EmoSpoof-TTS generator folders) so builders can be exercised
before the real data arrives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_manifests as bm


def _rows(csv_path: Path) -> list[dict]:
    import csv
    with open(csv_path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _validate_against_easv(csv_path: Path) -> None:
    """The generated CSV must load and validate in the real easv pipeline."""
    from easv import manifest
    trials = manifest.load_manifest(csv_path)
    assert trials, "no trials parsed"


# ----------------------------------------------------------------------
# ASVspoof 2019 LA fixture
# ----------------------------------------------------------------------
@pytest.fixture
def la_root_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "LA"
    (root / "ASVspoof2019_LA_cm_protocols").mkdir(parents=True)
    for split in ("train", "dev"):
        (root / f"ASVspoof2019_LA_{split}" / "flac").mkdir(parents=True)
        proto = (root / "ASVspoof2019_LA_cm_protocols"
                 / f"ASVspoof2019.LA.cm.{split}.trl.txt")
        lines = []
        for spk, i in (("LA_0011", range(2)), ("LA_0012", range(2))):
            for n in i:
                lines.append(f"{spk} - {spk}_{n:04d} bonafide -\n")
        lines.append("LA_0011 - LA_0123_9999 spoof A13\n")
        lines.append("LA_0012 - LA_0124_8888 spoof A07\n")
        proto.write_text("".join(lines), encoding="utf-8")
        for spk, i in (("LA_0011", range(2)), ("LA_0012", range(2))):
            for n in i:
                (root / f"ASVspoof2019_LA_{split}" / "flac"
                 / f"{spk}_{n:04d}.flac").touch()
        for spk, utt in (("LA_0011", "LA_0123_9999"), ("LA_0012", "LA_0124_8888")):
            (root / f"ASVspoof2019_LA_{split}" / "flac" / f"{utt}.flac").touch()
    return root


def test_asvspoof_builder(la_root_fixture: Path, tmp_path: Path) -> None:
    out = tmp_path / "asvspoof.csv"
    n = bm.build_manifest_asvspoof(la_root_fixture, out, nontarget_per_speaker=2)
    assert n > 0
    rows = _rows(out)
    types = {r["trial_type"] for r in rows}
    assert {"target", "spoof", "nontarget"} == types
    spoof = [r for r in rows if r["trial_type"] == "spoof"]
    assert all(r["attack"].startswith("A") for r in spoof)
    assert all(r["attack"] and r["attack"] != "none" for r in spoof)
    # target rows keep speaker == claimed speaker
    targets = [r for r in rows if r["trial_type"] == "target"]
    assert all(r["speaker_id"] == r["claimed_speaker"] for r in targets)
    # nontarget rows differ and note the derivation
    non = [r for r in rows if r["trial_type"] == "nontarget"]
    assert all(r["speaker_id"] != r["claimed_speaker"] for r in non)
    assert all("derived" in r["notes"] for r in non)
    _validate_against_easv(out)


# ----------------------------------------------------------------------
# ESD fixture
# ----------------------------------------------------------------------
@pytest.fixture
def esd_root_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "ESD"
    emotions = ["Angry", "Happy", "Sad", "Neutral"]
    speakers = ["0011", "0012", "0013", "0014", "0015", "0016"]
    for spk in speakers:
        for emo in emotions:
            d = root / "English" / spk / emo
            d.mkdir(parents=True)
            for i in range(2):
                (d / f"{emo[:3]}_{i}.wav").touch()
    return root


def test_esd_builder(esd_root_fixture: Path, tmp_path: Path) -> None:
    out = tmp_path / "esd.csv"
    n = bm.build_manifest_esd(esd_root_fixture, out, nontarget_per_speaker_emo=1)
    rows = _rows(out)
    types = {r["trial_type"] for r in rows}
    assert {"target", "nontarget"} == types
    # speaker-disjoint splits: 0011-0014 train, 0015-0016 dev (target rows only)
    by_spk_split = {r["speaker_id"]: r["split"] for r in rows
                    if r["trial_type"] == "target"}
    assert by_spk_split["0011"] == "train"
    assert by_spk_split["0015"] == "dev"
    # nontarget cross-speaker rows reference other speakers
    non = [r for r in rows if r["trial_type"] == "nontarget"]
    assert all(r["speaker_id"] != r["claimed_speaker"] for r in non)
    # emotions normalised to lowercase
    assert {r["emotion"] for r in rows} <= {"angry", "happy", "sad", "neutral"}
    _validate_against_easv(out)


# ----------------------------------------------------------------------
# EmoSpoof-TTS fixture
# ----------------------------------------------------------------------
@pytest.fixture
def emospoof_root_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "emospoof"
    structure = {
        "CosyVoice": {"0011": ["Angry", "Neutral"]},
        "F5TTS": {"0015": ["Happy"]},
        "StyleTTS2": {"0017": ["Sad"]},
    }
    for gen, spks in structure.items():
        for spk, emos in spks.items():
            for emo in emos:
                d = root / gen / spk / emo
                d.mkdir(parents=True)
                (d / f"{spk}_{emo}_{0}.wav").touch()
    return root


def test_emospoof_builder(emospoof_root_fixture: Path, tmp_path: Path) -> None:
    out = tmp_path / "emospoof.csv"
    n = bm.build_manifest_emospoof(emospoof_root_fixture, out)
    rows = _rows(out)
    attacks = {r["attack"] for r in rows}
    assert attacks == {"cosyvoice", "f5_tts", "styletts2"}
    splits_by_attack = {r["attack"]: r["split"] for r in rows}
    assert splits_by_attack["cosyvoice"] == "train"
    assert splits_by_attack["f5_tts"] == "dev"
    assert splits_by_attack["styletts2"] == "test"
    assert all(r["trial_type"] == "spoof" for r in rows)
    # emotion parsed from the path
    emo_by_spk = {(r["attack"], r["speaker_id"]): r["emotion"] for r in rows}
    assert emo_by_spk[("cosyvoice", "0011")] in {"angry", "neutral"}
    assert emo_by_spk[("f5_tts", "0015")] == "happy"
    _validate_against_easv(out)
