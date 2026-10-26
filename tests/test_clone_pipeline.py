"""Tests for the CREMA-D clone pipeline: planner, resume loop and quality gate.

The interesting failures here are quiet ones: reusing a cloning reference as a
synthesis target (which duplicates a transcript and inflates the corpus), a resume
loop that regenerates everything after a crash, and a gate that passes a corpus
whose clones are indistinguishable across generators. Each test pins one of those.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_clones as gc
import verify_clone_quality as vq

HEADER = ["trial_id", "split", "speaker_id", "claimed_speaker", "trial_type",
          "emotion", "attack", "dataset", "path", "source_trial_id", "notes"]


def _wav(path: Path, seconds: float) -> None:
    import wave
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(b"\x00\x00" * int(16000 * seconds))


@pytest.fixture
def crema_manifest(tmp_path: Path) -> Path:
    rows = []
    for actor in ("1001", "1002"):
        for emo_code, emotion in (("ANG", "angry"), ("HAP", "happy")):
            for sentence in ("IEO", "TAI", "IOM"):
                # The level token distinguishes clips of different lengths; the
                # builder only validates the four-part shape and the emotion code.
                for level, seconds in (("XX", 4.0), ("HI", 3.5), ("LO", 2.5)):
                    stem = f"{actor}_{sentence}_{emo_code}_{level}"
                    path = tmp_path / "AudioWAV" / f"{stem}.wav"
                    _wav(path, seconds)
                    rows.append(dict(
                        trial_id=f"crema_{stem}", split="train", speaker_id=actor,
                        claimed_speaker=actor, trial_type="target", emotion=emotion,
                        attack="none", dataset="crema-d", path=str(path),
                        source_trial_id="", notes=""))
    out = tmp_path / "m_crema.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return out


def test_plan_keeps_references_out_of_the_targets(crema_manifest, tmp_path):
    clips = gc.read_crema_targets(crema_manifest)
    transcripts = {"IEO": "It is eleven o'clock", "TAI": "That is exactly what happened",
                   "IOM": "I'm on my way to the meeting"}
    rows, report = gc.plan_generation(clips, transcripts, system="styletts2",
                                      out_root=tmp_path / "clones", refs_per_cell=1,
                                      targets_per_cell=10)

    assert rows, "plan must not be empty"
    assert report["cells"] == 4 and not report["skipped_cells"]
    refs = {r["ref_path"] for r in rows}
    sources = {r["source_path"] for r in rows}
    assert not (refs & sources), "a cloning reference must never be a synthesis target"
    assert {r["system"] for r in rows} == {"styletts2"}
    assert all(r["out_path"].endswith(".wav") for r in rows)
    # Deterministic: the same corpus must plan identically twice.
    again, _ = gc.plan_generation(clips, transcripts, system="styletts2",
                                  out_root=tmp_path / "clones", refs_per_cell=1,
                                  targets_per_cell=10)
    assert [r["trial_id"] for r in rows] == [r["trial_id"] for r in again]


def test_plan_requires_a_transcript_for_every_sentence(crema_manifest, tmp_path):
    clips = gc.read_crema_targets(crema_manifest)
    with pytest.raises(SystemExit) as excinfo:
        gc.plan_generation(clips, {"IEO": "only one"}, system="styletts2",
                           out_root=tmp_path / "clones")
    assert "IOM" in str(excinfo.value) or "TAI" in str(excinfo.value)


def test_plan_falls_back_to_the_longest_clip_in_a_short_cell(crema_manifest, tmp_path):
    """CREMA-D's median clip is 2.5 s, so the 3 s preference must not be fatal."""
    clips = gc.read_crema_targets(crema_manifest)
    transcripts = {"IEO": "a", "TAI": "b", "IOM": "c"}
    rows, report = gc.plan_generation(clips, transcripts, system="styletts2",
                                      out_root=tmp_path / "clones",
                                      min_ref_seconds=10.0, min_ref_floor=2.0)
    assert len(report["fallback_cells"]) == report["cells"] == 4
    assert report["min_ref_used"] == pytest.approx(4.0), "the 4 s clip is the longest"
    assert rows, "the fallback must still produce a plan"
    # Every row clones from a clip that actually exists, and the reference is not
    # itself a synthesis target.
    assert all(Path(r["ref_path"]).name.endswith("_XX.wav") for r in rows)
    assert not ({r["ref_path"] for r in rows} & {r["source_path"] for r in rows})


def test_plan_skips_cells_below_the_hard_floor(crema_manifest, tmp_path):
    clips = gc.read_crema_targets(crema_manifest)
    transcripts = {"IEO": "a", "TAI": "b", "IOM": "c"}
    rows, report = gc.plan_generation(clips, transcripts, system="styletts2",
                                      out_root=tmp_path / "clones",
                                      min_ref_seconds=10.0, min_ref_floor=10.0)
    assert rows == []
    assert len(report["skipped_cells"]) == report["cells"] == 4


def test_plan_strict_refs_refuses_to_fall_back(crema_manifest, tmp_path):
    clips = gc.read_crema_targets(crema_manifest)
    transcripts = {"IEO": "a", "TAI": "b", "IOM": "c"}
    with pytest.raises(SystemExit) as excinfo:
        gc.plan_generation(clips, transcripts, system="styletts2",
                           out_root=tmp_path / "clones",
                           min_ref_seconds=10.0, strict_refs=True)
    assert "nothing to clone from" in str(excinfo.value)


def test_run_resumes_instead_of_regenerating(crema_manifest, tmp_path):
    clips = gc.read_crema_targets(crema_manifest)
    transcripts = {"IEO": "a", "TAI": "b", "IOM": "c"}
    rows, _ = gc.plan_generation(clips, transcripts, system="cosyvoice",
                                 out_root=tmp_path / "clones", refs_per_cell=1,
                                 targets_per_cell=2)
    done = tmp_path / "clones" / "cosyvoice" / "generated.csv"
    calls: list[str] = []

    def backend(row):
        calls.append(row["trial_id"])
        Path(row["out_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(row["out_path"]).write_bytes(b"fake")

    written, skipped = gc.run(rows, "", done, backend=backend, verbose=False)
    assert written == len(rows) and skipped == 0

    written, skipped = gc.run(rows, "", done, backend=backend, verbose=False)
    assert written == 0 and skipped == len(rows)

    Path(rows[0]["out_path"]).unlink()  # a crash mid-run
    written, skipped = gc.run(rows, "", done, backend=backend, verbose=False)
    assert written == 1 and skipped == len(rows) - 1


# ----------------------------------------------------------------------
# Quality gate
# ----------------------------------------------------------------------
def _row(generator, emotion, speaker_score, cm_score, probs, trial_type="spoof"):
    return dict(trial_id=f"{generator}_{emotion}", split="test",
                trial_type=trial_type, emotion=emotion, generator=generator,
                speaker_score=speaker_score, cm_score=cm_score,
                emotion_probs=probs)


def _corpus(generator_acceptance: dict[str, float]) -> list[dict]:
    """Bona fide rows plus clones whose acceptance follows the given rates."""
    rows = [_row("none", "angry", 0.9, 3.0, {"emo_angry": 0.8, "emo_happy": 0.2},
                 trial_type="target")]
    for generator, rate in generator_acceptance.items():
        n = 20
        n_accept = int(round(rate * n))
        for i in range(n):
            # Clones: the SER confidently picks the wrong class (happy for an
            # angry clip), which is the pilot's "confidently wrong" behaviour.
            # ASV acceptance is driven by the speaker score; the countermeasure
            # detects every clone regardless, which is what a usable spoof corpus
            # looks like.
            rows.append(_row(generator, "angry", 0.9 if i < n_accept else 0.1, -2.0,
                             {"emo_angry": 0.05, "emo_happy": 0.90}))
    return rows


def test_gate_passes_a_corpus_that_reproduces_the_pilot():
    rows = _corpus({"cosyvoice": 0.6, "f5_tts": 0.5, "styletts2": 0.05})
    summary = vq.summarise(rows, threshold=0.5)
    assert summary["generators"]["cosyvoice"]["accept_rate"] == pytest.approx(0.6)
    assert summary["accept_spread_pts"] == pytest.approx(55.0)
    assert summary["ser_clones"]["accuracy"] == 0.0
    assert summary["ser_clones"]["mean_confidence"] == pytest.approx(0.90)

    checks = vq.gate(summary, min_spread_pts=10, max_ser_accuracy=0.5,
                     min_ser_confidence=0.6, max_cm_eer=0.05)
    assert all(c["pass"] for c in checks), checks
    assert "PASS" in vq.render_markdown(summary, checks)


def test_gate_fails_when_every_generator_behaves_the_same():
    rows = _corpus({"cosyvoice": 0.3, "f5_tts": 0.3, "styletts2": 0.3})
    summary = vq.summarise(rows, threshold=0.5)
    checks = vq.gate(summary, min_spread_pts=10, max_ser_accuracy=0.5,
                     min_ser_confidence=0.6, max_cm_eer=0.05)
    spread = next(c for c in checks if c["criterion"].startswith("generator spread"))
    assert spread["pass"] is False
    assert "FAIL" in vq.render_markdown(summary, checks)


def test_gate_fails_when_the_ser_recognises_the_clones():
    rows = _corpus({"cosyvoice": 0.6, "f5_tts": 0.5, "styletts2": 0.05})
    for row in rows:
        if row["trial_type"] == "spoof":
            row["emotion_probs"] = {"emo_angry": 0.92, "emo_happy": 0.08}
    summary = vq.summarise(rows, threshold=0.5)
    checks = vq.gate(summary, min_spread_pts=10, max_ser_accuracy=0.5,
                     min_ser_confidence=0.6, max_cm_eer=0.05)
    accuracy = next(c for c in checks if c["criterion"] == "SER accuracy on clones")
    assert accuracy["pass"] is False


# ----------------------------------------------------------------------
# Spoof rows from the generation log
# ----------------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bona(pair: tuple[str, str], split: str) -> dict:
    stem = f"{pair[0]}_IEO_{pair[1]}_XX"
    return dict(trial_id=f"crema_{stem}", split=split, speaker_id=pair[0],
                claimed_speaker=pair[0], trial_type="target", emotion="angry",
                attack="none", dataset="crema-d", path=f"/corpus/{stem}.wav",
                source_trial_id="", notes="")


def _gen(pair: tuple[str, str], system: str = "styletts2") -> dict:
    stem = f"{pair[0]}_IEO_{pair[1]}_XX"
    return dict(system=system, actor=pair[0], emotion="angry", sentence="IEO",
                text="It is eleven o'clock", ref_path=f"/corpus/{stem}.wav",
                source_path=f"/corpus/{stem}.wav",
                out_path=f"/clones/{system}/{pair[0]}/{stem}.wav",
                trial_id=f"crema_clone_{stem}_{system}")


def test_clone_manifest_inherits_the_source_split(tmp_path, capsys):
    import build_manifests as bm

    bona_csv = tmp_path / "m_crema.csv"
    _write_csv(bona_csv, [_bona(("1001", "ANG"), "train"),
                          _bona(("1002", "ANG"), "test")], bm.REQUIRED_HEADER)

    generated = tmp_path / "generated.csv"
    rows = [_gen(("1001", "ANG")), _gen(("1002", "ANG")),
            _gen(("1002", "ANG")),                     # duplicate output
            _gen(("1003", "ANG"))]                     # no source clip
    _write_csv(generated, rows, gc.PLAN_HEADER)

    out = tmp_path / "m_crema_clones.csv"
    n = bm.build_manifest_crema_clones(generated, bona_csv, out)
    captured = capsys.readouterr().out
    assert n == 2
    assert "1 duplicate" in captured and "1 with no source clip" in captured

    cloned = {r["speaker_id"]: r for r in csv.DictReader(open(out, encoding="utf-8"))}
    assert cloned["1001"]["split"] == "train"
    assert cloned["1002"]["split"] == "test", "a clone must never change split"
    for row in cloned.values():
        assert row["trial_type"] == "spoof"
        assert row["attack"] == "styletts2"
        assert row["dataset"] == "crema-clones"
        assert row["claimed_speaker"] == row["speaker_id"]
        assert row["source_trial_id"].startswith("crema_")


def test_merge_keeps_clones_with_their_speaker(tmp_path):
    import merge_manifests as mm
    import build_manifests as bm

    rows = [_bona(("1001", "ANG"), "train"), _bona(("1002", "ANG"), "test")]
    clone = _gen(("1002", "ANG"))
    rows.append(dict(trial_id=clone["trial_id"], split="train",
                     speaker_id="1002", claimed_speaker="1002",
                     trial_type="spoof", emotion="angry", attack="styletts2",
                     dataset="crema-clones", path=clone["out_path"],
                     source_trial_id="crema_1002_IEO_ANG_XX", notes=""))

    merged = mm.merge_rows(rows)
    clone_row = next(r for r in merged if r["trial_type"] == "spoof")
    assert clone_row["split"] == "test", "spoof rows follow the target split"
    assert "split demoted" in clone_row["notes"]
