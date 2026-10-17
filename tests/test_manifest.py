import csv
from pathlib import Path

from easv.audit import audit_manifest, audit_markdown
from easv.manifest import load_manifest, manifest_template, TARGET, NON_TARGET, SPOOF

HEADER = ["trial_id", "split", "speaker_id", "claimed_speaker", "trial_type", "emotion", "attack"]


def write_manifest(path: Path, rows: list[list[str]], header=None) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header or HEADER)
        writer.writerows(rows)
    return path


def valid_rows() -> list[list[str]]:
    return [
        ["t1", "train", "a", "a", TARGET, "neutral", "none"],
        ["t2", "train", "b", "a", NON_TARGET, "happy", "none"],
        ["t3", "train", "b", "a", SPOOF, "happy", "g1"],
        ["d1", "dev", "c", "c", TARGET, "sad", "none"],
        ["d2", "dev", "d", "c", NON_TARGET, "sad", "none"],
        ["d3", "dev", "d", "c", SPOOF, "sad", "g2"],
        ["e1", "test", "e", "e", TARGET, "angry", "none"],
        ["e2", "test", "f", "e", NON_TARGET, "angry", "none"],
        ["e3", "test", "f", "e", SPOOF, "angry", "g3"],
    ]


def test_valid_manifest_has_no_errors(tmp_path):
    path = write_manifest(tmp_path / "m.csv", valid_rows())
    manifest = load_manifest(path)
    assert len(manifest) == 9
    assert manifest.errors == ()
    assert manifest.splits == ("dev", "test", "train")
    assert len(manifest.by_split("train")) == 3


def test_manifest_template_round_trips(tmp_path):
    path = tmp_path / "template.csv"
    path.write_text(manifest_template(), encoding="utf-8")
    manifest = load_manifest(path)
    assert manifest.errors == ()
    assert len(manifest) == 1


def test_missing_column_is_an_error(tmp_path):
    rows = [row[:-1] for row in valid_rows()]
    path = write_manifest(tmp_path / "m.csv", rows, header=HEADER[:-1])
    manifest = load_manifest(path)
    assert any(f.code == "missing_column" for f in manifest.errors)


def test_duplicate_trial_id_is_an_error(tmp_path):
    rows = valid_rows() + [valid_rows()[0]]
    manifest = load_manifest(write_manifest(tmp_path / "m.csv", rows))
    assert any(f.code == "duplicate_trial_id" for f in manifest.errors)


def test_target_speaker_mismatch_is_an_error(tmp_path):
    rows = valid_rows()
    rows[0] = ["t1", "train", "b", "a", TARGET, "neutral", "none"]
    manifest = load_manifest(write_manifest(tmp_path / "m.csv", rows))
    assert any(f.code == "target_speaker_mismatch" for f in manifest.errors)


def test_nontarget_same_speaker_is_an_error(tmp_path):
    rows = valid_rows()
    rows[1] = ["t2", "train", "a", "a", NON_TARGET, "happy", "none"]
    manifest = load_manifest(write_manifest(tmp_path / "m.csv", rows))
    assert any(f.code == "nontarget_speaker_overlap" for f in manifest.errors)


def test_spoof_requires_attack_and_bonafide_forbids_it(tmp_path):
    rows = valid_rows()
    rows[2] = ["t3", "train", "b", "a", SPOOF, "happy", "none"]
    rows[0] = ["t1", "train", "a", "a", TARGET, "neutral", "g1"]
    manifest = load_manifest(write_manifest(tmp_path / "m.csv", rows))
    codes = {f.code for f in manifest.errors}
    assert "spoof_without_attack" in codes
    assert "bonafide_with_attack" in codes


def test_bad_trial_type_and_empty_manifest(tmp_path):
    rows = valid_rows()
    rows[0] = ["t1", "train", "a", "a", "genuine", "neutral", "none"]
    manifest = load_manifest(write_manifest(tmp_path / "m.csv", rows))
    assert any(f.code == "bad_trial_type" for f in manifest.errors)

    empty = load_manifest(write_manifest(tmp_path / "empty.csv", []))
    assert any(f.code == "empty_manifest" for f in empty.errors)


def test_audit_reports_speaker_and_generator_overlap(tmp_path):
    rows = valid_rows()
    # Reuse training speaker 'a' in test and training generator g1 in test.
    rows[6] = ["e1", "test", "a", "a", TARGET, "angry", "none"]
    rows[8] = ["e3", "test", "f", "a", SPOOF, "angry", "g1"]
    manifest = load_manifest(write_manifest(tmp_path / "m.csv", rows))
    report = audit_manifest(manifest)
    codes = {f.code for f in report.findings}
    assert "speaker_overlap" in codes
    assert "generator_overlap" in codes
    assert report.counts["train"]["trial_type"][TARGET] == 1
    assert "| split |" in audit_markdown(report)


def test_audit_allows_declared_speaker_overlap(tmp_path):
    rows = valid_rows()
    rows[6] = ["e1", "test", "a", "a", TARGET, "angry", "none"]
    manifest = load_manifest(write_manifest(tmp_path / "m.csv", rows))
    report = audit_manifest(manifest, allow_speaker_overlap=("train-test",))
    assert not any(f.code == "speaker_overlap" for f in report.findings)


def test_audit_flags_unseen_emotion_and_missing_trial_type(tmp_path):
    rows = [row for row in valid_rows() if not (row[1] == "test" and row[4] == SPOOF)]
    rows[6][5] = "surprise"  # test target emotion absent from train
    manifest = load_manifest(write_manifest(tmp_path / "m.csv", rows))
    report = audit_manifest(manifest)
    codes = {f.code for f in report.findings}
    assert "missing_trial_type" in codes
    assert "unseen_emotion_in_test" in codes
    assert not report.ok
