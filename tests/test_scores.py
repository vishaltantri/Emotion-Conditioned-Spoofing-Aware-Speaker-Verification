import csv
from pathlib import Path

import numpy as np
import pytest

from easv.manifest import load_manifest
from easv.scores import join, load_scores

HEADER = ["trial_id", "split", "speaker_id", "claimed_speaker", "trial_type", "emotion", "attack"]


def write_manifest(path: Path) -> Path:
    rows = [
        ["t1", "train", "a", "a", "target", "neutral", "none"],
        ["t2", "train", "b", "a", "nontarget", "happy", "none"],
        ["t3", "train", "b", "a", "spoof", "happy", "g1"],
        ["e1", "test", "c", "c", "target", "sad", "none"],
        ["e2", "test", "d", "c", "nontarget", "sad", "none"],
        ["e3", "test", "d", "c", "spoof", "sad", "g2"],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def write_scores(path: Path, rows: list[list[str]], header=None) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header or ["trial_id", "speaker_score", "cm_score", "emo_neutral", "emo_happy"])
        writer.writerows(rows)
    return path


def test_load_scores_and_emotion_columns(tmp_path):
    path = write_scores(
        tmp_path / "s.csv",
        [
            ["e1", "1.0", "0.5", "0.8", "0.2"],
            ["e2", "-1.0", "0.4", "0.1", "0.9"],
        ],
    )
    table = load_scores(path)
    assert table.emotion_labels == ("neutral", "happy")
    assert len(table) == 2
    assert table.rows["e1"].emotion_probs == pytest.approx((0.8, 0.2))
    assert not table.errors


def test_emotion_probabilities_are_renormalised(tmp_path):
    path = write_scores(
        tmp_path / "s.csv",
        [
            ["e1", "1.0", "0.5", "2.0", "2.0"],
        ],
    )
    table = load_scores(path)
    assert any(f.code == "emotion_probs_renormalised" for f in table.findings)
    assert sum(table.rows["e1"].emotion_probs) == pytest.approx(1.0)


def test_bad_emotion_probability_is_rejected(tmp_path):
    path = write_scores(
        tmp_path / "s.csv",
        [
            ["e1", "1.0", "0.5", "-0.1", "1.1"],
        ],
    )
    table = load_scores(path)
    assert any(f.code == "bad_emotion_probability" for f in table.errors)


def test_missing_score_columns_is_an_error(tmp_path):
    path = write_scores(tmp_path / "s.csv", [["e1", "1.0"]], header=["trial_id", "speaker_score"])
    table = load_scores(path)
    assert any(f.code == "missing_score_column" for f in table.errors)


def test_duplicate_score_row_is_dropped(tmp_path):
    path = write_scores(
        tmp_path / "s.csv",
        [["e1", "1.0", "0.5", "1.0", "0.0"], ["e1", "2.0", "0.5", "1.0", "0.0"]],
    )
    table = load_scores(path)
    assert len(table) == 1
    assert any(f.code == "duplicate_score_trial" for f in table.errors)


def test_join_reports_missing_and_unused(tmp_path):
    manifest = load_manifest(write_manifest(tmp_path / "m.csv"))
    scores = load_scores(
        write_scores(
            tmp_path / "s.csv",
            [
                ["e1", "1.0", "0.5", "1.0", "0.0"],
                ["e2", "-1.0", "0.4", "0.0", "1.0"],
                ["extra", "0.0", "0.0", "1.0", "0.0"],
            ],
        )
    )
    joined = join(manifest, scores)
    assert len(joined.items) == 2
    assert any(f.code == "missing_scores" and f.level == "error" for f in joined.findings)
    assert any(f.code == "unused_scores" for f in joined.findings)

    lenient = join(manifest, scores, require_all=False)
    assert any(f.code == "missing_scores" and f.level == "warning" for f in lenient.findings)


def test_batch_views_and_masks(tmp_path):
    manifest = load_manifest(write_manifest(tmp_path / "m.csv"))
    rows = [
        ["t1", "1.0", "0.5", "1.0", "0.0"],
        ["t2", "-1.0", "0.4", "1.0", "0.0"],
        ["t3", "-2.0", "-0.5", "1.0", "0.0"],
        ["e1", "1.0", "0.5", "1.0", "0.0"],
        ["e2", "-1.0", "0.4", "1.0", "0.0"],
        ["e3", "-2.0", "-0.5", "1.0", "0.0"],
    ]
    joined = join(manifest, load_scores(write_scores(tmp_path / "s.csv", rows)))
    test = joined.batch("test")
    assert test.n == 3
    assert test.is_target.tolist() == [True, False, False]
    assert test.is_bona_fide.tolist() == [True, True, False]
    assert test.is_spoof.tolist() == [False, False, True]
    assert test.is_nontarget.tolist() == [False, True, False]
    assert test.emotion_probs.shape == (3, 2)
    subset = test.subset(np.array([True, False, True]))
    assert subset.trial_ids == ("e1", "e3")
    assert subset.emotion_probs.shape == (2, 2)
    assert np.allclose(subset.speaker, [1.0, -2.0])


def test_batch_without_emotion_scores(tmp_path):
    manifest = load_manifest(write_manifest(tmp_path / "m.csv"))
    path = write_scores(
        tmp_path / "s.csv",
        [["t1", "1.0", "0.5"], ["t2", "-1.0", "0.4"], ["t3", "-2.0", "-0.5"]],
        header=["trial_id", "speaker_score", "cm_score"],
    )
    joined = join(manifest, load_scores(path))
    batch = joined.batch("train")
    assert batch.emotion_probs is None
    assert batch.has_emotion is False
