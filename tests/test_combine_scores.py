"""Tests for the score combiner: join semantics and refusal to fabricate scores."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "combine_scores.py"


def load_module():
    spec = importlib.util.spec_from_file_location("combine_scores", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def combiner():
    return load_module()


def test_joins_speaker_cm_and_emotion(combiner):
    base = [{"trial_id": "t1", "speaker_score": "0.5"}]
    cm = {"t1": "-2.0"}
    emo = {"t1": {"emo_angry": "0.1", "emo_happy": "0.9"}}
    rows, dropped_cm, missing_emo, cols = combiner.combine_rows(base, cm, emo)
    assert dropped_cm == 0 and missing_emo == 0
    assert cols == ["emo_angry", "emo_happy"]
    assert rows == [{"trial_id": "t1", "speaker_score": "0.5", "cm_score": "-2.0",
                     "emo_angry": "0.1", "emo_happy": "0.9"}]


def test_drops_trials_without_a_countermeasure_score(combiner):
    base = [{"trial_id": "t1", "speaker_score": "0.5"},
            {"trial_id": "t2", "speaker_score": "0.7"}]
    rows, dropped_cm, _, _ = combiner.combine_rows(base, {"t1": "1.0"})
    assert dropped_cm == 1
    assert [r["trial_id"] for r in rows] == ["t1"]
    # the surviving row keeps the real score; nothing is invented for t2
    assert rows[0]["cm_score"] == "1.0"


def test_counts_missing_emotion_but_keeps_the_trial(combiner):
    base = [{"trial_id": "t1", "speaker_score": "0.5"},
            {"trial_id": "t2", "speaker_score": "0.4"}]
    cm = {"t1": "1.0", "t2": "0.0"}
    emo = {"t1": {"emo_sad": "0.3"}}
    rows, _, missing_emo, cols = combiner.combine_rows(base, cm, emo)
    assert missing_emo == 1
    assert cols == ["emo_sad"]
    assert len(rows) == 2
    t2 = next(r for r in rows if r["trial_id"] == "t2")
    assert "emo_sad" not in t2  # absent, not zero-filled


def test_no_emotion_file_produces_three_column_rows(combiner):
    rows, _, missing_emo, cols = combiner.combine_rows(
        [{"trial_id": "t1", "speaker_score": "0.5"}], {"t1": "2.0"})
    assert cols == [] and missing_emo == 0
    assert set(rows[0]) == {"trial_id", "speaker_score", "cm_score"}
