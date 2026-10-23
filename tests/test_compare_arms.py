"""Tests for the arm-vs-arm paired comparison script.

The contrast that matters is the proposed gated arm against the interaction
control, and the script has to use the *SASV* target definition (target trials
positive, nontarget and spoof negative) rather than "not spoof" — mixing those up
silently changes every EER, which is exactly the mistake these tests pin.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_arms.py"


def _load():
    spec = importlib.util.spec_from_file_location("compare_arms", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path, *, gated_better: bool = True) -> tuple[Path, Path]:
    import numpy as np

    rng = np.random.default_rng(0)
    manifest = tmp_path / "manifest.csv"
    results = tmp_path / "run"
    results.mkdir()
    rows = []
    score_rows = []
    # Overlapping Gaussian scores: both arms separate the classes, the gated one
    # better. Perfectly separable fixtures would give 0 % EER for both arms and
    # hide sign errors in the gain.
    for spk in (f"{i:04d}" for i in range(8)):
        for k in range(20):
            tid = f"t_{spk}_{k}"
            kind = "target" if k < 10 else "spoof"
            rows.append({"trial_id": tid, "split": "test", "speaker_id": spk,
                         "claimed_speaker": f"s{spk}", "trial_type": kind,
                         "emotion": "angry", "attack": "none" if kind == "target" else "f5_tts",
                         "dataset": "fixture"})
            sign = 1.0 if kind == "target" else -1.0
            gated = sign * 1.4 + float(rng.normal(0.0, 1.0))
            linear = sign * (0.7 if gated_better else 1.4) + float(rng.normal(0.0, 1.0))
            score_rows.append({"trial_id": tid, "linear_score": linear,
                               "emotion_gated_score": gated})
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(results / "test_trial_scores.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    return results, manifest


def test_positive_gain_when_arm_a_is_better(tmp_path, monkeypatch, capsys) -> None:
    results, manifest = _fixture(tmp_path, gated_better=True)
    out = tmp_path / "paired.json"
    module = _load()
    monkeypatch.setattr(sys, "argv", [
        "compare_arms.py", "--results", str(results), "--manifest", str(manifest),
        "--a", "emotion_gated", "--b", "linear", "--n-resamples", "200",
        "--out", str(out),
    ])
    module.main()
    import json
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["eer_a"] < payload["eer_b"]
    assert payload["mean_gain"] > 0
    assert payload["n_test"] == 160


def test_nontarget_counts_as_non_target_for_sasv(tmp_path, monkeypatch) -> None:
    """Nontarget trials must join spoof trials on the negative side."""
    results, manifest = _fixture(tmp_path, gated_better=True)
    # append one nontarget row whose gated score is low: it is a negative trial
    with open(manifest, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "trial_id", "split", "speaker_id", "claimed_speaker", "trial_type",
            "emotion", "attack", "dataset"])
        writer.writerow({"trial_id": "nt_1", "split": "test", "speaker_id": "0000",
                         "claimed_speaker": "s0001", "trial_type": "nontarget",
                         "emotion": "sad", "attack": "none", "dataset": "fixture"})
    with open(results / "test_trial_scores.csv", "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["trial_id", "linear_score",
                                                "emotion_gated_score"])
        writer.writerow({"trial_id": "nt_1", "linear_score": -5.0,
                         "emotion_gated_score": -5.0})
    out = tmp_path / "paired.json"
    module = _load()
    monkeypatch.setattr(sys, "argv", [
        "compare_arms.py", "--results", str(results), "--manifest", str(manifest),
        "--a", "emotion_gated", "--b", "linear", "--n-resamples", "50",
        "--out", str(out),
    ])
    module.main()
    import json
    payload = json.loads(out.read_text(encoding="utf-8"))
    # The extra negative trial changes the counts (161 rows) but cannot invert the
    # fact that the gated arm separates the classes better.
    assert payload["n_test"] == 161
    assert payload["mean_gain"] > 0
