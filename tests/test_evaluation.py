import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from easv.audit import audit_manifest
from easv.demo import generate_demo_data
from easv.evaluation import evaluate, load_config, render_markdown, write_results
from easv.manifest import load_manifest
from easv.scores import JoinedData, join, load_scores

SMALL_CONFIG = {
    "training": {"l2": 0.003, "gated": {"epochs": 80, "patience": 40, "hidden": 2}},
    "bootstrap": {"n_resamples": 25},
    "provenance": {"kind": "synthetic", "note": "unit-test fixture"},
}


@pytest.fixture()
def small_dataset(tmp_path):
    data = generate_demo_data(tmp_path / "data", seed=3, n_train_speakers=8, n_dev_speakers=4, n_test_speakers=5)
    manifest = load_manifest(data.manifest_path)
    scores = load_scores(data.scores_path)
    return join(manifest, scores), manifest


def test_evaluate_reports_every_model(small_dataset):
    joined, manifest = small_dataset
    result = evaluate(joined, SMALL_CONFIG, audit=audit_manifest(manifest))
    names = [entry["name"] for entry in result["models"]]
    assert names == ["speaker_only", "linear", "emotion_additive", "emotion_interaction", "emotion_gated"]
    for entry in result["models"]:
        assert not entry["errors"], entry["errors"]
        assert 0.0 <= entry["test"]["sasv_eer"] <= 1.0
        assert np.isfinite(entry["test"]["min_a_dcf"])
        assert entry["training"]["n_train"] == joined.batch("train").n
        assert entry["threshold"]["policy"] == "eer"
    assert result["component_baselines"]["sv_eer_asv_only"]["eer"] >= 0
    assert any("Synthetic demonstration data" in caveat for caveat in result["caveats"])
    assert result["splits"]["test"]["n_spoof"] > 0


def test_thresholds_do_not_depend_on_test_labels_or_scores(small_dataset):
    joined, _ = small_dataset
    first = evaluate(joined, SMALL_CONFIG)

    rng = np.random.default_rng(0)
    shuffled = []
    for item in joined.items:
        if item.trial.split == "test":
            shuffled.append(replace(item, speaker_score=float(rng.normal()), cm_score=float(rng.normal())))
        else:
            shuffled.append(item)
    second = evaluate(
        JoinedData(items=tuple(shuffled), emotion_labels=joined.emotion_labels),
        SMALL_CONFIG,
    )
    for a, b in zip(first["models"], second["models"]):
        assert a["threshold"] == b["threshold"], f"{a['name']} threshold depended on test data"


def test_emotion_conditioned_models_use_test_emotion_breakdown(small_dataset):
    joined, _ = small_dataset
    result = evaluate(joined, SMALL_CONFIG)
    gated = next(entry for entry in result["models"] if entry["name"] == "emotion_gated")
    emotions = gated["breakdown_sasv_eer"]["emotion"]
    assert set(emotions) >= {"neutral", "angry"}
    assert "surprise" in emotions
    assert gated["gate_weights"] is not None
    assert all("mean_cm_weight" in stats for stats in gated["gate_weights"].values())


def test_target_far_threshold_policy_is_respected(small_dataset):
    joined, _ = small_dataset
    config = dict(SMALL_CONFIG)
    config["threshold"] = {"policy": "target_far", "far_target": 0.2}
    result = evaluate(joined, config)
    for entry in result["models"]:
        threshold = entry["threshold"]
        assert threshold["policy"] == "target_far"
        # The security budget is enforced on the *dev* split, where the threshold
        # is chosen; test FAR is reported, not constrained.
        assert threshold["dev"]["far"] <= 0.2 + 1e-9
        assert entry["test"]["at_dev_threshold"]["far"] >= 0.0


def test_models_are_skipped_without_emotion_scores(small_dataset, tmp_path):
    joined, _ = small_dataset
    stripped = tmp_path / "scores_no_emotion.csv"
    with stripped.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial_id", "speaker_score", "cm_score"])
        for item in joined.items:
            writer.writerow([item.trial.trial_id, item.speaker_score, item.cm_score])
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial_id", "split", "speaker_id", "claimed_speaker", "trial_type", "emotion", "attack"])
        for item in joined.items:
            t = item.trial
            writer.writerow([t.trial_id, t.split, t.speaker_id, t.claimed_speaker, t.trial_type, t.emotion, t.attack])

    manifest = load_manifest(manifest_path)
    scores = load_scores(stripped)
    result = evaluate(join(manifest, scores), SMALL_CONFIG)
    by_name = {entry["name"]: entry for entry in result["models"]}
    assert not by_name["linear"]["errors"]
    for name in ("emotion_additive", "emotion_interaction", "emotion_gated"):
        assert by_name[name]["errors"]


def test_write_results_creates_all_artifacts(small_dataset, tmp_path):
    joined, manifest = small_dataset
    result = evaluate(joined, SMALL_CONFIG)
    paths = write_results(result, tmp_path / "out", audit=audit_manifest(manifest))
    assert paths["results_json"].exists()
    assert paths["results_md"].exists()
    assert paths["trial_scores"].exists()
    payload = json.loads(paths["results_json"].read_text(encoding="utf-8"))
    assert payload["models"]
    assert "_models" not in payload
    for name in ("speaker_only", "linear", "emotion_additive", "emotion_interaction", "emotion_gated"):
        assert (tmp_path / "out" / "models" / f"{name}.json").exists()
    rows = paths["trial_scores"].read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == joined.batch("test").n + 1
    markdown = paths["results_md"].read_text(encoding="utf-8")
    assert "SASV-EER" in markdown
    assert "Caveat" in markdown


def test_config_files_layer_left_to_right(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("training:\n  l2: 0.02\n  use_uncertainty: true\nmodels: [linear]\n", encoding="utf-8")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("training:\n  use_uncertainty: false\n", encoding="utf-8")

    config = load_config([base, overlay])
    assert config["training"]["l2"] == pytest.approx(0.02)  # from the base
    assert config["training"]["use_uncertainty"] is False  # overridden
    assert config["models"] == ["linear"]
    # Unmentioned keys still come from the defaults.
    assert config["threshold"]["policy"] == "eer"


def test_load_config_validates_inputs(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: [not_a_model]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)

    bad_policy = tmp_path / "policy.yaml"
    bad_policy.write_text("threshold:\n  policy: whatever\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad_policy)

    config = load_config(override={"splits": {"test": "eval"}})
    assert config["splits"]["test"] == "eval"
    assert config["splits"]["train"] == "train"


def test_render_markdown_includes_audit_section(small_dataset):
    joined, manifest = small_dataset
    report = audit_manifest(manifest)
    text = render_markdown(evaluate(joined, SMALL_CONFIG), audit=report)
    assert "## Manifest audit" in text
    assert "## Paired comparisons vs baseline" in text
