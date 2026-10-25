import csv
import json

from easv.cli import main

FAST = [
    "training.gated.epochs=120",
    "training.gated.patience=40",
    "bootstrap.n_resamples=25",
]


def test_demo_command_runs_full_pipeline(tmp_path, capsys):
    out = tmp_path / "demo"
    code = main(["demo", "--out", str(out), "--seed", "5", "--config", "configs/demo.yaml", "--set", *FAST])
    captured = capsys.readouterr().out
    assert code == 0, captured
    assert (out / "manifest.csv").exists() or (out / "data" / "manifest.csv").exists()
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert len(payload["models"]) == 5
    assert payload["provenance"]["kind"] == "synthetic"
    assert (out / "results.md").exists()
    assert (out / "test_trial_scores.csv").exists()


def test_audit_command_on_generated_manifest(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out), "--seed", "6", "--set", *FAST]) == 0
    capsys.readouterr()
    manifest = out / "data" / "manifest.csv"
    code = main(["audit", "--manifest", str(manifest), "--out", str(tmp_path / "audit")])
    printed = capsys.readouterr().out
    assert code == 0, printed
    assert "| split |" in printed
    assert (tmp_path / "audit" / "audit.json").exists()


def test_audit_command_fails_on_broken_manifest(tmp_path, capsys):
    manifest = tmp_path / "broken.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial_id", "split", "speaker_id", "claimed_speaker", "trial_type", "emotion", "attack"])
        writer.writerow(["t1", "train", "a", "a", "target", "neutral", "none"])
    code = main(["audit", "--manifest", str(manifest)])
    capsys.readouterr()
    assert code == 2


def test_evaluate_and_predict_commands(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out), "--seed", "8", "--set", *FAST]) == 0
    capsys.readouterr()
    manifest = out / "data" / "manifest.csv"
    scores = out / "data" / "scores.csv"
    eval_out = tmp_path / "eval"
    code = main(
        [
            "evaluate",
            "--manifest",
            str(manifest),
            "--scores",
            str(scores),
            "--config",
            "configs/demo.yaml",
            "--out",
            str(eval_out),
            "--set",
            *FAST,
        ]
    )
    assert code == 0, capsys.readouterr().out

    predictions = tmp_path / "pred.csv"
    code = main(
        [
            "predict",
            "--model",
            str(eval_out / "models" / "emotion_gated.json"),
            "--manifest",
            str(manifest),
            "--scores",
            str(scores),
            "--split",
            "test",
            "--out",
            str(predictions),
        ]
    )
    assert code == 0, capsys.readouterr().out
    lines = predictions.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "trial_id,split,score,probability_accept"
    assert len(lines) > 1


def test_ablation_overlay_config_layers_on_the_base(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out), "--seed", "9", "--config", "configs/demo.yaml", "--set", *FAST]) == 0
    capsys.readouterr()

    ablation_out = tmp_path / "ablation"
    code = main(
        [
            "evaluate",
            "--manifest",
            str(out / "data" / "manifest.csv"),
            "--scores",
            str(out / "data" / "scores.csv"),
            "--config",
            "configs/demo.yaml",
            "--config",
            "configs/ablation_no_emotion_uncertainty.yaml",
            "--out",
            str(ablation_out),
            "--set",
            *FAST,
        ]
    )
    assert code == 0, capsys.readouterr().out
    payload = json.loads((ablation_out / "results.json").read_text(encoding="utf-8"))
    models = {entry["name"]: entry for entry in payload["models"]}
    # Without entropy/confidence columns the additive model needs two fewer parameters.
    assert models["emotion_additive"]["training"]["n_parameters"] == 8
    assert payload["config"]["training"]["use_uncertainty"] is False
    assert payload["config"]["training"]["gated"]["epochs"] == 120  # inherited from the base + --set


def test_manifest_template_command(tmp_path):
    target = tmp_path / "template.csv"
    assert main(["manifest-template", "--out", str(target)]) == 0
    assert target.read_text(encoding="utf-8").startswith("trial_id,split,speaker_id,claimed_speaker")


def test_repeated_set_flags_accumulate(tmp_path, capsys):
    """Two --set flags must both apply; the last one used to replace the first."""
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out), "--seed", "11", "--set", *FAST]) == 0
    capsys.readouterr()
    second = tmp_path / "fold"
    code = main(
        [
            "evaluate",
            "--manifest", str(out / "data" / "manifest.csv"),
            "--scores", str(out / "data" / "scores.csv"),
            "--config", "configs/demo.yaml",
            "--out", str(second),
            "--set", *FAST,
            "--set", "splits.test=test",
        ]
    )
    assert code == 0, capsys.readouterr().out
    config = json.loads((second / "results.json").read_text(encoding="utf-8"))["config"]
    assert config["splits"]["test"] == "test"
    assert config["training"]["gated"]["epochs"] == 120  # first flag survived
