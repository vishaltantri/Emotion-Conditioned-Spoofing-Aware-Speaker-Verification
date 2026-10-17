"""Command-line interface for the EASV pipeline.

Commands are journal-friendly: every one prints what it read, what it wrote and
whether it found problems, so a lab notebook entry can be reconstructed from the
terminal output.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from easv import __version__
from easv.audit import audit_manifest, audit_markdown
from easv.demo import generate_demo_data
from easv.evaluation import evaluate, load_config, write_results
from easv.fusion import FusionFeatures, load_model
from easv.manifest import load_manifest, manifest_template
from easv.scores import join, load_scores


def _print_findings(findings, limit: int = 40) -> None:
    for finding in findings[:limit]:
        print(f"  {finding.format()}")
    if len(findings) > limit:
        print(f"  ... {len(findings) - limit} more")


def _apply_overrides(config: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``a.b=value`` overrides, parsing the value as YAML."""

    out = json.loads(json.dumps(dict(config)))  # plain deep copy of JSON-able config
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got '{item}'")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        cursor = out
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return out


def cmd_audit(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    report = audit_manifest(
        manifest,
        allow_speaker_overlap=tuple(args.allow_speaker_overlap or ()),
    )
    text = audit_markdown(report)
    print(text)
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "audit.json").write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        (out_dir / "audit.md").write_text(text, encoding="utf-8")
        print(f"wrote {out_dir / 'audit.json'} and {out_dir / 'audit.md'}")
    print(f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    return 2 if report.errors else 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    scores = load_scores(args.scores)
    if scores.errors:
        print("score-file errors:")
        _print_findings(scores.errors)
        return 1
    joined = join(manifest, scores, require_all=not args.allow_missing_scores)
    report = audit_manifest(manifest, allow_speaker_overlap=tuple(args.allow_speaker_overlap or ()))
    if report.errors:
        print("manifest errors:")
        _print_findings(report.errors)
        return 1

    config = load_config(args.config)
    config = _apply_overrides(config, args.set or [])
    result = evaluate(joined, config, audit=report)
    paths = write_results(result, args.out, audit=report)

    print(result_markdown_summary(result))
    print(f"wrote {paths['results_json']} and {paths['results_md']}")
    if "trial_scores" in paths:
        print(f"wrote {paths['trial_scores']}")
    return 0


def result_markdown_summary(result: Mapping[str, Any]) -> str:
    lines = ["model                    SASV-EER (test)   params"]
    for entry in result.get("models", []):
        if entry.get("errors"):
            lines.append(f"{entry['name']:<24} skipped: {'; '.join(entry['errors'])}")
            continue
        eer = entry.get("test", {}).get("sasv_eer", float("nan"))
        params = entry.get("training", {}).get("n_parameters", "-")
        lines.append(f"{entry['name']:<24} {eer:.4f}           {params}")
    return "\n".join(lines)


def cmd_demo(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    data = generate_demo_data(out_dir / "data", seed=args.seed)
    print(f"generated {data.n_trials} synthetic trials in {data.manifest_path.parent}")

    config = load_config(args.config)
    config = _apply_overrides(
        config,
        [
            "provenance.kind=synthetic",
        ]
        + list(args.set or []),
    )
    manifest = load_manifest(data.manifest_path)
    scores = load_scores(data.scores_path)
    joined = join(manifest, scores)
    report = audit_manifest(manifest)
    result = evaluate(joined, config, audit=report)
    paths = write_results(result, out_dir, audit=report)
    print(result_markdown_summary(result))
    print(f"wrote {paths['results_json']} and {paths['results_md']}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    manifest = load_manifest(args.manifest)
    scores = load_scores(args.scores)
    joined = join(manifest, scores)
    batch = joined.batch(args.split)
    if len(batch) == 0:
        print(f"split '{args.split}' has no joined trials", file=sys.stderr)
        return 1
    features = FusionFeatures(
        speaker=batch.speaker,
        cm=batch.cm,
        emotion=batch.emotion_probs,
        emotion_labels=batch.emotion_labels,
    )
    z = model.decision_function(features)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial_id", "split", "score", "probability_accept"])
        probs = 1.0 / (1.0 + np.exp(-z))
        for trial_id, split, score, prob in zip(batch.trial_ids, [args.split] * len(batch), z, probs):
            writer.writerow([trial_id, split, f"{score:.6f}", f"{prob:.6f}"])
    print(f"wrote {out_path} ({len(batch)} trials, model={model.name})")
    return 0


def cmd_manifest_template(args: argparse.Namespace) -> int:
    text = manifest_template()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="easv", description="Emotion-conditioned spoofing-aware speaker verification")
    parser.add_argument("--version", action="version", version=f"easv {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="validate a manifest and report split/leakage statistics")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--out", help="directory for audit.json/audit.md")
    audit.add_argument("--allow-speaker-overlap", nargs="*", default=[], help="split pairs allowed to share speakers, e.g. train-dev")
    audit.set_defaults(func=cmd_audit)

    evaluate_cmd = sub.add_parser("evaluate", help="fit fusion models and evaluate on the test split")
    evaluate_cmd.add_argument("--manifest", required=True)
    evaluate_cmd.add_argument("--scores", required=True)
    evaluate_cmd.add_argument(
        "--config",
        action="append",
        default=None,
        help="config file; repeat to layer a partial config (ablation) on a base config",
    )
    evaluate_cmd.add_argument("--out", required=True)
    evaluate_cmd.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. --set splits.test=eval")
    evaluate_cmd.add_argument("--allow-missing-scores", action="store_true", help="tolerate manifest trials without scores")
    evaluate_cmd.add_argument("--allow-speaker-overlap", nargs="*", default=[])
    evaluate_cmd.set_defaults(func=cmd_evaluate)

    demo = sub.add_parser("demo", help="generate synthetic data and run the full pipeline")
    demo.add_argument("--out", default="results/demo")
    demo.add_argument("--config", action="append", default=None)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--set", nargs="*", default=[])
    demo.set_defaults(func=cmd_demo)

    predict = sub.add_parser("predict", help="apply a saved fusion model to a score file")
    predict.add_argument("--model", required=True, help="path to models/<name>.json")
    predict.add_argument("--manifest", required=True)
    predict.add_argument("--scores", required=True)
    predict.add_argument("--split", default="test")
    predict.add_argument("--out", required=True)
    predict.set_defaults(func=cmd_predict)

    template = sub.add_parser("manifest-template", help="print the manifest CSV header")
    template.add_argument("--out", default=None)
    template.set_defaults(func=cmd_manifest_template)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
