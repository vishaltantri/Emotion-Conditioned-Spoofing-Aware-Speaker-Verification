"""End-to-end evaluation pipeline.

The pipeline is deliberate about which split is used for what:

* fusion models and all scaling statistics are fitted on the **train** split;
* decision thresholds and any model selection are chosen on the **dev** split;
* the **test** split is touched once, for reporting, and never for tuning.

Every model in :data:`easv.fusion.MODEL_ORDER` is scored by identical metric
code so that baseline and proposed systems are directly comparable, and paired
grouped bootstrap is used for statistical comparisons.
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from easv.audit import AuditReport, audit_markdown
from easv import fusion as fusion_mod
from easv.fusion import MODEL_DESCRIPTIONS, MODEL_ORDER, FusionFeatures, create_model
from easv.metrics import (
    ASVSPOOF_C_FA,
    ASVSPOOF_C_FA_SPOOF,
    ASVSPOOF_C_MISS,
    ASVSPOOF_PI_NON,
    ASVSPOOF_PI_SPOOF,
    ASVSPOOF_PI_TARGET,
    accuracy_precision_recall_f1,
    brier_score,
    compute_eer,
    expected_calibration_error,
    grouped_bootstrap,
    min_a_dcf,
    min_tdcf,
    paired_grouped_bootstrap,
    rates_at_threshold,
    threshold_at_far,
)
from easv.scores import JoinedData, TrialBatch

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 7,
    "splits": {"train": "train", "dev": "dev", "test": "test"},
    # Generalisation folds: a ground-truth emotion named here is excluded from
    # fusion fitting and from dev threshold selection (test keeps every trial and
    # the per-emotion breakdown carries the held-out condition's numbers). Pair
    # with train_cm_indomain.py --holdout-emotion, which applies the same
    # exclusion to the countermeasure.
    "folds": {"holdout_emotion": None},
    "models": list(MODEL_ORDER),
    "training": {
        "l2": 1e-3,
        "max_iter": 2000,
        "use_uncertainty": True,
        # "full" uses the whole emo_* probability vector; "confidence_only"
        # replaces it with the single maximum-probability scalar. That is the
        # spoof-cue ablation: if the emotion arms' gain survives on one scalar,
        # it comes from the SER being confidently wrong on spoofed audio, not
        # from emotion information.
        "emotion_features": "full",
        "gated": {
            "hidden": 8,
            "lr": 0.05,
            "epochs": 600,
            "patience": 40,
            "validation_fraction": 0.15,
            "early_stopping": True,
        },
    },
    "threshold": {"policy": "eer", "far_target": 0.01},
    "metrics": {
        "tdcf_version": "2021",
        "pi_target": ASVSPOOF_PI_TARGET,
        "pi_non": ASVSPOOF_PI_NON,
        "pi_spoof": ASVSPOOF_PI_SPOOF,
        "c_miss": ASVSPOOF_C_MISS,
        "c_fa": ASVSPOOF_C_FA,
        "c_fa_spoof": ASVSPOOF_C_FA_SPOOF,
    },
    "bootstrap": {"n_resamples": 300, "seed": 7, "level": 0.95},
    "report": {"breakdown_by": ["emotion", "attack"], "baseline_model": "linear"},
    "provenance": {
        "kind": "unspecified",
        "note": "Record dataset licenses, versions and split provenance before citing results.",
    },
}


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(base))
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    path: str | Path | Sequence[str | Path] | None = None,
    override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the default config, then apply config files left to right, then overrides.

    Later files win, so a partial config (an ablation) can be layered on top of a
    base config that fixes the shared hyperparameters.
    """

    config = deepcopy(DEFAULT_CONFIG)
    paths: Sequence[str | Path]
    if path is None:
        paths = ()
    elif isinstance(path, (str, Path)):
        paths = (path,)
    else:
        paths = tuple(path)
    for one in paths:
        loaded = yaml.safe_load(Path(one).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError(f"config {one} must be a mapping")
        config = _merge(config, loaded)
    if override:
        config = _merge(config, override)
    unknown = set(config.get("models", [])) - set(MODEL_ORDER)
    if unknown:
        raise ValueError(f"unknown models in config: {sorted(unknown)}")
    if config["threshold"]["policy"] not in ("eer", "target_far"):
        raise ValueError("threshold.policy must be 'eer' or 'target_far'")
    return config


def _features(batch: TrialBatch, config: Mapping[str, Any] | None = None) -> FusionFeatures:
    mode = ((config or {}).get("training", {}) or {}).get("emotion_features", "full")
    if mode == "confidence_only" and batch.emotion_probs is not None:
        # One column: the SER's maximum predicted probability. Entropy is
        # undefined for a 1-of-1 taxonomy and confidence would duplicate the
        # column, so uncertainty features are disabled for this mode (enforced
        # in _model_kwargs).
        return FusionFeatures(
            speaker=batch.speaker,
            cm=batch.cm,
            emotion=np.max(batch.emotion_probs, axis=1, keepdims=True),
            emotion_labels=("ser_confidence",),
        )
    return FusionFeatures(
        speaker=batch.speaker,
        cm=batch.cm,
        emotion=batch.emotion_probs,
        emotion_labels=batch.emotion_labels,
    )


def _target_labels(batch: TrialBatch) -> np.ndarray:
    return batch.is_target


def _select_threshold(
    scores: np.ndarray, labels: np.ndarray, config: Mapping[str, Any]
) -> tuple[float, dict[str, Any]]:
    policy = config["threshold"]["policy"]
    if policy == "eer":
        result = compute_eer(scores, labels)
        return result.threshold, {"policy": "eer", "dev_eer": result.eer}
    far_target = float(config["threshold"]["far_target"])
    threshold = threshold_at_far(scores, labels, far_target)
    if not np.isfinite(threshold):
        return threshold, {"policy": "target_far", "far_target": far_target, "reachable": False}
    return threshold, {"policy": "target_far", "far_target": far_target, "reachable": True}


@dataclass
class ModelResult:
    name: str
    description: str
    n_features: int
    training: dict[str, Any]
    threshold: dict[str, Any]
    test: dict[str, Any]
    bootstrap: dict[str, Any] = field(default_factory=dict)
    paired_vs_baseline: dict[str, Any] | None = None
    breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    gate_weights: dict[str, dict[str, float]] | None = None
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "n_features": self.n_features,
            "training": self.training,
            "threshold": self.threshold,
            "test": self.test,
            "bootstrap_sasv_eer": self.bootstrap,
            "paired_vs_baseline": self.paired_vs_baseline,
            "breakdown_sasv_eer": self.breakdown,
            "gate_weights": self.gate_weights,
            "errors": self.errors,
        }


def _model_kwargs(name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    confidence_only = training.get("emotion_features", "full") == "confidence_only"
    kwargs: dict[str, Any] = {
        "l2": training.get("l2", 1e-3),
        "max_iter": training.get("max_iter", 2000),
        "use_uncertainty": False if confidence_only else training.get("use_uncertainty", True),
    }
    gated = training.get("gated", {})
    for key in ("hidden", "lr", "epochs", "patience", "validation_fraction", "early_stopping"):
        if key in gated:
            kwargs[key] = gated[key]
    kwargs["seed"] = int(config.get("seed", 7))
    return kwargs


def _breakdown_by(
    scores: np.ndarray,
    labels: np.ndarray,
    values: Sequence[str],
    batch: TrialBatch,
    threshold: float,
) -> dict[str, dict[str, float]]:
    """Per-group behaviour on the test split.

    A single EER per attack family is meaningless (attack subsets contain only
    negatives), so each group reports what can legitimately be measured there:
    SASV-EER where both classes exist, acceptance rates at the dev threshold,
    and the attack-success rate for spoof families.
    """

    values = np.asarray(list(values), dtype=object)
    out: dict[str, dict[str, float]] = {}
    for value in np.unique(values):
        index = np.flatnonzero(values == value)
        block: dict[str, float] = {
            "n": int(index.size),
            "n_target": int(np.sum(labels[index])),
            "n_negative": int(np.sum(~labels[index])),
        }
        if labels[index].any() and not labels[index].all():
            block["sasv_eer"] = compute_eer(scores[index], labels[index]).eer
        if np.isfinite(threshold):
            rates = rates_at_threshold(scores[index], labels[index], threshold)
            block["target_acceptance"] = 1.0 - rates["frr"]
            block["negative_acceptance"] = rates["far"]
            spoof_index = index[batch.is_spoof[index]]
            if spoof_index.size:
                accepted = scores[spoof_index] >= threshold
                block["spoof_acceptance_rate"] = float(np.mean(accepted))
                block["n_spoof"] = int(spoof_index.size)
            target_index = index[batch.is_target[index]]
            if target_index.size:
                accepted = scores[target_index] >= threshold
                block["target_acceptance_rate"] = float(np.mean(accepted))
            nontarget_index = index[batch.is_nontarget[index]]
            if nontarget_index.size:
                accepted = scores[nontarget_index] >= threshold
                block["nontarget_acceptance_rate"] = float(np.mean(accepted))
        out[str(value)] = block
    return out


def _gate_weight_summary(
    model,
    batch: TrialBatch,
    groups: Sequence[str],
    config: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, float]] | None:
    if not isinstance(model, fusion_mod.EmotionGatedFusion) or not batch.has_emotion:
        return None
    w_s, w_c = model.gate_weights(_features(batch, config))
    values = np.asarray(list(groups), dtype=object)
    summary: dict[str, dict[str, float]] = {}
    for value in np.unique(values):
        index = np.flatnonzero(values == value)
        summary[str(value)] = {
            "mean_speaker_weight": float(np.mean(w_s[index])),
            "mean_cm_weight": float(np.mean(w_c[index])),
            "n": int(index.size),
        }
    return summary


def evaluate(
    joined: JoinedData,
    config: Mapping[str, Any] | None = None,
    audit: AuditReport | None = None,
) -> dict[str, Any]:
    """Fit every configured fusion model and return a JSON-serialisable report."""

    config = _merge(DEFAULT_CONFIG, config or {})
    splits = config["splits"]
    train = joined.batch(splits["train"])
    dev = joined.batch(splits["dev"])
    test = joined.batch(splits["test"])
    if len(test) == 0:
        raise ValueError(f"test split '{splits['test']}' has no joined trials")
    if len(dev) == 0:
        raise ValueError(f"dev split '{splits['dev']}' has no joined trials")
    if len(train) == 0:
        raise ValueError(f"train split '{splits['train']}' has no joined trials")

    fold_emotion = config.get("folds", {}).get("holdout_emotion") or None
    if fold_emotion:
        if fold_emotion not in set(test.emotions):
            raise ValueError(f"folds.holdout_emotion '{fold_emotion}' does not appear in the test split")
        train = train.subset([e != fold_emotion for e in train.emotions])
        dev = dev.subset([e != fold_emotion for e in dev.emotions])
        if len(train) == 0 or len(dev) == 0:
            raise ValueError("the holdout-emotion fold emptied the train or dev split")

    y_train = _target_labels(train)
    y_dev = _target_labels(dev)
    y_test = _target_labels(test)
    if len(np.unique(y_test)) < 2:
        raise ValueError("test split needs both target and non-target trials to compute SASV-EER")

    train_features = _features(train, config)
    dev_features = _features(dev, config)
    test_features = _features(test, config)

    results: dict[str, Any] = {
        "config": _jsonable(config),
        "splits": {
            name: {
                "n_trials": len(batch),
                "n_target": int(np.sum(batch.is_target)),
                "n_nontarget": int(np.sum(batch.is_nontarget)),
                "n_spoof": int(np.sum(batch.is_spoof)),
                "emotions": sorted(set(batch.emotions)),
                "attacks": sorted({a for a in batch.attacks if a not in ("", "none")}),
            }
            for name, batch in (("train", train), ("dev", dev), ("test", test))
        },
        "provenance": _jsonable(config.get("provenance", {})),
        "folds": {"holdout_emotion": fold_emotion},
        "models": [],
        "component_baselines": {},
        "audit": audit.as_dict() if audit is not None else None,
        "caveats": [],
    }

    # Component diagnostics: ASV-only and CM-only behaviour on the test split.
    bona = test.is_bona_fide
    sv = compute_eer(test.speaker[bona], test.is_target[bona])
    spf = compute_eer(test.cm, test.is_bona_fide)
    raw_tdcf = min_tdcf(
        test.speaker,
        test.cm,
        test.trial_types,
        version=config["metrics"]["tdcf_version"],
        **_tdcf_kwargs(config),
    )
    results["component_baselines"] = {
        "sv_eer_asv_only": sv.as_dict(),
        "spf_eer_cm_only": spf.as_dict(),
        "min_tdcf_raw_cm": raw_tdcf.as_dict(),
    }

    baseline_name = config["report"].get("baseline_model", "linear")
    test_scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    for name in config["models"]:
        model = create_model(name, **_model_kwargs(name, config))
        entry = ModelResult(
            name=name,
            description=MODEL_DESCRIPTIONS.get(name, ""),
            n_features=0,
            training={},
            threshold={},
            test={},
        )
        try:
            model.fit(train_features, y_train)
        except ValueError as exc:
            entry.errors.append(str(exc))
            results["models"].append(entry.as_dict())
            continue

        dev_scores = model.decision_function(dev_features)
        test_score = model.decision_function(test_features)
        test_scores[name] = test_score

        if not np.isfinite(dev_scores).all() or not np.isfinite(test_score).all():
            entry.errors.append("model produced non-finite scores")
            results["models"].append(entry.as_dict())
            continue

        threshold, threshold_meta = _select_threshold(dev_scores, y_dev, config)
        threshold_meta["dev"] = rates_at_threshold(dev_scores, y_dev, threshold) if np.isfinite(threshold) else None
        thresholds[name] = threshold

        test_eer = compute_eer(test_score, y_test)
        at_threshold = (
            rates_at_threshold(test_score, y_test, threshold)
            if np.isfinite(threshold)
            else {"threshold": None, "note": "no admissible threshold under the requested FAR budget"}
        )
        probs = fusion_mod._sigmoid(test_score)
        a_dcf = min_a_dcf(test_score, test.trial_types, **_adcf_kwargs(config))
        t_dcf = min_tdcf(
            test.speaker,
            test_score,
            test.trial_types,
            version=config["metrics"]["tdcf_version"],
            **_tdcf_kwargs(config),
        )

        entry.n_features = int(getattr(model, "n_features", 0)) if not isinstance(model, fusion_mod.EmotionGatedFusion) else int(
            getattr(model, "params", {}).get("A", np.zeros((0, 0))).shape[0]
        )
        entry.training = {
            "n_train": int(train.n),
            "final_train_loss": (model.history[-1] if isinstance(model, fusion_mod.EmotionGatedFusion) and model.history else (model.final_objective if hasattr(model, "final_objective") else None)),
            "best_epoch": getattr(model, "best_epoch", None),
            "iterations": getattr(model, "n_iter", None),
            "n_parameters": _n_parameters(model),
        }
        entry.threshold = threshold_meta
        entry.test = {
            "sasv_eer": test_eer.eer,
            "sasv_eer_threshold_oracle": test_eer.threshold,
            "min_a_dcf": a_dcf.min_dcf,
            "min_tdcf_fused_as_cm": t_dcf.min_tdcf,
            "brier": brier_score(probs, y_test),
            "ece_10bins": expected_calibration_error(probs, y_test, n_bins=10),
            "at_dev_threshold": at_threshold,
            "accuracy_precision_recall_f1": accuracy_precision_recall_f1(test_score, y_test, threshold)
            if np.isfinite(threshold)
            else None,
        }

        bootstrap_cfg = config["bootstrap"]
        entry.bootstrap = grouped_bootstrap(
            lambda index: compute_eer(test_score[index], y_test[index]).eer,
            test.claimed_speakers,
            n_resamples=int(bootstrap_cfg["n_resamples"]),
            seed=int(bootstrap_cfg["seed"]),
            level=float(bootstrap_cfg["level"]),
        )

        for label, values in zip(
            config["report"].get("breakdown_by", []),
            (test.emotions, test.attacks),
        ):
            entry.breakdown[label] = _breakdown_by(test_score, y_test, values, test, threshold)

        entry.gate_weights = _gate_weight_summary(model, test, test.emotions, config)
        results["models"].append(entry.as_dict())

        # Persisted by write_results from the "_models" payload.
        results.setdefault("_models", {})[name] = model.state_dict()

    # Paired comparisons against the configured baseline, same resampled groups.
    baseline_scores = test_scores.get(baseline_name)
    if baseline_scores is not None:
        for entry in results["models"]:
            name = entry["name"]
            if name == baseline_name or name not in test_scores:
                continue
            entry["paired_vs_baseline"] = {
                "baseline": baseline_name,
                **paired_grouped_bootstrap(
                    lambda index: compute_eer(test_scores[name][index], y_test[index]).eer,
                    lambda index: compute_eer(baseline_scores[index], y_test[index]).eer,
                    test.claimed_speakers,
                    n_resamples=int(config["bootstrap"]["n_resamples"]),
                    seed=int(config["bootstrap"]["seed"]),
                    level=float(config["bootstrap"]["level"]),
                ),
            }
            # Lower SASV-EER is better; the paired test is expressed as
            # baseline_eer - model_eer so that positive means the model helps.
            paired = entry["paired_vs_baseline"]
            lo, hi = paired.get("lo"), paired.get("hi")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                paired["lo"], paired["hi"] = -hi, -lo  # flipping the sign reorders the interval
            if isinstance(paired.get("mean_diff"), (int, float)):
                paired["mean_diff"] = -paired["mean_diff"]
            entry["paired_vs_baseline"]["note"] = (
                "positive mean_diff = proposed SASV-EER gain over the baseline; "
                "p_b_not_better is the one-sided paired p-value"
            )

    results["test_trial_scores"] = {
        "trial_ids": list(test.trial_ids),
        "models": {name: [float(v) for v in scores] for name, scores in test_scores.items()},
    }

    if fold_emotion:
        results["caveats"].append(
            f"Held-out-emotion fold: '{fold_emotion}' trials are excluded from fusion "
            "fitting and from dev threshold selection; test reports every trial, and the "
            "per-emotion breakdown carries the held-out condition's generalisation numbers."
        )
    if results["provenance"].get("kind") == "synthetic":
        results["caveats"].append(
            "Synthetic demonstration data: scores are simulated. Numbers validate the pipeline, "
            "not any claim about emotion-conditioned fusion on real speech."
        )
    if len(np.unique(train.claimed_speakers)) < 2:
        results["caveats"].append("Training split speaker coverage is too small for reliable fitting.")
    return results


def _adcf_kwargs(config: Mapping[str, Any]) -> dict[str, float]:
    kwargs = _tdcf_kwargs(config)
    kwargs["c_fa_non"] = kwargs.pop("c_fa")
    return kwargs


def _tdcf_kwargs(config: Mapping[str, Any]) -> dict[str, float]:
    metrics = config.get("metrics", {})
    return {
        "pi_target": float(metrics.get("pi_target", ASVSPOOF_PI_TARGET)),
        "pi_non": float(metrics.get("pi_non", ASVSPOOF_PI_NON)),
        "pi_spoof": float(metrics.get("pi_spoof", ASVSPOOF_PI_SPOOF)),
        "c_miss": float(metrics.get("c_miss", ASVSPOOF_C_MISS)),
        "c_fa": float(metrics.get("c_fa", ASVSPOOF_C_FA)),
        "c_fa_spoof": float(metrics.get("c_fa_spoof", ASVSPOOF_C_FA_SPOOF)),
    }


def _n_parameters(model) -> int:
    if isinstance(model, fusion_mod.EmotionGatedFusion):
        return int(sum(np.asarray(v).size for v in model.params.values()))
    weights = getattr(model, "weights", None)
    if weights is None:
        return 0
    return int(np.asarray(weights).size + 1)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_results(
    result: Mapping[str, Any],
    out_dir: str | Path,
    *,
    audit: AuditReport | None = None,
) -> dict[str, Path]:
    """Write ``results.json``, ``results.md``, per-trial scores and model files."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    models_payload = payload.pop("_models", {})
    trial_payload = payload.pop("test_trial_scores", None)

    paths: dict[str, Path] = {}

    models_dir = out_dir / "models"
    for name, state in models_payload.items():
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / f"{name}.json"
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        paths[f"model_{name}"] = path

    if trial_payload:
        path = out_dir / "test_trial_scores.csv"
        names = list(trial_payload["models"])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["trial_id"] + [f"{n}_score" for n in names])
            for i, trial_id in enumerate(trial_payload["trial_ids"]):
                writer.writerow([trial_id] + [f"{trial_payload['models'][n][i]:.6f}" for n in names])
        paths["trial_scores"] = path

    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    paths["results_json"] = json_path

    md_path = out_dir / "results.md"
    md_path.write_text(render_markdown(payload, audit=audit), encoding="utf-8")
    paths["results_md"] = md_path
    return paths


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) and np.isfinite(value) else "n/a"


def render_markdown(result: Mapping[str, Any], audit: AuditReport | None = None) -> str:
    lines: list[str] = ["# Evaluation results", ""]
    provenance = result.get("provenance", {})
    lines.append(f"Data provenance: **{provenance.get('kind', 'unspecified')}**")
    if provenance.get("note"):
        lines.append("")
        lines.append(f"> {provenance['note']}")
    fold_emotion = (result.get("folds") or {}).get("holdout_emotion")
    if fold_emotion:
        lines.append("")
        lines.append(
            f"**Held-out-emotion fold: `{fold_emotion}`** — excluded from the fusion fit's "
            "train and dev splits (and from whatever produced the countermeasure scores); "
            "the test split is untouched, so its trials are scored by a pipeline that never "
            "saw that emotion."
        )
    for caveat in result.get("caveats", []):
        lines.append("")
        lines.append(f"**Caveat:** {caveat}")

    lines.extend(["", "## Split summary", ""])
    lines.append("| split | target | nontarget | spoof | emotions | attacks |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split, block in result.get("splits", {}).items():
        lines.append(
            f"| {split} | {block['n_target']} | {block['n_nontarget']} | {block['n_spoof']} | "
            f"{len(block['emotions'])} | {len(block['attacks'])} |"
        )

    component = result.get("component_baselines", {})
    if component:
        lines.extend(["", "## Component baselines (test split)", ""])
        lines.append(f"- ASV-only SV-EER: {component.get('sv_eer_asv_only', {}).get('eer'):.4f}")
        lines.append(f"- CM-only SPF-EER: {component.get('spf_eer_cm_only', {}).get('eer'):.4f}")
        lines.append(f"- min t-DCF (raw CM as countermeasure): {component.get('min_tdcf_raw_cm', {}).get('min_tdcf'):.4f}")

    lines.extend(["", "## Fusion models (test split)", ""])
    lines.append(
        "| model | SASV-EER | 95% CI (grouped bootstrap) | dev threshold | FAR@thr | FRR@thr | min a-DCF | min t-DCF | ECE | params |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for entry in result.get("models", []):
        if entry.get("errors"):
            lines.append(f"| {entry['name']} | skipped: {'; '.join(entry['errors'])} | | | | | | | | |")
            continue
        boot = entry.get("bootstrap_sasv_eer") or {}
        threshold = entry.get("threshold", {})
        test = entry.get("test", {})
        at = test.get("at_dev_threshold") or {}
        lo, hi = boot.get("lo"), boot.get("hi")
        ci = f"{lo:.4f} - {hi:.4f}" if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and np.isfinite(lo) and np.isfinite(hi) else "n/a"
        lines.append(
            "| {name} | {eer:.4f} | {ci} | {thr} | {far} | {frr} | {adcf:.4f} | {tdcf:.4f} | {ece:.3f} | {params} |".format(
                name=entry["name"],
                eer=test.get("sasv_eer", float("nan")),
                ci=ci,
                thr=("no admissible threshold" if threshold.get("reachable") is False else f"{threshold.get('policy')}"),
                far=(f"{at.get('far'):.4f}" if isinstance(at.get("far"), (int, float)) else "n/a"),
                frr=(f"{at.get('frr'):.4f}" if isinstance(at.get("frr"), (int, float)) else "n/a"),
                adcf=test.get("min_a_dcf", float("nan")),
                tdcf=test.get("min_tdcf_fused_as_cm", float("nan")),
                ece=test.get("ece_10bins", float("nan")),
                params=entry.get("training", {}).get("n_parameters", "-"),
            )
        )

    lines.extend(["", "## Paired comparisons vs baseline", ""])
    lines.append("| model | mean SASV-EER gain | 95% CI | p(baseline not worse) |")
    lines.append("|---|---:|---|---:|")
    for entry in result.get("models", []):
        paired = entry.get("paired_vs_baseline")
        if not paired:
            continue
        lines.append(
            f"| {entry['name']} vs {paired['baseline']} | {paired.get('mean_diff', float('nan')):.4f} | "
            f"{paired.get('lo', float('nan')):.4f} - {paired.get('hi', float('nan')):.4f} | "
            f"{paired.get('p_b_not_better', float('nan')):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Test-split breakdown",
            "",
            "Group-level behaviour at the dev-selected threshold. Attack families contain only spoof trials, "
            "so for them the meaningful quantity is the attack-success (spoof-acceptance) rate, not an EER.",
        ]
    )
    for entry in result.get("models", []):
        breakdown = entry.get("breakdown_sasv_eer") or {}
        if not breakdown:
            continue
        lines.extend(["", f"**{entry['name']}**"])
        for label, groups in breakdown.items():
            lines.extend(["", f"by {label}:", "", "| group | n | SASV-EER | target accept | negative accept | spoof accept |", "|---|---:|---:|---:|---:|---:|"])
            for group in sorted(groups):
                stats = groups[group]
                lines.append(
                    f"| {group} | {int(stats.get('n', 0))} | {_fmt(stats.get('sasv_eer'))} | "
                    f"{_fmt(stats.get('target_acceptance_rate'))} | {_fmt(stats.get('negative_acceptance'))} | "
                    f"{_fmt(stats.get('spoof_acceptance_rate'))} |"
                )
        lines.append("")

    gate_reported = False
    for entry in result.get("models", []):
        weights = entry.get("gate_weights")
        if not weights:
            continue
        gate_reported = True
        lines.extend([f"**Gate weights by ground-truth emotion ({entry['name']})**", ""])
        lines.append("| emotion | mean speaker weight | mean CM weight | trials |")
        lines.append("|---|---:|---:|---:|")
        for emotion, stats in sorted(weights.items()):
            lines.append(
                f"| {emotion} | {stats['mean_speaker_weight']:.3f} | {stats['mean_cm_weight']:.3f} | {stats['n']} |"
            )
        lines.append("")
    if gate_reported:
        lines.append(
            "Gate weights show how much speaker and countermeasure evidence the model uses per emotion "
            "condition; a single global weight cannot express this."
        )

    if audit is not None:
        lines.extend(["", "## Manifest audit", "", audit_markdown(audit).rstrip()])

    lines.extend(
        [
            "",
            "## Reproducing",
            "",
            "Configurations live in `configs/`; per-trial test scores are in `test_trial_scores.csv` and fitted "
            "models in `models/*.json`. Re-run with:",
            "",
            "```bash",
            "easv evaluate --manifest <manifest.csv> --scores <scores.csv> --config <config.yaml> --out <dir>",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
