"""Integrated evaluation metrics for spoofing-aware speaker verification.

Definitions (see ``docs/PROTOCOL.md`` for the full protocol and its sources):

* ``EER`` - equal error rate for a binary decision score; scores are accepted
  when ``score >= threshold``. Reported with an interpolated threshold.
* ``SASV-EER`` - the SASV 2022 primary metric: a single binary EER where target
  trials are positive and *both* bona fide non-target and spoof trials are
  negative. It does not distinguish impostor from spoof attempts.
* ``SV-EER`` / ``SPF-EER`` - component diagnostics from the same challenge:
  ASV scores on bona fide trials, and countermeasure scores on bona fide vs
  spoof trials.
* ``min a-DCF`` - architecture-agnostic detection cost function (Shim et al.),
  sweeping one threshold across target / non-target / spoof trials. The name is
  *architecture-agnostic*, not "aggregated"; priors and costs are application
  assumptions, so defaults are explicit and configurable.
* ``min t-DCF`` - tandem detection cost function for an ASV+CM pair. Defaults
  follow the ASVspoof 2021 evaluation plan, which states the priors and costs
  are the same as ASVspoof 2019. ``version="2019"`` implements the legacy
  normalization that omits the additive ``C0`` term.

Nothing here reads labels from a score file, and no metric requires the fusion
model: metrics operate on arrays so that baselines and proposed systems are
scored by identical code.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Iterable, Sequence

import numpy as np

# ASVspoof 2019/2021 shared cost model (2021 evaluation plan, Table 1).
ASVSPOOF_PI_TARGET = 0.9405
ASVSPOOF_PI_NON = 0.0095
ASVSPOOF_PI_SPOOF = 0.0500
ASVSPOOF_C_MISS = 1.0
ASVSPOOF_C_FA = 10.0
ASVSPOOF_C_FA_SPOOF = 10.0

_EPS = 1e-12


@dataclass(frozen=True)
class EERResult:
    eer: float
    threshold: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DCFResult:
    min_dcf: float
    threshold: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TdcfResult:
    min_tdcf: float
    cm_threshold: float
    asv_threshold: float
    version: str
    note: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _as_arrays(scores: Sequence[float] | np.ndarray, labels: Sequence[bool] | np.ndarray):
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    if scores.shape != labels.shape:
        raise ValueError(f"score/label length mismatch: {scores.shape} vs {labels.shape}")
    if scores.size == 0:
        raise ValueError("empty score array")
    return scores, labels


def error_rate_curve(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[bool] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(thresholds, far, frr)`` using every unique score as threshold."""

    scores, labels = _as_arrays(scores, labels)
    if not labels.any() or labels.all():
        raise ValueError("EER requires at least one positive and one negative trial")
    pos = np.sort(scores[labels])
    neg = np.sort(scores[~labels])
    thresholds = np.unique(scores)
    # Accept when score >= threshold.
    far = (neg.size - np.searchsorted(neg, thresholds, side="left")) / neg.size
    frr = np.searchsorted(pos, thresholds, side="left") / pos.size
    return thresholds, far, frr


def compute_eer(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[bool] | np.ndarray,
) -> EERResult:
    """Equal error rate with linear interpolation around the crossing point."""

    thresholds, far, frr = error_rate_curve(scores, labels)
    delta = frr - far
    idx = int(np.argmax(delta >= 0)) if bool((delta >= 0).any()) else len(thresholds) - 1
    if idx == 0:
        return EERResult(float((far[0] + frr[0]) / 2.0), float(thresholds[0]))

    d0, d1 = float(delta[idx - 1]), float(delta[idx])
    if d1 == d0:
        alpha = 0.0
    else:
        alpha = max(0.0, min(1.0, -d0 / (d1 - d0)))
    eer = float(
        0.5
        * (
            (far[idx - 1] + alpha * (far[idx] - far[idx - 1]))
            + (frr[idx - 1] + alpha * (frr[idx] - frr[idx - 1]))
        )
    )
    threshold = float(thresholds[idx - 1] + alpha * (thresholds[idx] - thresholds[idx - 1]))
    return EERResult(eer, threshold)


def compute_min_dcf(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[bool] | np.ndarray,
    *,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    p_target: float = 0.5,
) -> DCFResult:
    """Minimum normalized detection cost for a single-score verification system."""

    scores, labels = _as_arrays(scores, labels)
    if not labels.any() or labels.all():
        raise ValueError("minDCF requires at least one positive and one negative trial")
    thresholds, far, frr = error_rate_curve(scores, labels)
    cost = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
    default = min(c_miss * p_target, c_fa * (1.0 - p_target))
    idx = int(np.argmin(cost))
    return DCFResult(float(cost[idx] / default), float(thresholds[idx]))


def rates_at_threshold(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[bool] | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """FAR/FRR/accuracy/precision/recall/F1 at a fixed threshold (accept >=)."""

    scores, labels = _as_arrays(scores, labels)
    predicted = scores >= threshold
    tp = float(np.sum(predicted & labels))
    fp = float(np.sum(predicted & ~labels))
    fn = float(np.sum(~predicted & labels))
    tn = float(np.sum(~predicted & ~labels))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "far": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "frr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "accuracy": (tp + tn) / max(scores.size, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_target": int(np.sum(labels)),
        "n_nontarget": int(np.sum(~labels)),
        "n_accept": int(np.sum(predicted)),
    }


def threshold_at_far(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[bool] | np.ndarray,
    target_far: float,
) -> float:
    """Smallest threshold whose FAR is at or below ``target_far``.

    Selecting the smallest admissible threshold maximizes target acceptance
    while respecting the security budget. Returns ``inf`` when the budget is
    unreachable (no threshold admits acceptably few impostors).
    """

    scores, labels = _as_arrays(scores, labels)
    thresholds, far, _ = error_rate_curve(scores, labels)
    ok = np.flatnonzero(far <= target_far + 1e-12)
    if ok.size == 0:
        return float("inf")
    return float(thresholds[ok[0]])


def accuracy_precision_recall_f1(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[bool] | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    rates = rates_at_threshold(scores, labels, threshold)
    return {k: rates[k] for k in ("accuracy", "precision", "recall", "f1", "threshold")}


def brier_score(probabilities: Sequence[float] | np.ndarray, labels: Sequence[bool] | np.ndarray) -> float:
    probs, labels = _as_arrays(probabilities, labels)
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("Brier score requires probabilities in [0, 1]")
    return float(np.mean((probs - labels.astype(float)) ** 2))


def expected_calibration_error(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[bool] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error with equal-width confidence bins."""

    probs, labels = _as_arrays(probabilities, labels)
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("ECE requires probabilities in [0, 1]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = probs.size
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            in_bin = (probs >= lo) & (probs <= hi)
        else:
            in_bin = (probs >= lo) & (probs < hi)
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        conf = float(np.mean(probs[in_bin]))
        acc = float(np.mean(labels[in_bin].astype(float)))
        ece += (count / total) * abs(conf - acc)
    return float(ece)


def min_a_dcf(
    scores: Sequence[float] | np.ndarray,
    trial_types: Sequence[str],
    *,
    pi_target: float = ASVSPOOF_PI_TARGET,
    pi_non: float = ASVSPOOF_PI_NON,
    pi_spoof: float = ASVSPOOF_PI_SPOOF,
    c_miss: float = ASVSPOOF_C_MISS,
    c_fa_non: float = ASVSPOOF_C_FA,
    c_fa_spoof: float = ASVSPOOF_C_FA_SPOOF,
) -> DCFResult:
    """Minimum normalized architecture-agnostic detection cost function."""

    scores = np.asarray(scores, dtype=float).ravel()
    types = np.asarray(list(trial_types), dtype=object).ravel()
    if scores.shape != types.shape:
        raise ValueError("score/trial_type length mismatch")
    is_target = types == "target"
    is_non = types == "nontarget"
    is_spoof = types == "spoof"
    for name, mask in (("target", is_target), ("nontarget", is_non), ("spoof", is_spoof)):
        if not mask.any():
            raise ValueError(f"a-DCF requires at least one {name} trial")

    thresholds = np.unique(scores)
    # Target miss rate and the two false-alarm rates are measured on their own trial sets.
    neg_non = scores[is_non]
    neg_spoof = scores[is_spoof]
    pos = scores[is_target]
    p_miss = np.searchsorted(np.sort(pos), thresholds, side="left") / pos.size
    p_fa_non = (neg_non.size - np.searchsorted(np.sort(neg_non), thresholds, side="left")) / neg_non.size
    p_fa_spoof = (
        neg_spoof.size - np.searchsorted(np.sort(neg_spoof), thresholds, side="left")
    ) / neg_spoof.size

    cost = (
        c_miss * pi_target * p_miss
        + c_fa_non * pi_non * p_fa_non
        + c_fa_spoof * pi_spoof * p_fa_spoof
    )
    default = min(c_miss * pi_target, c_fa_non * pi_non + c_fa_spoof * pi_spoof)
    idx = int(np.argmin(cost))
    return DCFResult(float(cost[idx] / default), float(thresholds[idx]))


def min_tdcf(
    speaker_scores: Sequence[float] | np.ndarray,
    cm_scores: Sequence[float] | np.ndarray,
    trial_types: Sequence[str],
    *,
    version: str = "2021",
    pi_target: float = ASVSPOOF_PI_TARGET,
    pi_non: float = ASVSPOOF_PI_NON,
    pi_spoof: float = ASVSPOOF_PI_SPOOF,
    c_miss: float = ASVSPOOF_C_MISS,
    c_fa: float = ASVSPOOF_C_FA,
    c_fa_spoof: float = ASVSPOOF_C_FA_SPOOF,
    asv_threshold: float | None = None,
) -> TdcfResult:
    """Tandem detection cost function for an ASV score plus a CM score.

    The ASV threshold defaults to the EER point between target and non-target
    trials, which is how the ASVspoof evaluation plan sets it. The CM threshold
    is swept (``min`` t-DCF) so an uncalibrated countermeasure is not penalised
    for score scaling alone.
    """

    if version not in ("2019", "2021"):
        raise ValueError("version must be '2019' or '2021'")

    speaker = np.asarray(speaker_scores, dtype=float).ravel()
    cm = np.asarray(cm_scores, dtype=float).ravel()
    types = np.asarray(list(trial_types), dtype=object).ravel()
    if not (speaker.shape == cm.shape == types.shape):
        raise ValueError("speaker_scores, cm_scores and trial_types must align")

    is_target = types == "target"
    is_non = types == "nontarget"
    is_spoof = types == "spoof"
    is_bona = is_target | is_non
    for name, mask in (("target", is_target), ("nontarget", is_non), ("spoof", is_spoof)):
        if not mask.any():
            raise ValueError(f"t-DCF requires at least one {name} trial")

    if asv_threshold is None:
        asv_threshold = compute_eer(speaker[is_target | is_non], is_target[is_target | is_non]).threshold

    p_miss_asv = float(np.mean(speaker[is_target] < asv_threshold))
    p_fa_asv = float(np.mean(speaker[is_non] >= asv_threshold))
    p_fa_spoof_asv = float(np.mean(speaker[is_spoof] >= asv_threshold))

    thresholds = np.unique(cm)
    pos = np.sort(cm[is_bona])
    neg = np.sort(cm[is_spoof])
    p_miss_cm = np.searchsorted(pos, thresholds, side="left") / pos.size
    p_fa_cm = (neg.size - np.searchsorted(neg, thresholds, side="left")) / neg.size

    if version == "2021":
        c0 = pi_target * c_miss * p_miss_asv + pi_non * c_fa * p_fa_asv
        c1 = pi_target * c_miss - c0
        c2 = pi_spoof * c_fa_spoof * p_fa_spoof_asv
        denom = c0 + min(c1, c2)
        numerator = c0 + c1 * p_miss_cm + c2 * p_fa_cm
    else:
        # Legacy 2019 coefficients: no additive C0 term (see ASVspoof 2021 plan, App.).
        c1 = pi_target * (c_miss - c_miss * p_miss_asv) - pi_non * c_fa * p_fa_asv
        c2 = pi_spoof * c_fa_spoof * p_fa_spoof_asv
        denom = min(c1, c2)
        numerator = c1 * p_miss_cm + c2 * p_fa_cm

    if not np.isfinite(denom) or denom <= 0:
        return TdcfResult(
            float("nan"), float("nan"), float(asv_threshold), version, "degenerate cost coefficients"
        )
    curve = numerator / denom
    idx = int(np.argmin(curve))
    return TdcfResult(float(curve[idx]), float(thresholds[idx]), float(asv_threshold), version, None)


def grouped_bootstrap(
    metric_fn: Callable[[np.ndarray], float],
    groups: Sequence[str],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """Percentile bootstrap that resamples whole groups (e.g. claimed speakers).

    Trials from one speaker are correlated, so resampling trials would understate
    uncertainty. Groups are sampled with replacement and their trials pooled.
    """

    groups = np.asarray(list(groups), dtype=object)
    unique = np.unique(groups)
    if unique.size < 2:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_resamples": 0, "n_failed": 0}
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n_failed = 0
    for _ in range(n_resamples):
        picked = rng.integers(0, unique.size, size=unique.size)
        idx = np.concatenate([index_by_group[unique[i]] for i in picked])
        try:
            values.append(float(metric_fn(idx)))
        except ValueError:
            n_failed += 1
    if not values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_resamples": 0, "n_failed": n_failed}
    arr = np.asarray(values, dtype=float)
    lo, hi = np.percentile(arr, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {
        "mean": float(arr.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n_resamples": int(arr.size),
        "n_failed": int(n_failed),
    }


def paired_grouped_bootstrap(
    metric_a: Callable[[np.ndarray], float],
    metric_b: Callable[[np.ndarray], float],
    groups: Sequence[str],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """Paired bootstrap of ``metric_a - metric_b`` over resampled groups.

    The difference is ``metric_a - metric_b``; for error-rate metrics (lower is
    better) a negative difference favours ``metric_a``. ``p_b_not_better`` is
    the fraction of resamples in which ``metric_a`` is *not* better than
    ``metric_b`` (i.e. the difference is >= 0), a paired one-sided p-value.
    """

    groups = np.asarray(list(groups), dtype=object)
    unique = np.unique(groups)
    if unique.size < 2:
        return {"mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_b_not_better": float("nan")}
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(n_resamples):
        picked = rng.integers(0, unique.size, size=unique.size)
        idx = np.concatenate([index_by_group[unique[i]] for i in picked])
        try:
            diffs.append(float(metric_a(idx)) - float(metric_b(idx)))
        except ValueError:
            continue
    if not diffs:
        return {"mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_b_not_better": float("nan")}
    arr = np.asarray(diffs, dtype=float)
    lo, hi = np.percentile(arr, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {
        "mean_diff": float(arr.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_b_not_better": float(np.mean(arr >= 0.0)),
        "n_resamples": int(arr.size),
    }


def breakdown(
    metric_fn: Callable[[np.ndarray], float],
    labels: Iterable[str],
) -> dict[str, float]:
    """Compute a metric within each group of a categorical label list."""

    labels = np.asarray(list(labels), dtype=object)
    out: dict[str, float] = {}
    for value in np.unique(labels):
        idx = np.flatnonzero(labels == value)
        try:
            out[str(value)] = float(metric_fn(idx))
        except ValueError:
            continue
    return out
