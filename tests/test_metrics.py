import numpy as np
import pytest

from easv.metrics import (
    brier_score,
    compute_eer,
    compute_min_dcf,
    error_rate_curve,
    expected_calibration_error,
    grouped_bootstrap,
    min_a_dcf,
    min_tdcf,
    paired_grouped_bootstrap,
    rates_at_threshold,
    threshold_at_far,
)


def test_eer_perfect_separation_is_zero():
    scores = np.array([3.0, 2.5, 1.0, -1.0, -2.0, -3.0])
    labels = np.array([True, True, True, False, False, False])
    result = compute_eer(scores, labels)
    assert result.eer == pytest.approx(0.0)
    assert scores.min() <= result.threshold <= scores.max()


def test_eer_complements_under_score_sign_flip():
    # Acceptance is always "score >= threshold", so negating the score inverts
    # the system: FAR and FRR both become 1 - FAR and 1 - FRR, which makes the
    # EER cross to its complement. This guards the direction convention.
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, 200)
    labels = rng.random(200) < 0.4
    forward = compute_eer(scores, labels).eer
    backward = compute_eer(-scores, labels).eer
    assert forward + backward == pytest.approx(1.0)


def test_eer_matches_hand_computed_toy_case():
    scores = np.array([1.0, 0.0, 0.0, -1.0])
    labels = np.array([True, False, True, False])
    thresholds, far, frr = error_rate_curve(scores, labels)
    assert thresholds.tolist() == [-1.0, 0.0, 1.0]
    assert far.tolist() == [1.0, 0.5, 0.0]
    assert frr.tolist() == [0.0, 0.0, 0.5]
    result = compute_eer(scores, labels)
    # The crossing is interpolated halfway between thresholds 0.0 and 1.0.
    assert result.eer == pytest.approx(0.25)
    assert result.threshold == pytest.approx(0.5)


def test_eer_identical_distributions_is_near_half():
    rng = np.random.default_rng(1)
    labels = np.array([True] * 2000 + [False] * 2000)
    scores = rng.normal(0, 1, 4000)
    assert compute_eer(scores, labels).eer == pytest.approx(0.5, abs=0.03)


def test_eer_requires_both_classes():
    with pytest.raises(ValueError):
        compute_eer(np.arange(4.0), np.ones(4, dtype=bool))


def test_threshold_at_far_picks_the_lowest_admissible_score_value():
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    labels = np.array([True, False, True, False, False])
    threshold = threshold_at_far(scores, labels, 0.34)
    assert rates_at_threshold(scores, labels, threshold)["far"] <= 0.34 + 1e-12
    # The next admissible threshold down must be a score value that breaks the
    # budget (FAR is a step function of the threshold).
    thresholds, far, _ = error_rate_curve(scores, labels)
    index = int(np.flatnonzero(thresholds == threshold)[0])
    if index > 0:
        assert far[index - 1] > 0.34


def test_threshold_at_far_returns_inf_when_unreachable():
    # The highest-scoring trial is a negative, so FAR can never drop below 1.0.
    scores = np.array([1.0, 0.5])
    labels = np.array([False, True])
    assert np.isinf(threshold_at_far(scores, labels, 0.5))


def test_rates_at_threshold_counts():
    scores = np.array([2.0, 1.0, 0.0, -1.0])
    labels = np.array([True, False, True, False])
    rates = rates_at_threshold(scores, labels, 1.0)
    assert rates["precision"] == pytest.approx(0.5)
    assert rates["recall"] == pytest.approx(0.5)
    assert rates["f1"] == pytest.approx(0.5)
    assert rates["accuracy"] == pytest.approx(0.5)


def test_min_dcf_matches_curve_minimum():
    rng = np.random.default_rng(3)
    scores = np.concatenate([rng.normal(1, 1, 300), rng.normal(-0.5, 1, 700)])
    labels = np.concatenate([np.ones(300, bool), np.zeros(700, bool)])
    result = compute_min_dcf(scores, labels, c_miss=1, c_fa=1, p_target=0.5)
    thresholds, far, frr = error_rate_curve(scores, labels)
    best = np.min(0.5 * frr + 0.5 * far) / 0.5
    assert result.min_dcf == pytest.approx(best, rel=1e-9)


def test_min_a_dcf_is_zero_for_perfectly_separable_trials():
    scores = np.array([3.0, 2.0, 1.0, 0.0])
    types = ["target", "nontarget", "spoof", "spoof"]
    # Accepting only score >= 3 accepts the target and rejects everything else,
    # so the normalised cost reaches its minimum of zero.
    assert min_a_dcf(scores, types).min_dcf == pytest.approx(0.0)


def test_min_a_dcf_positive_when_classes_overlap():
    # The impostor outscores the target, so no threshold avoids a false accept.
    scores = np.array([0.9, 1.0, 0.5, 0.4])
    types = ["target", "nontarget", "spoof", "spoof"]
    result = min_a_dcf(scores, types)
    assert 0.0 < result.min_dcf <= 1.0 + 1e-12
    assert result.threshold == pytest.approx(0.9)


def test_min_tdcf_rewards_a_perfect_countermeasure():
    # The ASV branch alone lets some spoofs through (they imitate the target),
    # so C2 > 0 and the normalization is well defined; a perfect CM drives the
    # cost to zero.
    types = ["target"] * 20 + ["nontarget"] * 20 + ["spoof"] * 20
    rng = np.random.default_rng(4)
    speaker = np.concatenate([rng.normal(2.5, 0.5, 20), rng.normal(-2.5, 0.5, 20), rng.normal(1.0, 0.5, 20)])
    cm = np.concatenate([rng.normal(2.0, 0.3, 40), rng.normal(-2.0, 0.3, 20)])
    result = min_tdcf(speaker, cm, types, version="2021")
    assert result.min_tdcf == pytest.approx(0.0, abs=1e-6)
    assert result.version == "2021"
    assert min_tdcf(speaker, cm, types, version="2019").min_tdcf == pytest.approx(0.0, abs=1e-6)


def test_min_tdcf_reports_degenerate_coefficients():
    types = ["target"] * 10 + ["nontarget"] * 10 + ["spoof"] * 10
    speaker = np.concatenate([np.full(10, 3.0), np.full(20, -3.0)])
    cm = np.concatenate([np.full(20, 3.0), np.full(10, -3.0)])
    result = min_tdcf(speaker, cm, types, version="2021")
    assert np.isnan(result.min_tdcf)
    assert result.note is not None


def test_min_tdcf_rejects_unknown_version():
    with pytest.raises(ValueError):
        min_tdcf([1, 0], [1, 0], ["target", "spoof"], version="2026")


def test_min_tdcf_requires_all_trial_types():
    with pytest.raises(ValueError):
        min_tdcf([1, 0], [1, 0], ["target", "nontarget"])


def test_calibration_metrics():
    probs = np.array([0.5] * 100)
    labels = np.array([True] * 50 + [False] * 50)
    assert brier_score(probs, labels) == pytest.approx(0.25)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-12)
    assert expected_calibration_error(np.array([0.9] * 10), np.ones(10, bool)) == pytest.approx(0.1)


def test_calibration_metrics_reject_out_of_range():
    with pytest.raises(ValueError):
        brier_score(np.array([1.4]), np.array([True]))


def test_grouped_bootstrap_is_deterministic_and_grouped():
    rng = np.random.default_rng(5)
    scores = rng.normal(0, 1, 100)
    labels = np.concatenate([np.ones(40, bool), np.zeros(60, bool)])
    groups = np.array([f"spk{i % 10}" for i in range(100)])
    metric = lambda idx: compute_eer(scores[idx], labels[idx]).eer  # noqa: E731
    first = grouped_bootstrap(metric, groups, n_resamples=50, seed=11)
    second = grouped_bootstrap(metric, groups, n_resamples=50, seed=11)
    assert first == second
    assert first["lo"] <= first["hi"]


def test_paired_bootstrap_detects_better_system():
    rng = np.random.default_rng(6)
    groups = np.repeat([f"spk{i}" for i in range(20)], 10)
    labels = np.tile(np.concatenate([np.ones(4, bool), np.zeros(6, bool)]), 20)
    baseline = rng.normal(0, 1, 200)
    better = baseline + 1.5 * labels.astype(float)  # separates classes further
    metric = lambda idx: compute_eer(baseline[idx], labels[idx]).eer  # noqa: E731
    metric_better = lambda idx: compute_eer(better[idx], labels[idx]).eer  # noqa: E731
    result = paired_grouped_bootstrap(metric_better, metric, groups, n_resamples=60, seed=3)
    # Lower EER is better, so a negative difference favours `better`.
    assert result["mean_diff"] < 0
    assert result["p_b_not_better"] < 0.2
