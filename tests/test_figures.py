"""Tests for the figure generator's curve helper.

The DET figure is drawn from per-trial scores, and the first version of this helper
computed the rejection rate as the *accept* count — every DET curve came out
inverted (a system with 0.6 % EER plotted as if 95 % of targets were rejected).
These tests pin the direction by checking that the curve's crossing point
reproduces the EER the evaluation reported for the same scores.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"


def _load():
    spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_curve_crosses_near_the_equal_error_rate():
    rng = np.random.default_rng(3)
    n_pos, n_neg = 2000, 4000
    # Positives shifted higher than negatives: a usable but imperfect system.
    positives = rng.normal(1.0, 1.0, n_pos)
    negatives = rng.normal(0.0, 1.0, n_neg)
    scores = np.concatenate([positives, negatives])
    labels = np.concatenate([np.ones(n_pos, dtype=bool), np.zeros(n_neg, dtype=bool)])

    module = _load()
    far, frr = module.sasv_curve(scores, labels)
    crossing = far[np.argmin(np.abs(far - frr))]

    assert np.all(np.diff(far) >= 0)          # FAR rises as the threshold falls
    assert np.all(np.diff(frr) <= 0)          # FRR falls
    # Two unit-variance normals one standard deviation apart have EER = Phi(-0.5)
    # = 30.9 %, so the crossing point is an analytic check on the direction.
    assert abs(crossing - 0.3085) < 0.03
    # A curve with the direction inverted would cross at its mirror point instead,
    # so the extremes have to agree too.
    assert frr[0] > 0.6 and far[-1] > 0.9


def test_curve_tolerates_perfect_separation():
    module = _load()
    scores = np.array([3.0, 2.0, -1.0, -2.0])
    labels = np.array([True, True, False, False])
    far, frr = module.sasv_curve(scores, labels)
    # Curves are clipped to 1e-5 so they stay finite on probit/log axes.
    assert far[0] < 1e-4 and frr[-1] < 1e-4
    assert np.all(np.isfinite(far)) and np.all(np.isfinite(frr))
