# MIT License. See LICENSE in repository root.
"""Unit tests for the extended statistics (F2, F3)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.stats.metrics import (
    delong_test,
    expected_calibration_error,
    pr_auc,
    roc_auc,
    stratified_bootstrap_ci,
)


def _synthetic(n: int = 500, auc_gap: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    base = rng.standard_normal(n) + y
    alt = base + auc_gap * (2 * y - 1)
    return y, base, alt


def test_ece_better_for_calibrated_scores() -> None:
    """ECE should be smaller for well-calibrated scores than miscalibrated ones."""
    rng = np.random.default_rng(0)
    n = 2000
    # Well calibrated: P(y=1|score=s) ≈ s by construction.
    s_cal = rng.uniform(0, 1, n)
    y_cal = (rng.random(n) < s_cal).astype(int)
    # Badly calibrated: push all scores to 0.9 regardless of label.
    y_bad = (rng.random(n) < 0.3).astype(int)
    s_bad = np.full(n, 0.9)

    e_cal = expected_calibration_error(y_cal, s_cal, n_bins=15)
    e_bad = expected_calibration_error(y_bad, s_bad, n_bins=15)
    assert e_cal < e_bad
    assert 0.0 <= e_cal <= 1.0
    assert 0.0 <= e_bad <= 1.0


def test_bootstrap_ci_contains_point() -> None:
    y, scores, _ = _synthetic()
    ci = stratified_bootstrap_ci(y, scores, roc_auc, n_resamples=200, seed=0)
    assert ci.lower <= ci.point <= ci.upper


def test_delong_paired_zero_delta_gives_high_p() -> None:
    y, scores, _ = _synthetic()
    res = delong_test(scores, scores, y)
    assert res.p_value > 0.9
    assert abs(res.delta) < 1e-8


def test_delong_significant_delta_detected() -> None:
    y, a, b = _synthetic(n=800, auc_gap=2.0, seed=1)
    res = delong_test(b, a, y)  # b is the "improved" classifier
    assert res.delta > 0
    assert res.p_value < 0.05


def test_roc_auc_wrapper_matches_sklearn() -> None:
    y, scores, _ = _synthetic()
    assert abs(roc_auc(y, scores) - roc_auc_score(y, scores)) < 1e-9


def test_pr_auc_wrapper_runs() -> None:
    y, scores, _ = _synthetic()
    v = pr_auc(y, scores)
    assert 0.0 <= v <= 1.0
