"""Tests for the evaluation helpers."""
from __future__ import annotations

import math

import numpy as np

from src.models.classifiers import apply_threshold, evaluate


def test_evaluate_perfect_scores() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate(y, scores)
    assert math.isclose(m.roc_auc, 1.0, abs_tol=1e-9)
    assert math.isclose(m.pr_auc, 1.0, abs_tol=1e-9)
    assert math.isclose(m.best_f1, 1.0, abs_tol=1e-9)


def test_evaluate_random_scores_has_auc_around_half() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000)
    scores = rng.random(2000)
    m = evaluate(y, scores)
    # Not meant to be tight — we only check "not an extreme".
    assert 0.4 < m.roc_auc < 0.6


def test_apply_threshold_is_monotone() -> None:
    scores = np.array([0.1, 0.5, 0.9])
    assert apply_threshold(scores, 0.0).sum() == 3
    assert apply_threshold(scores, 1.0).sum() == 0
