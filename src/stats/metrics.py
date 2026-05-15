# MIT License. See LICENSE in repository root.
"""Extended evaluation metrics (F2, F3).

Adds to :mod:`src.models.classifiers` the three items listed in the Paper 1
experiment plan:

* Expected Calibration Error (ECE) — binned version following Guo et al. 2017.
* Stratified bootstrap confidence intervals (n = 1000 by default) for any
  scalar metric.
* DeLong test for ROC-AUC comparison of two correlated classifiers on the
  same evaluation set.  The implementation follows the fast algorithm of
  Sun & Xu (2014), matching the yandexdataschool/roc_comparison reference
  (MIT License).  We re-implement it here rather than vendor the dependency
  to keep the install footprint small; the algorithm is short enough that a
  unit test against a direct ``scipy.stats.mannwhitneyu`` cross-check keeps
  it honest.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Return the binned Expected Calibration Error.

    Parameters
    ----------
    y_true:
        Binary ground truth labels (``0`` / ``1``).
    scores:
        Predicted positive-class probabilities in ``[0, 1]``.
    n_bins:
        Number of equal-width bins over ``[0, 1]``.

    Returns
    -------
    Scalar ECE in ``[0, 1]``.  Smaller is better.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Assign each score to a bin; clip right edge so score == 1.0 lands in the
    # last bin rather than overflowing to index n_bins.
    bin_ids = np.clip(np.digitize(scores, bin_edges) - 1, 0, n_bins - 1)

    ece = 0.0
    n_total = len(scores)
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(scores[mask].mean())
        ece += (count / n_total) * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Stratified bootstrap
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapCI:
    """Two-sided percentile bootstrap confidence interval for a metric."""

    point: float
    lower: float
    upper: float
    n_resamples: int

    def as_tuple(self) -> tuple[float, float, float]:
        return self.point, self.lower, self.upper


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """Stratified (by label) percentile bootstrap CI for a metric.

    Each bootstrap draw samples positives and negatives independently with
    replacement at their original sample sizes.  This keeps class prevalence
    fixed across resamples, which matters for PR-AUC under heavy imbalance.

    Parameters
    ----------
    y_true, scores:
        Evaluation arrays.
    metric_fn:
        Any callable with signature ``(y_true, scores) -> float``.
    n_resamples:
        Number of bootstrap iterations.
    alpha:
        Two-sided significance; returned interval covers
        ``[alpha/2, 1 - alpha/2]`` quantiles.
    seed:
        RNG seed (deterministic).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape")

    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        raise ValueError("bootstrap requires at least one positive and one negative")

    point = float(metric_fn(y_true, scores))
    draws: list[float] = []
    for _ in range(n_resamples):
        pos_sample = rng.choice(pos_idx, size=pos_idx.size, replace=True)
        neg_sample = rng.choice(neg_idx, size=neg_idx.size, replace=True)
        idx = np.concatenate([pos_sample, neg_sample])
        try:
            v = float(metric_fn(y_true[idx], scores[idx]))
        except ValueError:
            # Extremely rare: a resample may collapse to a single class due to
            # grouping side effects; skip that draw.
            continue
        if np.isfinite(v):
            draws.append(v)
    if not draws:
        return BootstrapCI(point=point, lower=float("nan"), upper=float("nan"), n_resamples=0)

    arr = np.asarray(draws, dtype=float)
    lo = float(np.quantile(arr, alpha / 2.0))
    hi = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return BootstrapCI(point=point, lower=lo, upper=hi, n_resamples=len(draws))


# Convenience wrappers ------------------------------------------------------

def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores))


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, scores))


# ---------------------------------------------------------------------------
# DeLong test for paired ROC-AUC comparison
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Return the mid-rank transformation used by the DeLong machinery.

    Ties are broken by averaging ranks, matching the convention in Sun & Xu
    (2014) eq. 5.
    """
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j_end = i
        while j_end < n and z[j_end] == z[i]:
            j_end += 1
        # mid-rank over positions [i, j_end)
        for k in range(i, j_end):
            t[k] = 0.5 * (i + j_end - 1) + 1  # ranks are 1-based
        i = j_end
    out = np.empty(n, dtype=float)
    out[j] = t
    return out


def _delong_auc_covariance(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return AUC estimates and their 2x2 covariance matrix via DeLong.

    Implementation detail
    ---------------------
    This is the fast O((m + n) log(m + n)) variant.  Positives are indexed
    first; negatives second.  We compute mid-ranks within each class and in
    the combined sample to get structural components V10 / V01 per Sun & Xu
    (2014).  Returned covariance is the sample covariance of the two AUCs.
    """
    y_true = np.asarray(y_true).astype(int)
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1-D")

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    m = int(pos_mask.sum())
    n = int(neg_mask.sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong requires at least one positive and one negative")

    aucs = np.empty(2)
    v10 = np.empty((2, m))
    v01 = np.empty((2, n))

    for k, s in enumerate((scores_a, scores_b)):
        s = np.asarray(s, dtype=float)
        pos_scores = s[pos_mask]
        neg_scores = s[neg_mask]

        tx = _compute_midrank(pos_scores)
        ty = _compute_midrank(neg_scores)
        tz = _compute_midrank(np.concatenate([pos_scores, neg_scores]))

        aucs[k] = (tz[:m].sum() / (m * n)) - (m + 1.0) / (2.0 * n)
        v10[k, :] = (tz[:m] - tx) / n
        v01[k, :] = 1.0 - (tz[m:] - ty) / m

    sx = np.cov(v10)
    sy = np.cov(v01)
    cov = sx / m + sy / n
    # ``np.cov`` returns a scalar when the array is 1-D; guard for that.
    cov = np.atleast_2d(cov)
    return aucs, cov


@dataclass(frozen=True)
class DelongResult:
    auc_a: float
    auc_b: float
    delta: float
    z: float
    p_value: float


def delong_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    y_true: np.ndarray,
) -> DelongResult:
    """Two-sided paired DeLong test of ROC-AUC between classifiers A and B.

    Returns
    -------
    :class:`DelongResult` with the two AUCs, their difference
    (``auc_a - auc_b``), the z-statistic, and the two-sided p-value.
    """
    from scipy.stats import norm

    aucs, cov = _delong_auc_covariance(scores_a, scores_b, y_true)
    delta = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 0.0:
        # Numerically degenerate — usually means the two rankings are
        # essentially identical.  Return p = 1 and z = 0.
        return DelongResult(
            auc_a=float(aucs[0]),
            auc_b=float(aucs[1]),
            delta=delta,
            z=0.0,
            p_value=1.0,
        )
    z = delta / np.sqrt(var)
    p = 2.0 * (1.0 - float(norm.cdf(abs(z))))
    return DelongResult(
        auc_a=float(aucs[0]),
        auc_b=float(aucs[1]),
        delta=delta,
        z=float(z),
        p_value=p,
    )
