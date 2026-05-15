# MIT License. See LICENSE in repository root.
"""Regression tests pinning the Paper 1 headline statistics.

These tests fail if the numbers reported in the manuscript drift away
from what can be reproduced from the stored DeLong matrix and the
published predictions.

The purpose is to protect against two classes of bug:

1. An accidental re-run of :mod:`src.stats.metrics` that would invalidate
   the numbers quoted in the paper without anyone noticing.
2. A hand-computed Holm correction that violates monotonicity — for
   example, an earlier draft transcribed ``p_Holm`` for seed 45 as
   ``0.105`` rather than the monotonicity-enforced ``0.135``.

Both regressions are silent failure modes, so we bolt them down here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from src.stats.metrics import delong_test

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "reports"
PREDICTIONS = REPO_ROOT / "data" / "processed" / "predictions"

# Seeds used throughout Paper 1.
SEEDS = [42, 43, 44, 45, 46]


@pytest.fixture(scope="module")
def delong_matrix() -> dict:
    path = REPORTS / "f10_delong_matrix.json"
    if not path.exists():  # pragma: no cover - defensive
        pytest.skip(f"delong matrix not found at {path}")
    with path.open() as fh:
        return json.load(fh)


def _ps_esm650_vs_simple(matrix: dict) -> np.ndarray:
    """Pull per-seed DeLong p-values for esm650_xgb vs simple_xgb (phage-level)."""
    out = []
    for seed in SEEDS:
        block = matrix["phage_component"][str(seed)]["esm650_xgb"]["simple_xgb"]
        out.append(block["p_value"])
    return np.asarray(out, dtype=float)


def test_esm650_vs_simple_raw_pvalues_are_stable(delong_matrix: dict) -> None:
    """Raw DeLong p-values must match the Round 2 manuscript within 1e-3.

    Round 3 strengthening (C3-1): we additionally recompute the p-value
    from the published predictions parquet using
    :func:`src.stats.metrics.delong_test` and require three-way agreement
    between (i) the manuscript number, (ii) the precomputed
    ``f10_delong_matrix.json`` cache, and (iii) a fresh call into the
    live DeLong implementation.  This protects against a regression in
    the implementation that silently re-writes the JSON cache without
    the manuscript noticing.
    """
    expected = {
        42: 0.039,
        43: 0.013,
        44: 0.068,
        45: 0.105,
        46: 0.005,
    }
    for seed, expected_p in expected.items():
        # (i) cache vs manuscript
        actual = delong_matrix["phage_component"][str(seed)]["esm650_xgb"]["simple_xgb"][
            "p_value"
        ]
        assert abs(actual - expected_p) < 1e-3, (
            f"seed={seed}: raw DeLong p drifted from {expected_p} to {actual:.6f}"
        )

        # (ii) live recomputation vs manuscript.  Skip gracefully if the
        # predictions parquet is not materialised in this checkout.
        a_path = PREDICTIONS / f"predictions_esm650_xgb_phage_component_seed{seed}.parquet"
        b_path = PREDICTIONS / f"predictions_simple_xgb_phage_component_seed{seed}.parquet"
        if not (a_path.exists() and b_path.exists()):
            continue
        a = pd.read_parquet(a_path)
        b = pd.read_parquet(b_path)
        merged = a.merge(b, on=["host_id", "phage_id", "label"], suffixes=("_a", "_b"))
        res = delong_test(
            merged["score_a"].to_numpy(),
            merged["score_b"].to_numpy(),
            merged["label"].to_numpy(),
        )
        assert abs(res.p_value - expected_p) < 5e-3, (
            f"seed={seed}: live DeLong p = {res.p_value:.4f} disagrees "
            f"with manuscript {expected_p} (cache says {actual:.4f})"
        )


def test_holm_correction_is_monotone_and_matches_manuscript(delong_matrix: dict) -> None:
    """Machine-checked Holm–Bonferroni on the five seeds.

    Pins the Holm-corrected p-values reported in the manuscript so they
    cannot silently drift if statsmodels or the underlying DeLong
    implementation changes.
    """
    ps = _ps_esm650_vs_simple(delong_matrix)
    _, p_holm, _, _ = multipletests(ps, method="holm")

    # Expected Holm-adjusted values (rounded to three decimals as reported).
    expected_holm = {
        42: 0.117,
        43: 0.051,
        44: 0.135,
        45: 0.135,  # <-- fixed to 0.135 per R2-1, not 0.105
        46: 0.025,
    }
    for seed, adj in zip(SEEDS, p_holm, strict=True):
        assert abs(adj - expected_holm[seed]) < 5e-3, (
            f"seed={seed}: Holm p drifted — expected {expected_holm[seed]}, got {adj:.4f}"
        )

    # Monotonicity: sort ps ascending, then Holm-adjusted must be non-decreasing
    # in that order.
    order = np.argsort(ps)
    sorted_adj = p_holm[order]
    diffs = np.diff(sorted_adj)
    assert np.all(diffs >= -1e-9), (
        f"Holm monotonicity violated: sorted adjusted p = {sorted_adj.tolist()}"
    )


def test_mean_delta_and_one_sample_ttest_backup_statistic(delong_matrix: dict) -> None:
    """Supporting statistic quoted in §4.5: t(4) = 2.85, p ≈ 0.046 for mean delta.

    One-sample two-sided t-test of the five per-seed DeLong deltas against 0.
    """
    from scipy import stats

    deltas = []
    for seed in SEEDS:
        deltas.append(
            delong_matrix["phage_component"][str(seed)]["esm650_xgb"]["simple_xgb"]["delta"]
        )
    arr = np.asarray(deltas, dtype=float)

    # Mean ± std as reported.
    assert abs(arr.mean() - 0.185) < 5e-3
    assert abs(arr.std(ddof=1) - 0.145) < 5e-3

    t, p = stats.ttest_1samp(arr, 0.0)
    assert abs(t - 2.85) < 0.05, f"t drifted: expected 2.85, got {t:.3f}"
    assert abs(p - 0.046) < 5e-3, f"t-test p drifted: expected 0.046, got {p:.4f}"


# --- ESM-2 650M sliding vs truncated (R2-2: DeLong comparison target) --- #


def _load_predictions(method: str, seed: int) -> pd.DataFrame:
    path = (
        PREDICTIONS
        / f"predictions_{method}_phage_component_seed{seed}.parquet"
    )
    if not path.exists():  # pragma: no cover - defensive
        pytest.skip(f"prediction parquet not found: {path}")
    return pd.read_parquet(path)


def test_esm650_sliding_vs_truncated_delong_nonsignificant() -> None:
    """R2-2: verify the §4.4 claim that sliding vs truncated is not significant.

    We compute paired DeLong across all 5 seeds from the stored parquets and
    assert that no seed reaches Holm-adjusted p < 0.05 for
    ``esm650_host_sliding_xgb`` vs ``esm650_host_truncated_xgb``. If this
    fails, the claim in §4.4 needs to be revisited.

    Round 3 strengthening (C3-2): per-seed raw p-values are pinned at
    1e-3 tolerance so any silent drift is caught — §4.4 of the manuscript
    reports each individual value, not just the min / Holm-corrected set.
    """
    expected_raw = {
        42: 0.0586,
        43: 0.0951,
        44: 0.8383,
        45: 0.7863,
        46: 0.2356,
    }
    ps = []
    for seed in SEEDS:
        a = _load_predictions("esm650_host_sliding_xgb", seed)
        b = _load_predictions("esm650_host_truncated_xgb", seed)
        merged = a.merge(b, on=["host_id", "phage_id", "label"], suffixes=("_a", "_b"))
        res = delong_test(
            merged["score_a"].to_numpy(),
            merged["score_b"].to_numpy(),
            merged["label"].to_numpy(),
        )
        ps.append(res.p_value)
        exp = expected_raw[seed]
        assert abs(res.p_value - exp) < 1e-3, (
            f"seed={seed}: raw DeLong p drifted from manuscript {exp:.4f} "
            f"to {res.p_value:.4f}"
        )
    ps_arr = np.asarray(ps)
    _, p_holm, _, _ = multipletests(ps_arr, method="holm")
    assert np.all(p_holm >= 0.05), (
        f"Unexpected significance; Holm-adjusted p = {p_holm.tolist()}"
    )
    # Also pin the min raw p reported in §4.4 Methods (approximately 0.06).
    assert ps_arr.min() >= 0.05, (
        f"Minimum raw p = {ps_arr.min():.4f} dipped below 0.05"
    )
