# MIT License. See LICENSE in repository root.
"""Aggregation of per-method × per-split results into the Paper 1 main table (F10)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_all_predictions(pred_dir: Path) -> pd.DataFrame:
    """Concatenate every ``predictions_*.parquet`` in ``pred_dir``.

    Each prediction file is expected to have columns
    ``[method, split_kind, seed, host_id, phage_id, label, score]``.
    """
    parts: list[pd.DataFrame] = []
    for p in sorted(Path(pred_dir).glob("predictions_*.parquet")):
        parts.append(pd.read_parquet(p))
    if not parts:
        return pd.DataFrame(
            columns=[
                "method", "split_kind", "seed", "host_id", "phage_id", "label", "score",
            ]
        )
    return pd.concat(parts, ignore_index=True)


def method_split_table(
    preds: pd.DataFrame,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Build the method × split 2D table (ROC-AUC, PR-AUC, ECE with 95% CI).

    Assumes ``preds`` is the concatenation from :func:`load_all_predictions`.
    """
    from src.stats.metrics import (
        expected_calibration_error,
        pr_auc,
        roc_auc,
        stratified_bootstrap_ci,
    )

    if preds.empty:
        return pd.DataFrame(
            columns=[
                "method", "split_kind", "seed",
                "roc_auc", "roc_auc_lo", "roc_auc_hi",
                "pr_auc", "pr_auc_lo", "pr_auc_hi",
                "ece",
            ]
        )

    rows: list[dict[str, float | str | int]] = []
    groups = preds.groupby(["method", "split_kind", "seed"])
    for (method, split_kind, seed), sub in groups:
        y = sub["label"].to_numpy().astype(int)
        s = sub["score"].to_numpy().astype(float)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        roc_ci = stratified_bootstrap_ci(y, s, roc_auc, n_resamples=n_bootstrap, seed=int(seed))
        pr_ci = stratified_bootstrap_ci(y, s, pr_auc, n_resamples=n_bootstrap, seed=int(seed))
        # ECE expects scores in [0, 1]; min-max rescale if the method emits
        # raw logits (e.g. MMseqs bits).  We use a cheap heuristic: if any
        # value is outside [0, 1], min-max rescale.
        s_cal = s
        if (s < 0).any() or (s > 1).any():
            s_lo, s_hi = float(np.min(s)), float(np.max(s))
            s_cal = (s - s_lo) / (s_hi - s_lo) if s_hi > s_lo else np.full_like(s, 0.5)
        ece = expected_calibration_error(y, s_cal)
        rows.append({
            "method": method,
            "split_kind": split_kind,
            "seed": int(seed),
            "n_pairs": int(len(sub)),
            "n_positives": int(y.sum()),
            "roc_auc": roc_ci.point,
            "roc_auc_lo": roc_ci.lower,
            "roc_auc_hi": roc_ci.upper,
            "pr_auc": pr_ci.point,
            "pr_auc_lo": pr_ci.lower,
            "pr_auc_hi": pr_ci.upper,
            "ece": ece,
        })
    return pd.DataFrame(rows)


def summarise_across_seeds(per_run: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a :func:`method_split_table` frame across seeds."""
    if per_run.empty:
        return per_run
    agg = per_run.groupby(["method", "split_kind"]).agg(
        n_seeds=("seed", "nunique"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        ece_mean=("ece", "mean"),
        ece_std=("ece", "std"),
    ).reset_index()
    return agg
