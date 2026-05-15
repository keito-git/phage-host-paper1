# MIT License. See LICENSE in repository root.
"""Tests for the K-type stratification helper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.ktype import stratified_metrics


def test_stratified_metrics_returns_expected_columns() -> None:
    rng = np.random.default_rng(0)
    pairs = pd.DataFrame(
        {
            "host_id": np.repeat(["h1", "h2", "h3"], 40),
            "phage_id": [f"p{i}" for i in range(120)],
            "label": rng.integers(0, 2, 120),
            "score": rng.random(120),
        }
    )
    ktype = pd.DataFrame(
        {
            "host_id": ["h1", "h2", "h3"],
            "k_type": ["K1", "K1", "K2"],
            "k_type_source": ["cluster_surrogate"] * 3,
        }
    )
    out = stratified_metrics(pairs, ktype)
    assert set(out.columns) >= {"k_type", "n_pairs", "n_positives", "metric"}
