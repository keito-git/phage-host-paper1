# MIT License. See LICENSE in repository root.
"""SHAP analysis restricted to the E6 classical features (F9).

The Paper 1 methodology confines model interpretability to the *tabular*
AAC + dipeptide + ProtParam features produced by
:mod:`src.features.simple_features`, on an XGBoost classifier.  This keeps
the analysis lightweight and avoids scope creep into ESM-2 attention
visualisation (which Paper 2 will own).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ShapTopFeatures:
    """Top-K SHAP features summary."""

    top_features: pd.DataFrame  # [feature_name, mean_abs_shap]


# Canonical feature names from src.features.simple_features ---------------

AMINO_ACIDS: tuple[str, ...] = (
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
)


def feature_names() -> list[str]:
    """Return the 425 canonical feature names in column order."""
    aac = [f"AAC_{aa}" for aa in AMINO_ACIDS]
    dipep = [f"DI_{a}{b}" for a in AMINO_ACIDS for b in AMINO_ACIDS]
    physchem = [
        "mol_weight_1e4",
        "isoelectric_point",
        "aromaticity",
        "instability_index",
        "gravy",
    ]
    return aac + dipep + physchem


def compute_top_features(
    model: XGBClassifier,
    X: np.ndarray,
    top_k: int = 20,
) -> ShapTopFeatures:
    """Compute mean(|SHAP|) over rows, return top-K features.

    Returns
    -------
    :class:`ShapTopFeatures` with a DataFrame ``[feature_name, mean_abs_shap]``.
    """
    import shap

    names = feature_names()
    if X.shape[1] != len(names):
        raise ValueError(
            f"Expected {len(names)} features (AAC+dipeptide+protparam), got {X.shape[1]}"
        )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Binary classifier in xgboost ≥ 1.6 returns a (N, D) array directly.
    # Older builds may return a list of length 2; normalise.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature_name": names, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False).head(top_k).reset_index(drop=True)
    return ShapTopFeatures(top_features=df)
