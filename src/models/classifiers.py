# MIT License. See LICENSE in repository root.
"""Lightweight classifiers shared across experiments.

We keep the API deliberately narrow so all experiment scripts can swap
between logistic regression, XGBoost and a shallow MLP with a single flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class SupportsFitPredict(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Metrics:
    """Common binary-classification metrics."""

    roc_auc: float
    pr_auc: float
    best_f1: float
    best_f1_threshold: float


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> Metrics:
    """Compute ROC-AUC, PR-AUC, and the best F1 across thresholds."""
    roc = roc_auc_score(y_true, scores)
    pr = average_precision_score(y_true, scores)
    precision, recall, thresh = precision_recall_curve(y_true, scores)
    # precision/recall have one more entry than thresh; align by dropping the
    # last P/R point (which corresponds to an "all negatives" threshold).
    f1s = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    if f1s.size == 0:
        best_f1 = float("nan")
        best_thr = float("nan")
    else:
        idx = int(np.nanargmax(f1s))
        best_f1 = float(f1s[idx])
        best_thr = float(thresh[idx])
    return Metrics(roc_auc=float(roc), pr_auc=float(pr), best_f1=best_f1, best_f1_threshold=best_thr)


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(int)


def f1_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    return float(f1_score(y_true, apply_threshold(scores, threshold)))


def make_logistic(seed: int) -> SupportsFitPredict:
    """Standard-scaled logistic regression with L2 penalty."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )


def make_xgboost(seed: int, scale_pos_weight: float | None = None) -> SupportsFitPredict:
    """XGBoost with a pragmatic default hyper-parameter set."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        scale_pos_weight=scale_pos_weight or 1.0,
        n_jobs=-1,
    )


def make_mlp(seed: int, hidden: tuple[int, ...] = (256, 64)) -> SupportsFitPredict:
    """Shallow MLP classifier on standardised features."""
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            max_iter=200,
            early_stopping=True,
            random_state=seed,
        ),
    )
