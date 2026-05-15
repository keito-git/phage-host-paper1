# MIT License. See LICENSE in repository root.
"""Experiment 6: sanity-check with classical sequence features.

Uses the leak-free split from E5 and predicts interaction with logistic
regression + XGBoost on the concatenation of RBP and K-locus features.
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MULTI_SEEDS, PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from src.data.phlearn import flatten_loci, load_loci
from src.features.simple_features import summarise_frame
from src.models.classifiers import evaluate, make_logistic, make_xgboost
from src.utils.seed import set_global_seed


def _load_split(identity: float, seed: int) -> pd.DataFrame:
    path = PROCESSED_DIR / "splits" / f"split_id{identity}_seed{seed}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file missing: {path}. Run e5_cluster_split.py first."
        )
    return pd.read_parquet(path)


def build_feature_matrix(
    df: pd.DataFrame, loci_flat: pd.DataFrame, cache_key: str
) -> np.ndarray:
    """Return a concat(rbp_feat || klocus_feat) matrix."""
    cache_path = PROCESSED_DIR / "simple_features" / f"{cache_key}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        loaded = np.load(cache_path)
        return loaded["X"]

    rbp_feat = summarise_frame(df["rbp_sequence"])
    merged = df.merge(loci_flat, on="host_id", how="left")
    if merged["k_locus_concat"].isna().any():
        missing_hosts = merged[merged["k_locus_concat"].isna()]["host_id"].unique()
        raise RuntimeError(
            f"Hosts without K-locus sequences: {len(missing_hosts)} examples: "
            f"{missing_hosts[:5]}"
        )
    host_feat = summarise_frame(merged["k_locus_concat"])
    X = np.hstack([rbp_feat, host_feat]).astype(np.float32)
    np.savez_compressed(cache_path, X=X)
    return X


def run_once(
    df: pd.DataFrame, loci_flat: pd.DataFrame, seed: int, identity: float
) -> dict:
    set_global_seed(seed)

    train_df = df[df.split == "train"].reset_index(drop=True)
    test_df = df[df.split == "test"].reset_index(drop=True)

    X_train = build_feature_matrix(train_df, loci_flat, f"train_id{identity}_seed{seed}")
    X_test = build_feature_matrix(test_df, loci_flat, f"test_id{identity}_seed{seed}")
    y_train = train_df.label.to_numpy()
    y_test = test_df.label.to_numpy()

    scale_pos = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))

    results: dict[str, dict] = {}
    for name, clf in [
        ("logreg", make_logistic(seed)),
        ("xgboost", make_xgboost(seed, scale_pos_weight=scale_pos)),
    ]:
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        m = evaluate(y_test, probs)
        results[name] = {
            "roc_auc": m.roc_auc,
            "pr_auc": m.pr_auc,
            "best_f1": m.best_f1,
            "best_f1_threshold": m.best_f1_threshold,
        }
    return results


def main(identity: float, seeds: list[int]) -> None:
    ensure_dirs()
    loci_flat = flatten_loci(load_loci())
    per_seed: dict[int, dict] = {}
    for s in seeds:
        df = _load_split(identity, s) if (PROCESSED_DIR / "splits" / f"split_id{identity}_seed{s}.parquet").exists() else _load_split(identity, seeds[0])
        per_seed[s] = run_once(df, loci_flat, seed=s, identity=identity)

    aggregated: dict[str, dict[str, float]] = {}
    for model in per_seed[seeds[0]]:
        for metric in per_seed[seeds[0]][model]:
            values = [per_seed[s][model][metric] for s in seeds]
            aggregated.setdefault(model, {})[metric + "_mean"] = float(statistics.mean(values))
            aggregated[model][metric + "_std"] = float(statistics.pstdev(values))

    summary = {
        "identity": identity,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "aggregated": aggregated,
    }

    report_path = REPORTS_DIR / "experiment_06_simple_features.md"
    render_report(report_path, summary)
    print(json.dumps(summary, indent=2))


def render_report(path: Path, summary: dict) -> None:
    path.write_text(
        "# Experiment 6 — Classical sequence-feature sanity baseline\n\n"
        "**Goal.** Confirm that trivial physicochemical and k-mer features\n"
        "recover non-random performance on the leak-free split.  This is the\n"
        "*lower bound* every learned-representation model must beat.\n\n"
        f"- Identity threshold: `{summary['identity']}`\n"
        f"- Seeds: `{summary['seeds']}`\n\n"
        "## Results (mean +/- std over seeds)\n\n"
        "| Model | ROC-AUC | PR-AUC | best F1 |\n"
        "|---|---|---|---|\n"
        + "".join(
            f"| {model} | "
            f"{vals['roc_auc_mean']:.3f} +/- {vals['roc_auc_std']:.3f} | "
            f"{vals['pr_auc_mean']:.3f} +/- {vals['pr_auc_std']:.3f} | "
            f"{vals['best_f1_mean']:.3f} +/- {vals['best_f1_std']:.3f} |\n"
            for model, vals in summary["aggregated"].items()
        )
        + "\n## Interpretation\n\n"
        "PR-AUC is the more informative metric because the dataset is highly\n"
        "imbalanced (~3.6% positives).  If ROC-AUC is well above 0.5 but\n"
        "PR-AUC is close to the prevalence, the model is barely better than\n"
        "random in the regime that actually matters for downstream use.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(MULTI_SEEDS))
    args = parser.parse_args()
    main(identity=args.identity, seeds=list(args.seeds))
