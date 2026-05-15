# MIT License. See LICENSE in repository root.
"""Experiment 1: reproduce a PhageHostLearn-style baseline using the ESM-2
embeddings that ship with the Zenodo 11061100 artefact.

The upstream paper feeds concatenated ``[ESM2(RBP), ESM2(K-locus)]`` vectors
to an XGBoost classifier.  We repeat that setup *under our leak-free
cluster-aware split* from E5 so the reported metric is directly comparable
with E2/E3/E6.

Note: this is intentionally not a verbatim re-run of the upstream tuning;
the goal is to land inside a plausible operating range and anchor the
comparison, not to chase the published 0.818 ROC-AUC (which was reported on
a different, random split).
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

from src.config import MULTI_SEEDS, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs
from src.models.classifiers import evaluate, make_logistic, make_xgboost
from src.utils.seed import set_global_seed


def _load_split(identity: float, seed: int) -> pd.DataFrame:
    path = PROCESSED_DIR / "splits" / f"split_id{identity}_seed{seed}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Split missing: {path}. Run E5 first.")
    return pd.read_parquet(path)


def _load_embeddings(raw_dir: Path = RAW_DIR) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return ``(phage_id -> vec, host_id -> vec)`` dictionaries.

    The upstream CSVs store per-protein embeddings: the RBP file has one
    row per RBP (``phage_ID``, ``protein_ID``, 1280-d vector), while the
    loci file has one row per K-locus protein (``accession``, 1280-d).
    For each phage we take the mean of the rows sharing ``phage_ID``;
    for each host we recover ``host_id`` from the accession prefix
    convention ``<host_id>_<protein_idx>``.
    """
    rbp_df = pd.read_csv(raw_dir / "esm2_embeddings_rbp.csv")
    feature_cols = [c for c in rbp_df.columns if c.isdigit()]
    phage_vec = {
        pid: sub[feature_cols].mean(axis=0).to_numpy(dtype=np.float32)
        for pid, sub in rbp_df.groupby("phage_ID")
    }

    loc_df = pd.read_csv(raw_dir / "esm2_embeddings_loci.csv")
    loc_feature_cols = [c for c in loc_df.columns if c.isdigit()]
    # The ``accession`` column in the upstream CSV coincides 1:1 with the
    # ``host_id`` used by ``phage_host_interactions.csv`` (we verified this
    # by set-comparison).  No re-mapping is necessary.
    host_vec = {
        row["accession"]: np.asarray(
            [row[c] for c in loc_feature_cols], dtype=np.float32
        )
        for _, row in loc_df.iterrows()
    }
    return phage_vec, host_vec


def _assemble_matrix(
    df: pd.DataFrame,
    phage_vec: dict[str, np.ndarray],
    host_vec: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build concat(RBP||K-locus) features; drop rows missing an embedding."""
    rows: list[np.ndarray] = []
    labels: list[int] = []
    dropped = 0
    for row in df.itertuples(index=False):
        pv = phage_vec.get(row.phage_id)
        hv = host_vec.get(row.host_id)
        if pv is None or hv is None:
            dropped += 1
            continue
        rows.append(np.concatenate([pv, hv]))
        labels.append(int(row.label))
    if not rows:
        raise RuntimeError("No rows with available embeddings.")
    return (
        np.vstack(rows).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        dropped,
    )


def run_once(
    df: pd.DataFrame,
    phage_vec: dict[str, np.ndarray],
    host_vec: dict[str, np.ndarray],
    seed: int,
) -> dict:
    set_global_seed(seed)

    train = df[df.split == "train"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)

    X_train, y_train, dropped_train = _assemble_matrix(train, phage_vec, host_vec)
    X_test, y_test, dropped_test = _assemble_matrix(test, phage_vec, host_vec)

    scale_pos = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))

    results: dict[str, dict] = {
        "dropped_train": dropped_train,
        "dropped_test": dropped_test,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    for name, clf in [
        ("logreg", make_logistic(seed)),
        ("xgboost", make_xgboost(seed, scale_pos_weight=scale_pos)),
    ]:
        clf.fit(X_train, y_train)
        scores = clf.predict_proba(X_test)[:, 1]
        m = evaluate(y_test, scores)
        results[name] = {
            "roc_auc": m.roc_auc,
            "pr_auc": m.pr_auc,
            "best_f1": m.best_f1,
        }
    return results


def main(identity: float, seeds: list[int]) -> None:
    ensure_dirs()
    phage_vec, host_vec = _load_embeddings()

    per_seed: dict[int, dict] = {}
    for s in seeds:
        df = _load_split(identity, s)
        per_seed[s] = run_once(df, phage_vec, host_vec, seed=s)

    aggregated: dict[str, dict[str, float]] = {}
    for model in ("logreg", "xgboost"):
        for metric in ("roc_auc", "pr_auc", "best_f1"):
            vals = [per_seed[s][model][metric] for s in seeds]
            aggregated.setdefault(model, {})[f"{metric}_mean"] = float(statistics.mean(vals))
            aggregated[model][f"{metric}_std"] = float(statistics.pstdev(vals))

    summary = {
        "identity": identity,
        "seeds": list(seeds),
        "phage_vec_dim": int(next(iter(phage_vec.values())).shape[0]),
        "host_vec_dim": int(next(iter(host_vec.values())).shape[0]),
        "n_phages_in_vec": len(phage_vec),
        "n_hosts_in_vec": len(host_vec),
        "per_seed": per_seed,
        "aggregated": aggregated,
    }
    render_report(REPORTS_DIR / "experiment_01_reproduce_phlearn.md", summary)
    print(json.dumps(summary, indent=2))


def render_report(path: Path, summary: dict) -> None:
    lines = [
        "# Experiment 1 -- PhageHostLearn embeddings + XGBoost under the E5 split",
        "",
        "**Goal.** Use the ESM-2 embeddings released with Zenodo 11061100 as",
        "features to classify phage-host pairs, but evaluate under the leak-",
        "free cluster-aware split from E5 so the result is comparable to E3",
        "and E6.  This is our strongest \"learned representation\" anchor.",
        "",
        f"- Identity: `{summary['identity']}`, Seeds: `{summary['seeds']}`",
        f"- RBP embedding dimensionality: {summary['phage_vec_dim']}",
        f"- K-locus embedding dimensionality: {summary['host_vec_dim']}",
        f"- Phages covered: {summary['n_phages_in_vec']}, "
        f"Hosts covered: {summary['n_hosts_in_vec']}",
        "",
        "## Aggregated metrics",
        "",
        "| Model | ROC-AUC | PR-AUC | best F1 |",
        "|---|---|---|---|",
    ]
    for model, vals in summary["aggregated"].items():
        lines.append(
            f"| {model} | {vals['roc_auc_mean']:.3f} +/- {vals['roc_auc_std']:.3f} | "
            f"{vals['pr_auc_mean']:.3f} +/- {vals['pr_auc_std']:.3f} | "
            f"{vals['best_f1_mean']:.3f} +/- {vals['best_f1_std']:.3f} |"
        )
    lines += [
        "",
        "## Per-seed",
        "",
        "| Seed | n_train | n_test | logreg ROC | xgb ROC | xgb PR-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for seed, run in summary["per_seed"].items():
        lines.append(
            f"| {seed} | {run['n_train']} | {run['n_test']} | "
            f"{run['logreg']['roc_auc']:.3f} | {run['xgboost']['roc_auc']:.3f} | "
            f"{run['xgboost']['pr_auc']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This is the reference anchor that every subsequent learned-",
        "representation experiment must beat.  Because the E5 split breaks",
        "RBP sequence redundancy, we expect this metric to be *below* the",
        "upstream paper's reported 0.818 ROC-AUC, and that gap itself is an",
        "honest quantification of how much of the reported accuracy came",
        "from train/test RBP leakage.",
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(MULTI_SEEDS))
    args = parser.parse_args()
    main(identity=args.identity, seeds=list(args.seeds))
