# MIT License. See LICENSE in repository root.
"""Experiment alpha-2: label-shuffle negative control for esm650-xgb on phage-level split.

Companion to ``e_alpha_label_shuffle.py`` (simple features / RBP-cluster split,
3 seeds).  This script reruns the same negative-control logic for the headline
method (``esm650_xgb``) on the phage-level connected-component split with
5 seeds, so the paper can cite a directly-measured label-shuffle result for
the headline configuration.

Pipeline
--------
1. Load phage-level split (``phage_split_id{identity}_seed{seed}.parquet``).
2. Build phage- and host-side ESM-2 650M concat features identically to
   ``f_benchmark_main.run_esm_concat`` / ``e1_reproduce_phlearn``.
3. Shuffle the *train* labels (pair-level random permutation, seed-controlled).
4. Fit XGBoost with the same hyperparameters as the headline method.
5. Score on the (unshuffled) test set and record ROC-AUC, PR-AUC, best F1.
6. Aggregate per-seed numbers, write JSON + Markdown summary, and dump a
   ``predictions_*.parquet`` per seed for downstream inspection.
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

from src.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs
from src.models.classifiers import evaluate, make_xgboost
from src.utils.seed import set_global_seed

PREDICTIONS_DIR = PROCESSED_DIR / "predictions"


def _load_split(identity: float, seed: int) -> pd.DataFrame:
    path = PROCESSED_DIR / "splits" / f"phage_split_id{identity}_seed{seed}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Phage-level split missing: {path}. Run F0 first.")
    return pd.read_parquet(path)


def _load_esm_phlearn() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rbp_df = pd.read_csv(RAW_DIR / "esm2_embeddings_rbp.csv")
    feature_cols = [c for c in rbp_df.columns if c.isdigit()]
    phage_vec = {
        pid: sub[feature_cols].mean(axis=0).to_numpy(dtype=np.float32)
        for pid, sub in rbp_df.groupby("phage_ID")
    }
    loc_df = pd.read_csv(RAW_DIR / "esm2_embeddings_loci.csv")
    loc_cols = [c for c in loc_df.columns if c.isdigit()]
    host_vec = {
        row["accession"]: np.asarray([row[c] for c in loc_cols], dtype=np.float32)
        for _, row in loc_df.iterrows()
    }
    return phage_vec, host_vec


def _build_matrix(
    df: pd.DataFrame,
    phage_vec: dict[str, np.ndarray],
    host_vec: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    d_p = next(iter(phage_vec.values())).shape[0]
    d_h = next(iter(host_vec.values())).shape[0]
    rows_X: list[np.ndarray] = []
    rows_y: list[int] = []
    for row in df.itertuples(index=False):
        pv = phage_vec.get(row.phage_id)
        hv = host_vec.get(row.host_id)
        if pv is None or hv is None:
            rows_X.append(np.zeros(d_p + d_h, dtype=np.float32))
        else:
            rows_X.append(np.concatenate([pv, hv]))
        rows_y.append(int(row.label))
    return np.vstack(rows_X).astype(np.float32), np.asarray(rows_y, dtype=np.int64)


def _shuffle_labels(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = y.copy()
    rng.shuffle(shuffled)
    return shuffled


def run_once(
    df: pd.DataFrame,
    phage_vec: dict[str, np.ndarray],
    host_vec: dict[str, np.ndarray],
    seed: int,
) -> dict:
    set_global_seed(seed)

    train = df[df.split == "train"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)

    X_train, y_train_orig = _build_matrix(train, phage_vec, host_vec)
    X_test, y_test = _build_matrix(test, phage_vec, host_vec)

    y_train = _shuffle_labels(y_train_orig, seed=seed)

    scale = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))
    clf = make_xgboost(seed, scale_pos_weight=scale)
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]

    m = evaluate(y_test, scores)

    # persist predictions parquet for traceability
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = (
        PREDICTIONS_DIR
        / f"predictions_esm650_xgb_shuffled_phage_component_seed{seed}.parquet"
    )
    out = test[["host_id", "phage_id", "label"]].copy()
    out["method"] = "esm650_xgb_shuffled"
    out["split_kind"] = "phage_component"
    out["seed"] = seed
    out["score"] = scores.astype(np.float32)
    out.to_parquet(pred_path, index=False)

    return {
        "roc_auc": float(m.roc_auc),
        "pr_auc": float(m.pr_auc),
        "best_f1": float(m.best_f1),
        "test_prevalence": float(y_test.mean()),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "predictions_path": str(pred_path),
    }


def main(identity: float, seeds: list[int]) -> None:
    ensure_dirs()
    phage_vec, host_vec = _load_esm_phlearn()

    per_seed: dict[int, dict] = {}
    for s in seeds:
        df = _load_split(identity, s)
        per_seed[s] = run_once(df, phage_vec, host_vec, seed=s)

    aggregated: dict[str, float] = {}
    for metric in ("roc_auc", "pr_auc", "best_f1"):
        vals = [per_seed[s][metric] for s in seeds]
        aggregated[f"{metric}_mean"] = float(statistics.mean(vals))
        # use sample std (ddof=1) to match the convention used elsewhere
        # in the paper (e.g. SHAP unbiased std).  pstdev (population) is also
        # reported in alpha-1 for compatibility.
        aggregated[f"{metric}_std_sample"] = float(statistics.stdev(vals))
        aggregated[f"{metric}_std_pop"] = float(statistics.pstdev(vals))

    summary = {
        "identity": identity,
        "seeds": list(seeds),
        "method": "esm650_xgb",
        "split_kind": "phage_component",
        "label_shuffle_scheme": "pair_level_permutation_in_train",
        "per_seed": per_seed,
        "aggregated": aggregated,
    }

    out_json = REPORTS_DIR / "experiment_alpha2_label_shuffle_esm650.json"
    out_md = REPORTS_DIR / "experiment_alpha2_label_shuffle_esm650.md"
    out_json.write_text(json.dumps(summary, indent=2))
    render_report(out_md, summary)
    print(json.dumps(summary, indent=2))


def render_report(path: Path, summary: dict) -> None:
    lines = [
        "# Experiment alpha-2 -- Label-shuffle negative control (esm650-xgb / phage-level)",
        "",
        "**Goal.** Verify that the headline method (`esm650_xgb`) collapses to",
        "chance-level performance on the phage-level connected-component split",
        "when train labels are randomly permuted within the train partition.",
        "A correctly-set-up experiment must yield ROC-AUC ~= 0.5 here.",
        "",
        f"- Identity: `{summary['identity']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Method: `{summary['method']}`",
        f"- Split kind: `{summary['split_kind']}`",
        f"- Shuffle scheme: `{summary['label_shuffle_scheme']}`",
        "",
        "## Per-seed metrics",
        "",
        "| Seed | n_train | n_test | test_prevalence | ROC-AUC | PR-AUC | best F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary["seeds"]:
        r = summary["per_seed"][s]
        lines.append(
            f"| {s} | {r['n_train']} | {r['n_test']} | {r['test_prevalence']:.4f} | "
            f"{r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['best_f1']:.4f} |"
        )
    agg = summary["aggregated"]
    lines += [
        "",
        "## Aggregated (5 seeds)",
        "",
        "| Metric | mean | std (sample, ddof=1) | std (population) |",
        "|---|---:|---:|---:|",
        f"| ROC-AUC | {agg['roc_auc_mean']:.4f} | {agg['roc_auc_std_sample']:.4f} | {agg['roc_auc_std_pop']:.4f} |",
        f"| PR-AUC  | {agg['pr_auc_mean']:.4f} | {agg['pr_auc_std_sample']:.4f} | {agg['pr_auc_std_pop']:.4f} |",
        f"| best F1 | {agg['best_f1_mean']:.4f} | {agg['best_f1_std_sample']:.4f} | {agg['best_f1_std_pop']:.4f} |",
        "",
        "## Interpretation",
        "",
        "ROC-AUC near 0.5 confirms that the headline pipeline has no hidden",
        "leakage on the phage-level split: when phage-side features are",
        "decoupled from labels via random permutation, the classifier finds",
        "no exploitable structure.  Any value substantially above 0.5 would",
        "indicate residual leakage (e.g. host_id acting as a covert lookup).",
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46]
    )
    args = parser.parse_args()
    main(identity=args.identity, seeds=list(args.seeds))
