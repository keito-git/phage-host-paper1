# MIT License. See LICENSE in repository root.
"""F1-B (classifier side) — Train logreg / XGB on ESM-2 650M host embeddings.

Prerequisites
-------------
``f1b_esm650_sliding_gpu.py`` has been run on the university GPU server and
its two parquet outputs copied to
``data/processed/f1_cache/host_esm650_{truncated,sliding}.parquet``.

What this script does
---------------------
1. Load the two ESM-2 650M host-embedding parquets (one per mode).
2. Load the phage-side ESM-2 650M embedding that was already shipped in
   ``data/raw/esm2_embeddings_rbp.csv`` (same vector used in all existing
   ``esm650_*`` runs in F10 summary).
3. For each ``mode in {truncated, sliding}`` × ``seed in {42..46}`` on the
   ``phage_component`` split, train logreg and XGBoost and record metrics.
4. Write predictions parquet into ``data/processed/predictions/`` with
   **new** names (``predictions_esm650_host_{mode}_{model}_phage_component_seed{seed}.parquet``)
   and a dated report ``reports/f1_sliding_window_esm650_2026-XX-XX.md``.

Existing files are never overwritten — all outputs go to new paths.
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import json
import warnings
from datetime import date

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs
from src.models.classifiers import make_logistic, make_xgboost
from src.stats.metrics import (
    expected_calibration_error,
    pr_auc,
    roc_auc,
    stratified_bootstrap_ci,
)
from src.utils.seed import set_global_seed

warnings.filterwarnings("ignore", category=UserWarning)

PREDICTIONS_DIR = PROCESSED_DIR / "predictions"
CACHE_DIR = PROCESSED_DIR / "f1_cache"


def _load_phage_vectors() -> dict[str, np.ndarray]:
    """Mean-pool the upstream Zenodo ESM-2 650M RBP embeddings per phage."""
    df = pd.read_csv(RAW_DIR / "esm2_embeddings_rbp.csv")
    feat = [c for c in df.columns if c.isdigit()]
    return {
        pid: sub[feat].mean(axis=0).to_numpy(dtype=np.float32)
        for pid, sub in df.groupby("phage_ID")
    }


def _load_host_vectors(mode: str) -> dict[str, np.ndarray]:
    """Load per-host ESM-2 650M embedding parquet produced on the GPU server."""
    path = CACHE_DIR / f"host_esm650_{mode}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path} — run f1b_esm650_sliding_gpu.py on GPU first")
    df = pd.read_parquet(path)
    out: dict[str, np.ndarray] = {}
    for row in df.itertuples(index=False):
        out[row.host_id] = np.asarray(row.embedding, dtype=np.float32)
    return out


def _run_one_seed(
    mode: str,
    split_df: pd.DataFrame,
    phage_vec: dict[str, np.ndarray],
    host_vec: dict[str, np.ndarray],
    seed: int,
) -> list[dict]:
    set_global_seed(seed)
    d_p = next(iter(phage_vec.values())).shape[0]
    d_h = next(iter(host_vec.values())).shape[0]

    def build(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X: list[np.ndarray] = []
        y: list[int] = []
        for row in df.itertuples(index=False):
            pv = phage_vec.get(row.phage_id, np.zeros(d_p, dtype=np.float32))
            hv = host_vec.get(row.host_id, np.zeros(d_h, dtype=np.float32))
            X.append(np.concatenate([pv, hv]))
            y.append(int(row.label))
        return np.vstack(X).astype(np.float32), np.asarray(y, dtype=np.int64)

    tr = split_df[split_df.split == "train"]
    te = split_df[split_df.split == "test"].reset_index(drop=True)
    if te.empty or (te.label == 1).sum() < 2 or (te.label == 0).sum() < 2:
        return []

    X_tr, y_tr = build(tr)
    X_te, y_te = build(te)
    scale = max(1.0, (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))

    rows: list[dict] = []
    for name, clf in [
        ("logreg", make_logistic(seed)),
        ("xgb", make_xgboost(seed, scale_pos_weight=scale)),
    ]:
        clf.fit(X_tr, y_tr)
        scores = clf.predict_proba(X_te)[:, 1]
        roc_ci = stratified_bootstrap_ci(y_te, scores, roc_auc, n_resamples=500, seed=seed)
        pr_ci = stratified_bootstrap_ci(y_te, scores, pr_auc, n_resamples=500, seed=seed)
        ece = expected_calibration_error(y_te, scores)
        method = f"esm650_host_{mode}_{name}"
        rows.append(
            {
                "method": method,
                "mode": mode,
                "model": name,
                "seed": seed,
                "n_train": int(len(y_tr)),
                "n_test": int(len(y_te)),
                "roc_auc": roc_ci.point,
                "roc_auc_lo": roc_ci.lower,
                "roc_auc_hi": roc_ci.upper,
                "pr_auc": pr_ci.point,
                "pr_auc_lo": pr_ci.lower,
                "pr_auc_hi": pr_ci.upper,
                "ece": ece,
            }
        )

        # Save predictions (new filename, no overwrite of existing).
        out_pred = te[["host_id", "phage_id", "label"]].copy()
        out_pred["method"] = method
        out_pred["split_kind"] = "phage_component"
        out_pred["seed"] = seed
        out_pred["score"] = scores.astype(np.float32)
        out_pred.to_parquet(
            PREDICTIONS_DIR
            / f"predictions_esm650_host_{mode}_{name}_phage_component_seed{seed}.parquet",
            index=False,
        )
    return rows


def main(identity: float, seeds: list[int]) -> dict:
    ensure_dirs()
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    phage_vec = _load_phage_vectors()

    all_rows: list[dict] = []
    for mode in ("truncated", "sliding"):
        try:
            host_vec = _load_host_vectors(mode)
        except FileNotFoundError as e:
            print(f"[F1B-cls] SKIP mode={mode}: {e}", flush=True)
            continue
        for seed in seeds:
            split_path = PROCESSED_DIR / "splits" / f"phage_split_id{identity}_seed{seed}.parquet"
            if not split_path.exists():
                continue
            split_df = pd.read_parquet(split_path)
            rows = _run_one_seed(mode, split_df, phage_vec, host_vec, seed)
            all_rows.extend(rows)
            for r in rows:
                print(
                    f"[F1B-cls] {r['method']:>40s} seed={seed} "
                    f"ROC={r['roc_auc']:.3f} PR={r['pr_auc']:.3f}",
                    flush=True,
                )

    df = pd.DataFrame(all_rows)
    today = date.today().isoformat()
    out_parquet = REPORTS_DIR / f"f1_sliding_window_esm650_{today}.parquet"
    df.to_parquet(out_parquet, index=False)

    # Aggregate & write dated markdown report
    md_lines = [
        f"# F1-B — Sliding-window ESM-2 650M host pooling ({today})",
        "",
        "**Design:** the F1 sliding-window experiment (originally run with ",
        "ESM-2 8M on MacBook) is re-run with ESM-2 650M on the university ",
        "GPU server (GPU 2 only, H100 NVL, fp16).  Host-side K-locus ",
        "concatenations are embedded under two modes:",
        "",
        "- `truncated`: first 1022 aa only, matching existing upstream Zenodo behaviour.",
        "- `sliding`: window 1022, stride 511, mean pool across windows.",
        "",
        "The phage-side vector is unchanged (upstream ESM-2 650M RBP mean).  ",
        "All evaluation uses the F0 `phage_component` split (5 seeds, 42-46).",
        "",
        "## Aggregated across seeds",
        "",
        "| method | n_seeds | ROC-AUC (mean ± std) | PR-AUC (mean ± std) | ECE (mean) |",
        "|---|---|---|---|---|",
    ]
    if not df.empty:
        agg = df.groupby(["method"]).agg(
            n=("seed", "count"),
            roc_m=("roc_auc", "mean"),
            roc_s=("roc_auc", "std"),
            pr_m=("pr_auc", "mean"),
            pr_s=("pr_auc", "std"),
            ece_m=("ece", "mean"),
        ).reset_index().sort_values("method")
        for _, r in agg.iterrows():
            md_lines.append(
                f"| {r.method} | {int(r.n)} | "
                f"{r.roc_m:.3f} ± {r.roc_s:.3f} | "
                f"{r.pr_m:.3f} ± {r.pr_s:.3f} | {r.ece_m:.3f} |"
            )
    md_lines.append("")
    md_lines.append("## Per-seed detail")
    md_lines.append("")
    md_lines.append("| method | seed | n_test | ROC-AUC | PR-AUC | ECE |")
    md_lines.append("|---|---|---|---|---|---|")
    for r in all_rows:
        md_lines.append(
            f"| {r['method']} | {r['seed']} | {r['n_test']} | "
            f"{r['roc_auc']:.3f} [{r['roc_auc_lo']:.3f}, {r['roc_auc_hi']:.3f}] | "
            f"{r['pr_auc']:.3f} [{r['pr_auc_lo']:.3f}, {r['pr_auc_hi']:.3f}] | {r['ece']:.3f} |"
        )
    md_lines.append("")
    md_lines.append("## Delta analysis (sliding − truncated)")
    md_lines.append("")
    if not df.empty:
        pivot = df.pivot_table(
            index=["model", "seed"], columns="mode", values="roc_auc"
        ).dropna()
        if {"truncated", "sliding"}.issubset(pivot.columns):
            pivot["delta_roc"] = pivot["sliding"] - pivot["truncated"]
            md_lines.append("| model | mean Δ ROC-AUC (sliding − truncated) | seeds |")
            md_lines.append("|---|---|---|")
            for model, sub in pivot.groupby(level=0):
                md_lines.append(
                    f"| {model} | {sub['delta_roc'].mean():+.3f} (std {sub['delta_roc'].std():.3f}) | "
                    f"{len(sub)} |"
                )
            md_lines.append("")
            md_lines.append("Per-seed:")
            md_lines.append("")
            md_lines.append("| model | seed | truncated ROC | sliding ROC | Δ |")
            md_lines.append("|---|---|---|---|---|")
            for (model, seed), row in pivot.iterrows():
                md_lines.append(
                    f"| {model} | {seed} | {row['truncated']:.3f} | "
                    f"{row['sliding']:.3f} | {row['delta_roc']:+.3f} |"
                )

    report_path = REPORTS_DIR / f"f1_sliding_window_esm650_{today}.md"
    report_path.write_text("\n".join(md_lines))
    return {"n_rows": len(df), "report": str(report_path), "parquet": str(out_parquet)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    print(json.dumps(main(args.identity, list(args.seeds)), indent=2))
