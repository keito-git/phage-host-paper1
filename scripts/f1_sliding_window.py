# MIT License. See LICENSE in repository root.
"""F1 — Sliding-window ESM-2 pooling over K-locus concatenations.

Implementation plan
-------------------
1. Use ``ESMEmbedder(esm2_t6_8M_UR50D)`` — the smallest ESM-2 that still
   runs on MacBook Pro M4 Max within minutes.  The 650M variant is
   earmarked for the GPU server once it is online; the sliding-window
   gain itself should already be visible at 8M.
2. Embed all K-locus concatenations both as truncated (baseline) and with
   :func:`sliding_window_embed` (mean-pool at stride = 511).
3. Concatenate each host vector with the phage's upstream ESM-2 650M mean
   (from ``esm2_embeddings_rbp.csv``).  This keeps the phage side
   comparable between truncated and sliding-window runs so the delta is
   purely the host-side length recovery.
4. Train / evaluate on the F0 phage-level split, 5 seeds, XGBoost and
   logreg.
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from src.config import MULTI_SEEDS, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs
from src.data.phlearn import flatten_loci, load_all
from src.features.esm_embedding import ESMEmbedder
from src.features.sliding_window import WindowConfig, sliding_window_embed
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


def _load_phage_embeddings() -> dict[str, np.ndarray]:
    """Return mean-pooled upstream ESM-2 650M embedding per phage."""
    df = pd.read_csv(RAW_DIR / "esm2_embeddings_rbp.csv")
    feat_cols = [c for c in df.columns if c.isdigit()]
    return {
        pid: sub[feat_cols].mean(axis=0).to_numpy(dtype=np.float32)
        for pid, sub in df.groupby("phage_ID")
    }


def embed_hosts(
    loci_concat: dict[str, str],
    sliding: bool,
    model_name: str,
    max_length: int = 1022,
) -> dict[str, np.ndarray]:
    """Return host_id -> (D,) embedding.  ``sliding=True`` uses windows."""
    embedder = ESMEmbedder(model_name=model_name, max_length=max_length, batch_size=2)
    if not sliding:
        truncated = {k: v[:max_length] for k, v in loci_concat.items()}
        return embedder.embed_many(truncated)
    cfg = WindowConfig(window_size=max_length, stride=max_length // 2, pooling="mean")
    return sliding_window_embed(embedder.embed_many, loci_concat, cfg)


def _run_one_mode(
    mode: str,
    split_df: pd.DataFrame,
    phage_vec: dict[str, np.ndarray],
    host_vec: dict[str, np.ndarray],
    seed: int,
) -> list[dict]:
    set_global_seed(seed)

    def build(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        rows: list[np.ndarray] = []
        labels: list[int] = []
        d_p = next(iter(phage_vec.values())).shape[0]
        d_h = next(iter(host_vec.values())).shape[0]
        for row in df.itertuples(index=False):
            pv = phage_vec.get(row.phage_id)
            hv = host_vec.get(row.host_id)
            if pv is None:
                pv = np.zeros(d_p, dtype=np.float32)
            if hv is None:
                hv = np.zeros(d_h, dtype=np.float32)
            rows.append(np.concatenate([pv, hv]))
            labels.append(int(row.label))
        return np.vstack(rows).astype(np.float32), np.asarray(labels, dtype=np.int64)

    train_df = split_df[split_df.split == "train"]
    test_df = split_df[split_df.split == "test"].reset_index(drop=True)
    X_tr, y_tr = build(train_df)
    X_te, y_te = build(test_df)
    scale = max(1.0, (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))

    out_rows: list[dict] = []
    for name, clf in [
        ("logreg", make_logistic(seed)),
        ("xgb", make_xgboost(seed, scale_pos_weight=scale)),
    ]:
        clf.fit(X_tr, y_tr)
        scores = clf.predict_proba(X_te)[:, 1]
        if y_te.sum() < 2 or (y_te == 0).sum() < 2:
            continue
        roc_ci = stratified_bootstrap_ci(y_te, scores, roc_auc, n_resamples=500, seed=seed)
        pr_ci = stratified_bootstrap_ci(y_te, scores, pr_auc, n_resamples=500, seed=seed)
        ece = expected_calibration_error(y_te, scores)
        out_rows.append(
            {
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
        # Also save predictions for F10 aggregation.
        out_pred = test_df[["host_id", "phage_id", "label"]].copy()
        out_pred["score"] = scores
        out_pred["seed"] = seed
        out_pred["method"] = f"esm8M_host_{mode}_{name}"
        out_pred["split_kind"] = "phage_component"
        out_pred.to_parquet(
            PREDICTIONS_DIR / f"predictions_esm8M_host_{mode}_{name}_phage_component_seed{seed}.parquet",
            index=False,
        )
    return out_rows


def main(identity: float, seeds: list[int], sliding_only: bool = False) -> dict:
    ensure_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tables = load_all()
    loci_concat = dict(
        zip(flatten_loci(tables.loci)["host_id"], flatten_loci(tables.loci)["k_locus_concat"], strict=True)
    )
    phage_vec = _load_phage_embeddings()

    all_rows: list[dict] = []
    for mode in (["sliding"] if sliding_only else ["truncated", "sliding"]):
        sliding = mode == "sliding"
        cache_path = CACHE_DIR / f"host_esm8M_{mode}.parquet"
        if cache_path.exists():
            print(f"[F1] Loading host vectors from cache: {cache_path}", flush=True)
            host_df = pd.read_parquet(cache_path)
            host_vec = {
                row.host_id: np.asarray(row.embedding, dtype=np.float32)
                for row in host_df.itertuples(index=False)
            }
        else:
            print(f"[F1] Computing host embeddings (mode={mode})", flush=True)
            host_vec = embed_hosts(loci_concat, sliding=sliding, model_name="esm2_t6_8M_UR50D")
            host_df = pd.DataFrame(
                [{"host_id": h, "embedding": v.tolist()} for h, v in host_vec.items()]
            )
            host_df.to_parquet(cache_path, index=False)

        for seed in seeds:
            split_path = PROCESSED_DIR / "splits" / f"phage_split_id{identity}_seed{seed}.parquet"
            if not split_path.exists():
                continue
            split_df = pd.read_parquet(split_path)
            rows = _run_one_mode(mode, split_df, phage_vec, host_vec, seed)
            all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)
    out_path = REPORTS_DIR / "f1_sliding_window_results.parquet"
    out_df.to_parquet(out_path, index=False)

    # Aggregate delta (sliding − truncated) per model.
    if not out_df.empty:
        agg = out_df.groupby(["mode", "model"]).agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            ece_mean=("ece", "mean"),
        ).reset_index()
        agg_path = REPORTS_DIR / "f1_sliding_window_summary.parquet"
        agg.to_parquet(agg_path, index=False)
        md_lines = ["# F1 — Sliding-window ESM-2 pooling (8M)",
                    "",
                    "| mode | model | ROC-AUC | PR-AUC | ECE |",
                    "|---|---|---|---|---|"]
        for _, r in agg.iterrows():
            md_lines.append(
                f"| {r.mode} | {r.model} | {r.roc_auc_mean:.3f} ± {r.roc_auc_std:.3f} | "
                f"{r.pr_auc_mean:.3f} ± {r.pr_auc_std:.3f} | {r.ece_mean:.3f} |"
            )
        (REPORTS_DIR / "f1_sliding_window_summary.md").write_text("\n".join(md_lines))
    return {"path_parquet": str(out_path), "n_rows": len(out_df)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(MULTI_SEEDS))
    parser.add_argument("--sliding-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(main(args.identity, list(args.seeds), args.sliding_only), indent=2))
