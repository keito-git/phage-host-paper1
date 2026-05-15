# MIT License. See LICENSE in repository root.
"""F9 — SHAP analysis on the E6 classical-feature XGBoost model.

We train XGBoost on the AAC + dipeptide + ProtParam concatenation of
RBP and K-locus sequences (the same features used by E6) under the
phage-level split (F0) for each seed, then extract the top-20 SHAP
features averaged over seeds.
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import json

import numpy as np
import pandas as pd

from src.config import MULTI_SEEDS, PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from src.data.phlearn import flatten_loci, load_all
from src.eval.shap_analysis import compute_top_features
from src.features.simple_features import summarise_sequence
from src.models.classifiers import make_xgboost
from src.utils.seed import set_global_seed


def _load_split(split_kind: str, identity: float, seed: int) -> pd.DataFrame | None:
    if split_kind == "phage_component":
        p = PROCESSED_DIR / "splits" / f"phage_split_id{identity}_seed{seed}.parquet"
    else:
        p = PROCESSED_DIR / "splits" / f"split_id{identity}_seed{seed}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def build_features(
    df: pd.DataFrame,
    rbp_map: dict[str, str],
    loci_concat: dict[str, str],
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build features for either RBP side, host side, or concatenation."""
    rows_X: list[np.ndarray] = []
    rows_y: list[int] = []
    for row in df.itertuples(index=False):
        rbp_seq = rbp_map.get(row.phage_id)
        host_seq = loci_concat.get(row.host_id)
        if rbp_seq is None or host_seq is None:
            rbp_seq = rbp_seq or ""
            host_seq = host_seq or ""
        if side == "rbp":
            vec = summarise_sequence(rbp_seq)
        elif side == "host":
            vec = summarise_sequence(host_seq)
        else:  # concat (but SHAP will treat as one feature block per side)
            vec = np.concatenate([summarise_sequence(rbp_seq), summarise_sequence(host_seq)])
        rows_X.append(vec.astype(np.float32))
        rows_y.append(int(row.label))
    return np.vstack(rows_X), np.asarray(rows_y, dtype=np.int64)


def main(identity: float, seeds: list[int], split_kind: str) -> dict:
    ensure_dirs()
    tables = load_all()
    rbp_map = dict(zip(tables.rbps["phage_id"], tables.rbps["sequence"], strict=True))
    loci_concat = dict(
        zip(
            flatten_loci(tables.loci)["host_id"],
            flatten_loci(tables.loci)["k_locus_concat"],
            strict=True,
        )
    )

    # We compute SHAP separately for the RBP side so the feature names map
    # cleanly onto the 425 classical features in src.eval.shap_analysis.
    all_tops: list[pd.DataFrame] = []
    for seed in seeds:
        split_df = _load_split(split_kind, identity, seed)
        if split_df is None:
            continue
        set_global_seed(seed)
        train_df = split_df[split_df.split == "train"]
        test_df = split_df[split_df.split == "test"]

        X_tr, y_tr = build_features(train_df, rbp_map, loci_concat, side="rbp")
        X_te, _ = build_features(test_df, rbp_map, loci_concat, side="rbp")

        scale = max(1.0, (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
        clf = make_xgboost(seed, scale_pos_weight=scale)
        clf.fit(X_tr, y_tr)
        # SHAP on a bounded sample of test points keeps runtime modest.
        sample_n = min(len(X_te), 500)
        X_for_shap = X_te[:sample_n]
        tops = compute_top_features(clf, X_for_shap, top_k=30)
        df_top = tops.top_features.assign(seed=seed)
        all_tops.append(df_top)

    if not all_tops:
        return {"status": "no_seeds_evaluated"}

    combined = pd.concat(all_tops, ignore_index=True)
    agg = (
        combined.groupby("feature_name")
        .agg(mean_shap=("mean_abs_shap", "mean"), std_shap=("mean_abs_shap", "std"), n_seeds=("seed", "nunique"))
        .reset_index()
        .sort_values("mean_shap", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    out_path = REPORTS_DIR / "f9_shap_top20.parquet"
    agg.to_parquet(out_path, index=False)

    md_path = REPORTS_DIR / "f9_shap_top20.md"
    md_lines = [
        "# F9 — SHAP top-20 features on RBP classical features (XGBoost)",
        "",
        f"**Split kind:** `{split_kind}`  |  **seeds:** {seeds}",
        "",
        "| rank | feature | mean\\|SHAP\\| | std\\|SHAP\\| | n_seeds |",
        "|---|---|---|---|---|",
    ]
    for i, row in agg.iterrows():
        md_lines.append(
            f"| {i+1} | `{row.feature_name}` | "
            f"{row.mean_shap:.4f} | {row.std_shap:.4f} | {int(row.n_seeds)} |"
        )
    md_path.write_text("\n".join(md_lines))

    return {
        "output_parquet": str(out_path),
        "output_md": str(md_path),
        "top_features": agg.to_dict(orient="records"),
        "n_seeds_used": len(all_tops),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(MULTI_SEEDS))
    parser.add_argument("--split-kind", choices=["rbp_cluster", "phage_component"],
                        default="phage_component")
    args = parser.parse_args()
    out = main(args.identity, list(args.seeds), args.split_kind)
    print(json.dumps(out, indent=2, default=str))
