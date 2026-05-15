# MIT License. See LICENSE in repository root.
"""F4 — K-type stratified evaluation.

Since Kaptive requires external BLAST databases we fall back to the
cluster-surrogate K-typing introduced in :func:`src.eval.ktype.wzy_wzx_fallback_typing`.
We call out the fallback explicitly in the report (``k_type_source =
cluster_surrogate``) so the findings report never overstates the
granularity of the K-typing.
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import json

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from src.data.phlearn import load_all
from src.eval.aggregate import load_all_predictions
from src.eval.ktype import stratified_metrics, wzy_wzx_fallback_typing


def main() -> dict:
    ensure_dirs()
    tables = load_all()
    ktype_df = wzy_wzx_fallback_typing(tables.loci, identity=0.6)
    ktype_df.to_parquet(PROCESSED_DIR / "host_ktypes.parquet", index=False)

    preds = load_all_predictions(PROCESSED_DIR / "predictions")
    if preds.empty:
        return {"status": "no_predictions"}

    rows: list[pd.DataFrame] = []
    for (method, split_kind), sub in preds.groupby(["method", "split_kind"]):
        # Evaluate per seed but also pooled; pooled is more stable for small K-types.
        pooled = sub.rename(columns={"score": "score"})
        strat = stratified_metrics(pooled, ktype_df)
        strat["method"] = method
        strat["split_kind"] = split_kind
        rows.append(strat)

    if not rows:
        return {"status": "no_stratifications"}
    combined = pd.concat(rows, ignore_index=True)
    combined.to_parquet(REPORTS_DIR / "f4_ktype_stratified.parquet", index=False)

    # Human-readable markdown
    md = ["# F4 — K-type stratified evaluation (cluster-surrogate typing)", ""]
    md.append(
        "**Note.** K-typing is performed via MMseqs2 clustering over full K-locus "
        "concatenations (`identity=0.6`), not Kaptive.  Cluster labels are "
        "shown as surrogate K-types; absolute K-type identity (K1/K2/...) is "
        "not claimed here."
    )
    md.append("")
    for (method, split_kind), sub in combined.groupby(["method", "split_kind"]):
        md.append(f"## {method} / {split_kind}")
        md.append("")
        md.append("| k_type | n_pairs | n_pos | ROC-AUC | PR-AUC |")
        md.append("|---|---|---|---|---|")
        for _, r in sub.head(12).iterrows():
            md.append(
                f"| {r.k_type[:18]}... | {int(r.n_pairs)} | "
                f"{int(r.n_positives)} | {r.metric:.3f} | {r.pr_auc:.3f} |"
            )
        md.append("")
    (REPORTS_DIR / "f4_ktype_stratified.md").write_text("\n".join(md))
    return {
        "path_parquet": str(REPORTS_DIR / "f4_ktype_stratified.parquet"),
        "path_md": str(REPORTS_DIR / "f4_ktype_stratified.md"),
        "n_ktypes_overall": int(ktype_df.k_type.nunique()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(main(), indent=2))
