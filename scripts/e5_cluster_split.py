# MIT License. See LICENSE in repository root.
"""Experiment 5: leak-free phage-level cluster-aware split.

Outputs
-------
``data/processed/splits/split_id{identity}_seed{seed}.parquet`` containing the
pair-level split assignment with columns
``[host_id, phage_id, label, split, cluster_id]``.

The cluster report is written to ``reports/experiment_05_cluster_split.md``.
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from src.data.phlearn import load_all, pair_with_first_rbp
from src.data.split import cluster_aware_split, leakage_report
from src.utils.seed import set_global_seed


def main(identity: float, seed: int) -> None:
    ensure_dirs()
    set_global_seed(seed)

    tables = load_all()
    pairs = pair_with_first_rbp(tables.interactions, tables.rbps)

    result = cluster_aware_split(
        pairs,
        sequence_col="rbp_sequence",
        id_col="phage_id",
        identity=identity,
        seed=seed,
    )

    out_dir = PROCESSED_DIR / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"split_id{identity}_seed{seed}.parquet"

    assembled = pd.concat(
        [
            result.train.assign(split="train"),
            result.val.assign(split="val"),
            result.test.assign(split="test"),
        ],
        axis=0,
        ignore_index=True,
    )
    assembled.to_parquet(out_path, index=False)

    leak = leakage_report(result, id_col="phage_id")
    cluster_sizes = result.clusters.groupby("cluster_id").size()
    summary = {
        "identity": identity,
        "seed": seed,
        "n_pairs_total": int(len(assembled)),
        "n_pairs_train": int(len(result.train)),
        "n_pairs_val": int(len(result.val)),
        "n_pairs_test": int(len(result.test)),
        "n_clusters": int(result.clusters.cluster_id.nunique()),
        "largest_cluster": int(cluster_sizes.max()),
        "median_cluster_size": float(cluster_sizes.median()),
        "leakage_report": leak,
        "positives_by_split": {
            k: int((df.label == 1).sum())
            for k, df in {"train": result.train, "val": result.val, "test": result.test}.items()
        },
        "output": str(out_path),
    }

    (REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "experiment_05_cluster_split.md"
    render_report(report_path, summary)

    print(json.dumps(summary, indent=2))


def render_report(path: Path, summary: dict) -> None:
    path.write_text(
        "# Experiment 5 — Cluster-aware train/val/test split\n\n"
        "**Goal.** Create a leak-free partitioning at the RBP cluster level so\n"
        "downstream baselines (E2, E3, E6) are evaluated on phages whose RBP\n"
        "is not redundant with any training phage.\n\n"
        f"- Identity threshold: `{summary['identity']}`\n"
        f"- Seed: `{summary['seed']}`\n"
        f"- Total pairs: {summary['n_pairs_total']}\n"
        f"- Train / Val / Test pairs: "
        f"{summary['n_pairs_train']} / {summary['n_pairs_val']} / {summary['n_pairs_test']}\n"
        f"- Positives per split: {summary['positives_by_split']}\n"
        f"- Number of RBP clusters: {summary['n_clusters']}\n"
        f"- Largest cluster size: {summary['largest_cluster']}\n"
        f"- Median cluster size: {summary['median_cluster_size']}\n"
        f"- Leakage report: `{summary['leakage_report']}`\n"
        f"- Output parquet: `{summary['output']}`\n\n"
        "## Interpretation\n\n"
        "All overlap counts between train/val/test must be **zero** — this is\n"
        "the minimum guarantee before any metric reported in subsequent\n"
        "experiments can be taken at face value.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(identity=args.identity, seed=args.seed)
