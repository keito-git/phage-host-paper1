# MIT License. See LICENSE in repository root.
"""F0 — Phage-level leak-free cluster-aware split.

This re-creates the E5 split but clusters *every* RBP of each phage (not
just the first-listed one) and assigns whole connected components to
partitions.  See :mod:`src.data.phage_split` for the detailed rationale.
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import json

import pandas as pd

from src.config import MULTI_SEEDS, PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from src.data.phage_split import phage_level_split
from src.data.phlearn import load_all
from src.data.split import leakage_report
from src.utils.seed import set_global_seed


def main(identity: float, seed: int) -> dict:
    ensure_dirs()
    set_global_seed(seed)

    tables = load_all()
    split, report = phage_level_split(
        tables.interactions,
        tables.rbps,
        identity=identity,
        seed=seed,
    )

    out_dir = PROCESSED_DIR / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phage_split_id{identity}_seed{seed}.parquet"

    assembled = pd.concat(
        [
            split.train.assign(split="train"),
            split.val.assign(split="val"),
            split.test.assign(split="test"),
        ],
        ignore_index=True,
    )
    assembled.to_parquet(out_path, index=False)

    leak = leakage_report(split, id_col="phage_id")

    summary = {
        "identity": identity,
        "seed": seed,
        "n_pairs_train": int(len(split.train)),
        "n_pairs_val": int(len(split.val)),
        "n_pairs_test": int(len(split.test)),
        "n_positives_train": int((split.train.label == 1).sum()),
        "n_positives_val": int((split.val.label == 1).sum()),
        "n_positives_test": int((split.test.label == 1).sum()),
        "leakage": leak,
        "report": {
            "n_rbps": report.n_rbps,
            "n_rbp_clusters": report.n_rbp_clusters,
            "n_phages": report.n_phages,
            "n_phage_components": report.n_phage_components,
            "largest_component": report.largest_component,
            "median_component": report.median_component,
        },
        "output": str(out_path),
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(MULTI_SEEDS))
    args = parser.parse_args()

    reports = []
    for s in args.seeds:
        reports.append(main(identity=args.identity, seed=s))

    (REPORTS_DIR / "experiment_f0_phage_split.json").write_text(
        json.dumps(reports, indent=2)
    )
    print(json.dumps(reports, indent=2))
