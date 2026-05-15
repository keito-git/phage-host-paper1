# MIT License. See LICENSE in repository root.
"""Experiment 3: homology-based nearest-neighbour baseline.

Given the cluster-aware split from E5 we compute all-vs-all RBP similarity
(via MMseqs2) and transfer the label of the most similar *training* phage to
each test phage for each host.  This directly answers the typical reviewer
question "can't you just BLAST for this?".

Outputs
-------
``reports/experiment_03_homology.md`` plus the per-seed metrics JSON.
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

from src.baselines.homology import mmseqs_all_vs_all, predict_by_nearest_neighbour
from src.config import CACHE_DIR, MULTI_SEEDS, PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from src.data.phlearn import load_all, pair_with_first_rbp
from src.models.classifiers import evaluate
from src.utils.seed import set_global_seed


def _load_split(identity: float, seed: int) -> pd.DataFrame:
    path = PROCESSED_DIR / "splits" / f"split_id{identity}_seed{seed}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Split missing: {path}. Run E5 first.")
    return pd.read_parquet(path)


def _similarity_df(sequences: dict[str, str], workdir: Path) -> pd.DataFrame:
    """Compute and cache the all-vs-all similarity DataFrame.

    The same similarity matrix is reused across seeds because it only depends
    on the phage sequences, which are identical across splits.
    """
    cache_tsv = workdir / "results.tsv"
    if cache_tsv.exists():
        return pd.read_csv(
            cache_tsv,
            sep="\t",
            header=None,
            names=["query", "target", "fident", "alnlen", "evalue", "bits"],
        )
    return mmseqs_all_vs_all(sequences, workdir=workdir)


def run_once(df: pd.DataFrame, similarity: pd.DataFrame, seed: int) -> dict:
    set_global_seed(seed)
    train = df[df.split == "train"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)

    scores = predict_by_nearest_neighbour(train, test, similarity)
    m = evaluate(test.label.to_numpy(), scores)
    return {
        "n_test_pairs": int(len(test)),
        "n_test_positives": int(test.label.sum()),
        "roc_auc": m.roc_auc,
        "pr_auc": m.pr_auc,
        "best_f1": m.best_f1,
        "best_f1_threshold": m.best_f1_threshold,
    }


def main(identity: float, seeds: list[int]) -> None:
    ensure_dirs()
    tables = load_all()
    pairs = pair_with_first_rbp(tables.interactions, tables.rbps)

    # Unique phage -> RBP representative sequence.
    seq_map = (
        pairs[["phage_id", "rbp_sequence"]]
        .drop_duplicates("phage_id")
        .set_index("phage_id")["rbp_sequence"]
        .to_dict()
    )

    workdir = CACHE_DIR / "mmseqs_homology_allvsall"
    workdir.mkdir(parents=True, exist_ok=True)
    similarity = _similarity_df(seq_map, workdir)

    per_seed: dict[int, dict] = {}
    for s in seeds:
        df = _load_split(identity, s)
        per_seed[s] = run_once(df, similarity, seed=s)

    aggregated: dict[str, float] = {}
    for metric in ("roc_auc", "pr_auc", "best_f1"):
        vals = [per_seed[s][metric] for s in seeds]
        aggregated[f"{metric}_mean"] = float(statistics.mean(vals))
        aggregated[f"{metric}_std"] = float(statistics.pstdev(vals))

    summary = {
        "identity": identity,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "aggregated": aggregated,
    }

    render_report(REPORTS_DIR / "experiment_03_homology.md", summary)
    print(json.dumps(summary, indent=2))


def render_report(path: Path, summary: dict) -> None:
    lines = [
        "# Experiment 3 -- Homology-based nearest-neighbour baseline",
        "",
        "**Goal.** Rule out the trivial objection that a BLAST-style nearest",
        "neighbour of the RBP sequence is already sufficient for host-range",
        "prediction.  We use MMseqs2 all-vs-all on the RBP representatives",
        "and transfer the label of the most similar *training* phage for each",
        "`(test_phage, host)` pair.",
        "",
        f"- Cluster identity (for E5 split): `{summary['identity']}`",
        f"- Seeds: `{summary['seeds']}`",
        "",
        "## Per-seed metrics",
        "",
        "| Seed | n_pairs | n_pos | ROC-AUC | PR-AUC | best F1 |",
        "|---|---|---|---|---|---|",
    ]
    for s, m in summary["per_seed"].items():
        lines.append(
            f"| {s} | {m['n_test_pairs']} | {m['n_test_positives']} | "
            f"{m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['best_f1']:.3f} |"
        )

    agg = summary["aggregated"]
    lines += [
        "",
        "## Aggregated (mean +/- std over seeds)",
        "",
        f"- ROC-AUC: {agg['roc_auc_mean']:.3f} +/- {agg['roc_auc_std']:.3f}",
        f"- PR-AUC:  {agg['pr_auc_mean']:.3f} +/- {agg['pr_auc_std']:.3f}",
        f"- best F1: {agg['best_f1_mean']:.3f} +/- {agg['best_f1_std']:.3f}",
        "",
        "## Interpretation",
        "",
        "A non-trivial PR-AUC here means homology alone already carries",
        "meaningful signal; a near-prevalence PR-AUC means learned",
        "representations are likely needed.  Either outcome is informative as",
        "long as subsequent experiments (E2, E6) are compared on the same",
        "split.",
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(MULTI_SEEDS))
    args = parser.parse_args()
    main(identity=args.identity, seeds=list(args.seeds))
