# MIT License. See LICENSE in repository root.
"""F+ — DNA-level k-mer Markov baseline (WIsH analog, supplementary).

This supplements the amino-acid k-mer Markov baseline in
``f_benchmark_main.py`` with a DNA-level variant at k=6 (WIsH paper body
default).  DNA is available for phage RBPs via ``RBPbase.csv`` column
``dna_sequence``; host K-locus is only available as protein, so we apply
the same "per-host Markov on positively-interacting phage DNA" scheme as
the AA variant (see ``src/baselines/dna_kmer_markov.py`` for the
clean-room statement).

Outputs
-------
* ``data/processed/predictions/predictions_dna_kmer_markov_k{K}_{split_kind}_seed{seed}.parquet``
* ``reports/f_plus_dna_kmer_markov_2026-04-24.md`` — summary table

The result is explicitly labelled as **supplementary** in the report so
reviewers can see it without it displacing the main benchmark rows.
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from src.baselines.dna_kmer_markov import train as dna_train
from src.config import PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from src.data.phlearn import load_all
from src.stats.metrics import (
    expected_calibration_error,
    pr_auc,
    roc_auc,
    stratified_bootstrap_ci,
)
from src.utils.seed import set_global_seed

PREDICTIONS_DIR = PROCESSED_DIR / "predictions"


def _load_split(split_kind: str, identity: float, seed: int) -> pd.DataFrame | None:
    if split_kind == "rbp_cluster":
        path = PROCESSED_DIR / "splits" / f"split_id{identity}_seed{seed}.parquet"
    elif split_kind == "phage_component":
        path = PROCESSED_DIR / "splits" / f"phage_split_id{identity}_seed{seed}.parquet"
    else:
        raise ValueError(f"unknown split kind: {split_kind}")
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _build_phage_dna_map(tables) -> dict[str, str]:
    """Return phage_id -> concatenated DNA of all its RBPs (A/C/G/T only)."""
    df = tables.rbps.copy()
    df = df[df["dna"].notna()]
    concat: dict[str, str] = {}
    for pid, sub in df.groupby("phage_id"):
        joined = "".join(str(d).upper() for d in sub["dna"].tolist() if isinstance(d, str))
        # sanitise to A/C/G/T only
        concat[pid] = "".join(c for c in joined if c in ("A", "C", "G", "T"))
    return concat


def _run_one(
    split_df: pd.DataFrame,
    phage_dna: dict[str, str],
    seed: int,
    k: int,
) -> np.ndarray:
    """Per-host Markov model trained on positively-interacting phage DNA; score = LL."""
    set_global_seed(seed)
    train_df = split_df[split_df.split == "train"]
    test_df = split_df[split_df.split == "test"].reset_index(drop=True)

    host_models: dict[str, object] = {}
    for host, sub in train_df.groupby("host_id"):
        pos = sub[sub.label == 1]
        seqs = [
            phage_dna[pid]
            for pid in pos.phage_id
            if pid in phage_dna and len(phage_dna[pid]) > k + 10
        ]
        if len(seqs) >= 2:
            host_models[host] = dna_train(seqs, k=k, alpha=1.0)

    prevalence = float(train_df.label.mean())
    scores = np.full(len(test_df), prevalence, dtype=np.float32)
    for i, row in enumerate(test_df.itertuples(index=False)):
        model = host_models.get(row.host_id)
        seq = phage_dna.get(row.phage_id)
        if model is None or seq is None:
            continue
        scores[i] = model.mean_log_likelihood(seq)  # type: ignore[attr-defined]
    return scores


def main(identity: float, seeds: list[int], k_values: list[int]) -> dict:
    ensure_dirs()
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    tables = load_all()
    phage_dna = _build_phage_dna_map(tables)
    print(f"[F+DNA] loaded {len(phage_dna)} phages with DNA, "
          f"median len = {int(np.median([len(v) for v in phage_dna.values()]))}")

    rows: list[dict] = []
    for k in k_values:
        for split_kind in ("rbp_cluster", "phage_component"):
            for seed in seeds:
                split_df = _load_split(split_kind, identity, seed)
                if split_df is None:
                    continue
                test_df = split_df[split_df.split == "test"].reset_index(drop=True)
                if test_df.empty:
                    continue
                if (test_df.label == 1).sum() < 2 or (test_df.label == 0).sum() < 2:
                    continue
                scores = _run_one(split_df, phage_dna, seed, k=k)

                # metrics
                y = test_df.label.to_numpy()
                roc_ci = stratified_bootstrap_ci(y, scores, roc_auc, n_resamples=500, seed=seed)
                pr_ci = stratified_bootstrap_ci(y, scores, pr_auc, n_resamples=500, seed=seed)
                # ECE needs [0, 1] scores; log-likelihoods are not, so rescale per-run
                finite = np.isfinite(scores)
                if finite.any():
                    s_min, s_max = scores[finite].min(), scores[finite].max()
                    denom = max(1e-6, s_max - s_min)
                    scores_norm = np.where(finite, (scores - s_min) / denom, 0.5)
                else:
                    scores_norm = np.full_like(scores, 0.5)
                ece = expected_calibration_error(y, scores_norm)

                rows.append(
                    {
                        "method": f"dna_kmer_markov_k{k}",
                        "split_kind": split_kind,
                        "seed": seed,
                        "k": k,
                        "n_test": int(len(test_df)),
                        "n_pos": int((y == 1).sum()),
                        "roc_auc": roc_ci.point,
                        "roc_auc_lo": roc_ci.lower,
                        "roc_auc_hi": roc_ci.upper,
                        "pr_auc": pr_ci.point,
                        "pr_auc_lo": pr_ci.lower,
                        "pr_auc_hi": pr_ci.upper,
                        "ece": ece,
                    }
                )

                # Save predictions parquet (new file; never overwrite existing).
                out = test_df[["host_id", "phage_id", "label"]].copy()
                out["method"] = f"dna_kmer_markov_k{k}"
                out["split_kind"] = split_kind
                out["seed"] = seed
                out["score"] = scores.astype(np.float32)
                out_path = PREDICTIONS_DIR / (
                    f"predictions_dna_kmer_markov_k{k}_{split_kind}_seed{seed}.parquet"
                )
                out.to_parquet(out_path, index=False)

    # Write summary report — dated new file; do not overwrite existing reports.
    today = date.today().isoformat()
    df = pd.DataFrame(rows)
    report_path = REPORTS_DIR / f"f_plus_dna_kmer_markov_{today}.md"
    agg_lines: list[str] = [
        f"# F+ — DNA k-mer Markov baseline (supplementary, {today})",
        "",
        "**Design:** clean-room MIT implementation of a WIsH-style DNA k-mer ",
        "Markov model (see `src/baselines/dna_kmer_markov.py`).  Host-specific ",
        "models are trained on the DNA of phage RBPs that positively interact ",
        "with each host in the training set.  Test-pair score = mean log-",
        "likelihood of the test phage's RBP DNA under the candidate host's ",
        "model.  ECE is computed after min-max rescaling per run since LL is ",
        "not a calibrated probability.",
        "",
        "**Status:** supplementary; the main benchmark (`f10_summary.md`) uses ",
        "the amino-acid k=3 variant (`kmer_markov`).",
        "",
        "## Aggregated per (method, split_kind) across seeds",
        "",
        "| method | split_kind | n_seeds | ROC-AUC (mean ± std) | PR-AUC (mean ± std) | ECE (mean) |",
        "|---|---|---|---|---|---|",
    ]
    if not df.empty:
        agg = df.groupby(["method", "split_kind"]).agg(
            n=("seed", "count"),
            roc_m=("roc_auc", "mean"),
            roc_s=("roc_auc", "std"),
            pr_m=("pr_auc", "mean"),
            pr_s=("pr_auc", "std"),
            ece_m=("ece", "mean"),
        ).reset_index()
        for _, r in agg.iterrows():
            agg_lines.append(
                f"| {r.method} | {r.split_kind} | {int(r.n)} | "
                f"{r.roc_m:.3f} ± {r.roc_s:.3f} | "
                f"{r.pr_m:.3f} ± {r.pr_s:.3f} | "
                f"{r.ece_m:.3f} |"
            )
    agg_lines.append("")
    agg_lines.append("## Per-seed detail")
    agg_lines.append("")
    agg_lines.append("| method | split_kind | seed | n_test | n_pos | ROC-AUC | PR-AUC | ECE |")
    agg_lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        agg_lines.append(
            f"| {r['method']} | {r['split_kind']} | {r['seed']} | {r['n_test']} | "
            f"{r['n_pos']} | {r['roc_auc']:.3f} [{r['roc_auc_lo']:.3f}, {r['roc_auc_hi']:.3f}] | "
            f"{r['pr_auc']:.3f} [{r['pr_auc_lo']:.3f}, {r['pr_auc_hi']:.3f}] | {r['ece']:.3f} |"
        )
    report_path.write_text("\n".join(agg_lines))
    return {"n_rows": len(rows), "report": str(report_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument("--k", type=int, nargs="*", default=[6])
    args = parser.parse_args()
    out = main(args.identity, list(args.seeds), list(args.k))
    print(json.dumps(out, indent=2))
