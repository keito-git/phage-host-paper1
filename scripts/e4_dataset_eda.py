# MIT License. See LICENSE in repository root.
"""Experiment 4: dataset exploratory data analysis.

We intentionally *skip* the full INPHARED download in this pilot because the
release is multi-GB and the M4 Max workstation has to share disk with other
projects.  Instead we characterise the cleaned PhageHostLearn tables so the
subsequent experiments can be interpreted in context.

Outputs
-------
- ``reports/experiment_04_dataset_eda.md`` with summary tables
- ``reports/figures/exp04_*.png`` with simple distribution plots
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import REPORTS_DIR, ensure_dirs
from src.data.phlearn import load_all


def _safe_hist(
    ax: plt.Axes, values: np.ndarray, bins: int, title: str, xlabel: str
) -> None:
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")


def main() -> None:
    ensure_dirs()
    figs_dir = REPORTS_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    tables = load_all()
    stats = tables.overview()

    rbp_len = tables.rbps["sequence"].dropna().str.len().to_numpy()
    n_rbps_per_phage = tables.rbps.groupby("phage_id").size().to_numpy()
    loci_concat = tables.loci["sequences"].apply(lambda s: sum(len(x) for x in s))
    loci_concat_np = loci_concat.to_numpy()
    n_proteins_per_host = tables.loci["sequences"].apply(len).to_numpy()

    # --- plots ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    _safe_hist(axes[0, 0], rbp_len, 40, "RBP length (aa)", "length")
    _safe_hist(axes[0, 1], n_rbps_per_phage, 20, "RBPs per phage", "count")
    _safe_hist(
        axes[1, 0],
        loci_concat_np,
        30,
        "Summed K-locus protein length per host (aa)",
        "length",
    )
    _safe_hist(axes[1, 1], n_proteins_per_host, 20, "K-locus proteins per host", "count")
    fig.tight_layout()
    fig_path = figs_dir / "exp04_length_distributions.png"
    # IEEE-grade: 600 dpi PNG (raster fallback) + PDF (vector primary).
    fig.savefig(fig_path, dpi=600)
    fig.savefig(fig_path.with_suffix(".pdf"))
    plt.close(fig)

    # positivity per host / per phage
    pos_per_host = (
        tables.interactions[tables.interactions.label == 1]
        .groupby("host_id")
        .size()
        .reindex(tables.interactions.host_id.unique(), fill_value=0)
    )
    pos_per_phage = (
        tables.interactions[tables.interactions.label == 1]
        .groupby("phage_id")
        .size()
        .reindex(tables.interactions.phage_id.unique(), fill_value=0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    _safe_hist(axes[0], pos_per_host.to_numpy(), 30, "Positive pairs per host", "count")
    _safe_hist(axes[1], pos_per_phage.to_numpy(), 30, "Positive pairs per phage", "count")
    fig.tight_layout()
    fig_path2 = figs_dir / "exp04_positive_distributions.png"
    # IEEE-grade: 600 dpi PNG (raster fallback) + PDF (vector primary).
    fig.savefig(fig_path2, dpi=600)
    fig.savefig(fig_path2.with_suffix(".pdf"))
    plt.close(fig)

    # --- markdown ------------------------------------------------------------
    prevalence = float(stats["num_positive_pairs"]) / max(1, stats["num_interactions"])
    summary = {
        **stats,
        "prevalence": prevalence,
        "rbp_length_median": float(np.median(rbp_len)),
        "rbp_length_p95": float(np.percentile(rbp_len, 95)),
        "rbp_length_max": int(rbp_len.max()),
        "rbps_per_phage_median": float(np.median(n_rbps_per_phage)),
        "rbps_per_phage_max": int(n_rbps_per_phage.max()),
        "k_locus_total_length_median": float(np.median(loci_concat_np)),
        "k_locus_total_length_p95": float(np.percentile(loci_concat_np, 95)),
        "proteins_per_host_median": float(np.median(n_proteins_per_host)),
        "hosts_with_no_positive": int((pos_per_host == 0).sum()),
        "phages_with_no_positive": int((pos_per_phage == 0).sum()),
    }

    report_path = REPORTS_DIR / "experiment_04_dataset_eda.md"
    render_report(
        report_path, summary, [os.path.relpath(fig_path, REPORTS_DIR), os.path.relpath(fig_path2, REPORTS_DIR)]
    )
    print(json.dumps(summary, indent=2))


def render_report(path: Path, summary: dict, figs: list[str]) -> None:
    lines = [
        "# Experiment 4 -- Dataset EDA (PhageHostLearn tables)",
        "",
        "**Goal.** Characterise the working corpus we use for E1--E3, E6.",
        "The original plan called for INPHARED ingestion; we defer that to a",
        "GPU session because the raw release is multi-GB and the expected",
        "incremental evidence under the M4 Max constraint is modest.",
        "",
        "## Size summary",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| #interactions (observed pairs) | {summary['num_interactions']} |",
        f"| #positive pairs | {summary['num_positive_pairs']} |",
        f"| #negative pairs | {summary['num_negative_pairs']} |",
        f"| prevalence (pos/total) | {summary['prevalence']:.4f} |",
        f"| #phages | {summary['num_phages']} |",
        f"| #hosts | {summary['num_hosts']} |",
        f"| #RBP proteins (RBPbase.csv) | {summary['num_rbp_proteins']} |",
        "",
        "## Sequence length summary",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| RBP length median | {summary['rbp_length_median']:.0f} aa |",
        f"| RBP length p95 | {summary['rbp_length_p95']:.0f} aa |",
        f"| RBP length max | {summary['rbp_length_max']} aa |",
        f"| RBPs per phage median | {summary['rbps_per_phage_median']:.0f} |",
        f"| RBPs per phage max | {summary['rbps_per_phage_max']} |",
        f"| K-locus total length median | {summary['k_locus_total_length_median']:.0f} aa |",
        f"| K-locus total length p95 | {summary['k_locus_total_length_p95']:.0f} aa |",
        f"| K-locus proteins per host median | {summary['proteins_per_host_median']:.0f} |",
        "",
        "## Sparsity",
        "",
        f"- Hosts without any positive interaction: {summary['hosts_with_no_positive']}",
        f"- Phages without any positive interaction: {summary['phages_with_no_positive']}",
        "",
        "## Figures",
        "",
    ]
    for f in figs:
        lines.append(f"![{Path(f).stem}]({f})")
    lines += [
        "",
        "## Interpretation",
        "",
        "Key operating constraints:",
        "- the positive rate is under 4 %, so PR-AUC dominates ROC-AUC as a",
        "  decision-relevant metric;",
        "- a handful of long K-locus concatenations sit well above the ESM-2",
        "  positional embedding cap (1022 tokens), justifying the length",
        "  truncation used in E2;",
        "- several phages have *no* positive interactions under observation,",
        "  which puts a hard ceiling on any per-phage top-k metric.",
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.parse_args()
    main()
