# MIT License. See LICENSE in repository root.
"""Regenerate paper figures at IEEE-grade resolution.

This script *only* writes into ``paper/figures/`` — it does **not** touch
``reports/`` (which is treated as immutable for this task).

Outputs (per figure, side-by-side):
  - ``<name>.pdf``  — vector format, primary inclusion target for IEEEtran
  - ``<name>.png``  — 600 dpi raster fallback (IEEE recommended)

Inputs are pulled from existing artefacts:
  - ``reports/f10_summary.parquet``  (read-only)
  - PhageHostLearn tables via ``src.data.phlearn.load_all`` (read-only)

The plotting code intentionally mirrors the *layout* of the original
generators (``f10_aggregate.plot_main_figure`` and ``e4_dataset_eda.main``)
so the only visible change is resolution / format. Figure dimensions
(figsize), axis labels, colours, bin counts and order are preserved.

Usage
-----
    cd code
    python scripts/regen_paper_figures_hires.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ensure_path

ensure_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import REPORTS_DIR
from src.data.phlearn import load_all

# ---------------------------------------------------------------------------
# Paths

ROOT = Path(__file__).resolve().parents[2]  # project root
PAPER_FIGS = ROOT / "paper" / "figures"

# IEEE recommends 600 dpi for raster figures; keep PDF as vector primary.
RASTER_DPI = 600


# ---------------------------------------------------------------------------
# F10 main figure (grouped bar chart of ROC-AUC)


def plot_f10_main_figure(summary: pd.DataFrame, base_path: Path) -> list[Path]:
    """Reproduces ``f10_aggregate.plot_main_figure`` layout at hi-res.

    Layout (figsize, bar widths, ticks, legend, ylim) is identical to the
    original; only resolution and emitted formats change.
    """
    pivot = summary.pivot(index="method", columns="split_kind", values="roc_auc_mean")
    pivot_std = summary.pivot(index="method", columns="split_kind", values="roc_auc_std")
    pivot = pivot.reindex(sorted(pivot.index), axis=0)
    pivot_std = pivot_std.reindex(pivot.index, axis=0)

    methods = pivot.index.tolist()
    split_kinds = pivot.columns.tolist()
    x = np.arange(len(methods))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, sk in enumerate(split_kinds):
        vals = pivot[sk].to_numpy()
        errs = pivot_std[sk].to_numpy() if sk in pivot_std.columns else None
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            label=sk,
            yerr=errs,
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("ROC-AUC (mean over seeds)")
    ax.set_title("Method x split kind -- ROC-AUC")
    ax.axhline(0.5, linestyle="--", color="grey", linewidth=1)
    ax.legend(title="split_kind")
    ax.set_ylim(0.4, 0.9)
    plt.tight_layout()

    pdf_path = base_path.with_suffix(".pdf")
    png_path = base_path.with_suffix(".png")
    fig.savefig(pdf_path)  # vector
    fig.savefig(png_path, dpi=RASTER_DPI)  # 600 dpi raster fallback
    plt.close(fig)
    return [pdf_path, png_path]


# ---------------------------------------------------------------------------
# Exp04 EDA figures (length & positivity distributions)


def _safe_hist(ax, values: np.ndarray, bins: int, title: str, xlabel: str) -> None:
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")


def plot_exp04_figures(base_dir: Path) -> list[Path]:
    """Reproduces ``e4_dataset_eda.main`` figure layout at hi-res."""
    tables = load_all()

    rbp_len = tables.rbps["sequence"].dropna().str.len().to_numpy()
    n_rbps_per_phage = tables.rbps.groupby("phage_id").size().to_numpy()
    loci_concat = tables.loci["sequences"].apply(lambda s: sum(len(x) for x in s))
    loci_concat_np = loci_concat.to_numpy()
    n_proteins_per_host = tables.loci["sequences"].apply(len).to_numpy()

    written: list[Path] = []

    # --- length distributions (2x2 grid, figsize 9x7) -----------------------
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
    base_len = base_dir / "exp04_length_distributions"
    fig.savefig(base_len.with_suffix(".pdf"))
    fig.savefig(base_len.with_suffix(".png"), dpi=RASTER_DPI)
    plt.close(fig)
    written += [base_len.with_suffix(".pdf"), base_len.with_suffix(".png")]

    # --- positivity distributions (1x2, figsize 9x3.5) ----------------------
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
    base_pos = base_dir / "exp04_positive_distributions"
    fig.savefig(base_pos.with_suffix(".pdf"))
    fig.savefig(base_pos.with_suffix(".png"), dpi=RASTER_DPI)
    plt.close(fig)
    written += [base_pos.with_suffix(".pdf"), base_pos.with_suffix(".png")]

    return written


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)

    # F10 main figure — read precomputed summary, regenerate at hi-res.
    summary_path = REPORTS_DIR / "f10_summary.parquet"
    summary = pd.read_parquet(summary_path)
    f10_outputs = plot_f10_main_figure(summary, PAPER_FIGS / "f10_main_figure")

    # Exp04 EDA figures — recompute from raw tables (deterministic).
    exp04_outputs = plot_exp04_figures(PAPER_FIGS)

    print("Wrote:")
    for p in [*f10_outputs, *exp04_outputs]:
        print(f"  - {p.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.parse_args()
    main()
