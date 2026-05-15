# MIT License. See LICENSE in repository root.
"""F10 — Aggregate all method × split × seed predictions into main tables.

Produces
--------
* ``reports/f10_per_run.parquet`` — one row per (method, split_kind, seed)
  with point estimates and 95% bootstrap CIs for ROC-AUC / PR-AUC and ECE.
* ``reports/f10_summary.parquet`` — one row per (method, split_kind),
  aggregated across seeds (mean and std).
* ``reports/f10_delong_matrix.json`` — pairwise DeLong p-values computed on
  the concatenated predictions per split_kind.
* ``reports/f10_main_figure.png`` — grouped bar chart (Matplotlib).
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from src.eval.aggregate import load_all_predictions, method_split_table, summarise_across_seeds
from src.stats.metrics import delong_test


def compute_delong_matrix(preds: pd.DataFrame) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Compute pairwise DeLong matrices per (split_kind, seed)."""
    results: dict = {}
    groups = preds.groupby(["split_kind", "seed"])
    for (split_kind, seed), sub in groups:
        methods = sorted(sub.method.unique())
        per_seed: dict[str, dict[str, float]] = {}
        wide = sub.pivot_table(
            index=["host_id", "phage_id"], columns="method", values="score"
        )
        labels = sub.drop_duplicates(["host_id", "phage_id"]).set_index(["host_id", "phage_id"])["label"]
        wide = wide.loc[labels.index]
        y = labels.to_numpy().astype(int)
        for a in methods:
            for b in methods:
                if a >= b:
                    continue
                try:
                    res = delong_test(wide[a].to_numpy(), wide[b].to_numpy(), y)
                    per_seed.setdefault(a, {})[b] = {
                        "delta": res.delta,
                        "z": res.z,
                        "p_value": res.p_value,
                    }
                except Exception as e:  # noqa: BLE001
                    per_seed.setdefault(a, {})[b] = {
                        "error": f"{type(e).__name__}: {e}"
                    }
        results.setdefault(split_kind, {})[int(seed)] = per_seed
    return results


def plot_main_figure(summary: pd.DataFrame, out_path) -> None:
    if summary.empty:
        return
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
    ax.set_title("Method × split kind — ROC-AUC")
    ax.axhline(0.5, linestyle="--", color="grey", linewidth=1)
    ax.legend(title="split_kind")
    ax.set_ylim(0.4, 0.9)
    plt.tight_layout()
    # IEEE-grade: 600 dpi PNG (raster fallback) + PDF (vector primary).
    out_path = type(out_path)(out_path)  # ensure pathlib-like
    plt.savefig(out_path, dpi=600)
    try:
        pdf_path = out_path.with_suffix(".pdf")
        plt.savefig(pdf_path)
    except AttributeError:
        # Fallback when out_path is a plain str.
        from pathlib import Path

        plt.savefig(Path(str(out_path)).with_suffix(".pdf"))
    plt.close(fig)


def main() -> dict:
    ensure_dirs()
    preds = load_all_predictions(PROCESSED_DIR / "predictions")
    if preds.empty:
        return {"status": "no_predictions"}

    per_run = method_split_table(preds, n_bootstrap=1000)
    per_run_path = REPORTS_DIR / "f10_per_run.parquet"
    per_run.to_parquet(per_run_path, index=False)

    summary = summarise_across_seeds(per_run)
    summary_path = REPORTS_DIR / "f10_summary.parquet"
    summary.to_parquet(summary_path, index=False)

    delong = compute_delong_matrix(preds)
    delong_path = REPORTS_DIR / "f10_delong_matrix.json"
    delong_path.write_text(json.dumps(delong, indent=2, default=float))

    fig_path = REPORTS_DIR / "f10_main_figure.png"
    plot_main_figure(summary, fig_path)

    md_lines = [
        "# F10 — Method × Split kind summary",
        "",
        f"**Bootstrap:** n=1000 per run. **Seeds:** {sorted(per_run['seed'].unique().tolist())}.",
        "",
        "## Aggregated across seeds",
        "",
        "| method | split_kind | n_seeds | ROC-AUC (mean ± std) | PR-AUC (mean ± std) | ECE (mean ± std) |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        md_lines.append(
            f"| {r.method} | {r.split_kind} | {int(r.n_seeds)} | "
            f"{r.roc_auc_mean:.3f} ± {r.roc_auc_std:.3f} | "
            f"{r.pr_auc_mean:.3f} ± {r.pr_auc_std:.3f} | "
            f"{r.ece_mean:.3f} ± {r.ece_std:.3f} |"
        )
    (REPORTS_DIR / "f10_summary.md").write_text("\n".join(md_lines))

    return {
        "per_run_rows": int(len(per_run)),
        "summary_rows": int(len(summary)),
        "per_run": str(per_run_path),
        "summary": str(summary_path),
        "delong": str(delong_path),
        "figure": str(fig_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    print(json.dumps(main(), indent=2))
