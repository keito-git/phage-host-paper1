# MIT License. See LICENSE in repository root.
"""K-type stratification helper (F4).

The primary K-typing tool we use is Kaptive
(https://github.com/klebgenomics/Kaptive, GPLv3).  Kaptive requires an
external nucleotide database and a blastn binary, so it is out of process.
This module therefore provides two entry points:

1. :func:`parse_kaptive_report` — reads the JSON produced by Kaptive 3.x.
2. :func:`wzy_wzx_fallback_typing` — a homology-based fallback that clusters
   the ``wzy`` / ``wzx`` marker sequences out of each host's K-locus set
   via MMseqs2.  This is the "副" option called out in the Paper 1 plan.

Both return a ``DataFrame[host_id, k_type, k_type_source]`` that downstream
per-K-type stratified evaluators can join against the split.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd


def parse_kaptive_report(path: Path) -> pd.DataFrame:
    """Parse a Kaptive v3 JSON report into (host_id, k_type) tuples.

    The report format evolved across Kaptive releases; we only rely on the
    lowest-common keys:

    * ``best_match.type`` or ``best_match.locus``
    * the top-level assembly name, which we map to ``host_id`` via a provided
      ``name -> host_id`` map (Kaptive takes fasta filenames as IDs).

    Callers that run Kaptive with one FASTA per host can skip the remapping
    by naming the files ``<host_id>.fna``.
    """
    payload = json.loads(Path(path).read_text())
    rows: list[dict[str, str]] = []
    for entry in payload:
        name = entry.get("assembly") or entry.get("name")
        best = entry.get("best_match", {})
        k_type = (
            best.get("type")
            or best.get("locus")
            or "unknown"
        )
        rows.append({"host_id": str(name), "k_type": str(k_type), "k_type_source": "kaptive"})
    return pd.DataFrame(rows)


def wzy_wzx_fallback_typing(
    loci: pd.DataFrame,
    identity: float = 0.8,
    workdir: Path | None = None,
) -> pd.DataFrame:
    """Cluster ``wzy`` + ``wzx`` proteins as a crude K-type proxy.

    ``loci`` is the DataFrame produced by :func:`src.data.phlearn.load_loci`
    with columns ``host_id`` and ``sequences`` (list[str]).  We take the
    concatenated ``wzy + wzx`` region as the typing anchor; if annotations
    are not available in Locibase (they usually aren't — it ships raw
    protein lists), we fall back to clustering the *full* K-locus
    concatenation and treat cluster IDs as surrogate K-types.

    This is a deliberate approximation.  The report column
    ``k_type_source`` is set to ``"cluster_surrogate"`` to remind the
    downstream user not to over-interpret.
    """
    from src.data.phlearn import flatten_loci
    from src.data.split import mmseqs_cluster

    flat = flatten_loci(loci)
    seq_map = dict(zip(flat["host_id"], flat["k_locus_concat"], strict=True))
    clusters = mmseqs_cluster(seq_map, identity=identity, workdir=workdir)
    clusters = clusters.rename(columns={"sequence_id": "host_id", "cluster_id": "k_type"})
    clusters["k_type_source"] = "cluster_surrogate"
    return clusters


def stratified_metrics(
    pair_df: pd.DataFrame,
    ktype_df: pd.DataFrame,
    scores: dict[str, dict[int, float]] | None = None,
    metric_fn: Callable[..., float] | None = None,
) -> pd.DataFrame:
    """Evaluate a scalar metric for each K-type slice separately.

    Parameters
    ----------
    pair_df:
        Pair-level evaluation frame with ``host_id``, ``phage_id``, ``label``,
        and ``score`` columns.
    ktype_df:
        Output of :func:`parse_kaptive_report` or :func:`wzy_wzx_fallback_typing`.
    scores, metric_fn:
        Optional; present for API symmetry but unused in this reference
        implementation.  Callers typically pre-fill ``pair_df["score"]`` and
        pass ``metric_fn = lambda y, s: roc_auc_score(y, s)``.

    Returns
    -------
    DataFrame with one row per K-type and columns
    ``[k_type, n_pairs, n_positives, metric]``.  K-types with fewer than
    two positives are dropped because per-slice ROC-AUC is undefined.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    metric = metric_fn or roc_auc_score

    if "score" not in pair_df.columns:
        raise ValueError("pair_df must contain a 'score' column")
    merged = pair_df.merge(ktype_df, on="host_id", how="left")
    merged["k_type"] = merged["k_type"].fillna("unassigned")

    out_rows: list[dict[str, float | str | int]] = []
    for k_type, sub in merged.groupby("k_type"):
        n_pos = int((sub["label"] == 1).sum())
        n_neg = int((sub["label"] == 0).sum())
        if n_pos < 2 or n_neg < 2:
            continue
        try:
            m = float(metric(sub["label"].to_numpy(), sub["score"].to_numpy()))
        except ValueError:
            continue
        try:
            ap = float(average_precision_score(sub["label"].to_numpy(), sub["score"].to_numpy()))
        except ValueError:
            ap = float("nan")
        out_rows.append(
            {
                "k_type": str(k_type),
                "n_pairs": int(len(sub)),
                "n_positives": n_pos,
                "n_negatives": n_neg,
                "metric": m,
                "pr_auc": ap,
            }
        )
    return pd.DataFrame(out_rows).sort_values("metric", ascending=False)
