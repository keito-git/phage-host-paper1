# MIT License. See LICENSE in repository root.
"""Homology-based baseline for phage-host interaction prediction (E3).

Implements the classic "nearest-neighbour in RBP space predicts host
specificity" heuristic using MMseqs2 all-vs-all alignment.  Given a test
phage, we look up the most similar train phage and propagate its K-locus
affinities as the predicted score.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_DIR


def mmseqs_all_vs_all(
    sequences: dict[str, str],
    sensitivity: float = 7.5,
    workdir: Path | None = None,
    mmseqs_bin: str = "mmseqs",
) -> pd.DataFrame:
    """Run ``mmseqs easy-search`` against itself.

    Returns a DataFrame with the default BLAST-tab fields; the caller is
    expected to join this on the query / target IDs.
    """
    if shutil.which(mmseqs_bin) is None:
        raise RuntimeError(f"mmseqs binary '{mmseqs_bin}' not found")
    workdir = Path(workdir) if workdir else CACHE_DIR / "mmseqs_search"
    workdir.mkdir(parents=True, exist_ok=True)

    fasta = workdir / "seqs.fasta"
    with fasta.open("w") as fh:
        for name, seq in sequences.items():
            fh.write(f">{name}\n{seq}\n")

    out_tsv = workdir / "results.tsv"
    tmp = workdir / "tmp"
    tmp.mkdir(exist_ok=True)

    cmd = [
        mmseqs_bin,
        "easy-search",
        str(fasta),
        str(fasta),
        str(out_tsv),
        str(tmp),
        "-s",
        str(sensitivity),
        "--format-output",
        "query,target,fident,alnlen,evalue,bits",
        "-v",
        "1",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    df = pd.read_csv(
        out_tsv, sep="\t", header=None,
        names=["query", "target", "fident", "alnlen", "evalue", "bits"],
    )
    return df


def predict_by_nearest_neighbour(
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    similarity: pd.DataFrame,
    id_col: str = "phage_id",
    host_col: str = "host_id",
    label_col: str = "label",
) -> np.ndarray:
    """Return a similarity-weighted positive-class score for each test row.

    For each (test_phage, host) we:
        1. Collect all train phages with some positive or negative label for
           that host.
        2. Compute the best similarity of the test phage to any of those
           train phages (via ``similarity``).
        3. Return the label of the most similar train phage.  If no labelled
           train phage exists for the host we fall back to the global
           prevalence.
    """
    prevalence = train_pairs[label_col].mean()
    sim_lookup = similarity.set_index(["query", "target"])["bits"].to_dict()
    train_by_host = dict(list(train_pairs.groupby(host_col)))

    scores = np.full(len(test_pairs), prevalence, dtype=np.float32)

    for i, row in enumerate(test_pairs.itertuples(index=False)):
        host = getattr(row, host_col)
        query = getattr(row, id_col)

        sub = train_by_host.get(host)
        if sub is None or sub.empty:
            continue

        best_bits = -np.inf
        best_label: float | None = None
        for _, tr in sub.iterrows():
            target = tr[id_col]
            if query == target:
                continue
            bits = sim_lookup.get((query, target))
            if bits is None:
                bits = sim_lookup.get((target, query))
            if bits is None:
                continue
            if bits > best_bits:
                best_bits = bits
                best_label = float(tr[label_col])
        if best_label is not None:
            scores[i] = best_label
    return scores
