# MIT License. See LICENSE in repository root.
"""Cluster-aware train/val/test splitting (Experiment 5).

Phage receptor-binding proteins often share very high sequence identity.  A
naive random split will therefore leak information between train and test
partitions and inflate the apparent accuracy.  We cluster the RBP sequences
with ``mmseqs easy-cluster`` at a configurable identity threshold and assign
each *cluster* (not each sequence) to a single partition.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_DIR


@dataclass(frozen=True)
class SplitResult:
    """Container for a single cluster-aware split."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    clusters: pd.DataFrame  # phage_id -> cluster_id


def _write_fasta(records: Iterable[tuple[str, str]], path: Path) -> None:
    with path.open("w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n{seq}\n")


def mmseqs_cluster(
    sequences: dict[str, str],
    identity: float = 0.5,
    coverage: float = 0.8,
    cov_mode: int = 0,
    workdir: Path | None = None,
    mmseqs_bin: str = "mmseqs",
) -> pd.DataFrame:
    """Cluster sequences via ``mmseqs easy-cluster``.

    Parameters
    ----------
    sequences:
        Mapping from unique ID to amino-acid sequence.
    identity:
        Minimum sequence identity inside a cluster (``--min-seq-id``).
    coverage:
        Minimum alignment coverage (``-c``).
    cov_mode:
        MMseqs2 coverage mode; 0 = bidirectional, 1 = target, 2 = query.
    workdir:
        Scratch directory; a temporary one under ``CACHE_DIR`` is created when
        omitted.
    mmseqs_bin:
        Path to the mmseqs executable.

    Returns
    -------
    DataFrame with columns ``[sequence_id, cluster_id]``.
    """
    if shutil.which(mmseqs_bin) is None:
        raise RuntimeError(
            f"mmseqs executable '{mmseqs_bin}' not found in PATH. "
            "Install it with `brew install mmseqs2`."
        )

    workdir = Path(workdir) if workdir else CACHE_DIR / "mmseqs_cluster"
    workdir.mkdir(parents=True, exist_ok=True)

    fasta_path = workdir / "input.fasta"
    _write_fasta(sequences.items(), fasta_path)

    out_prefix = workdir / "clusterRes"
    tmp_dir = workdir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    cmd = [
        mmseqs_bin,
        "easy-cluster",
        str(fasta_path),
        str(out_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(identity),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "-v",
        "1",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    cluster_tsv = out_prefix.with_suffix("").as_posix() + "_cluster.tsv"
    clusters = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["cluster_id", "sequence_id"])
    return clusters[["sequence_id", "cluster_id"]]


def cluster_aware_split(
    pairs: pd.DataFrame,
    sequence_col: str,
    id_col: str,
    identity: float = 0.5,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    workdir: Path | None = None,
) -> SplitResult:
    """Split ``pairs`` on clusters of the ``id_col`` entities.

    The function clusters the unique sequences referenced by ``id_col``
    (e.g. each phage's representative RBP sequence) and then assigns whole
    clusters to train / val / test partitions.  This guarantees zero
    cluster-level leakage between partitions.

    Parameters
    ----------
    pairs:
        Long-form interaction table.
    sequence_col:
        Column containing the amino-acid sequence used for clustering.
    id_col:
        Column identifying the entity to cluster (often ``phage_id``).
    identity:
        Minimum sequence identity (see :func:`mmseqs_cluster`).
    val_size, test_size:
        Target fractions of *clusters* going to val / test.  The remainder
        goes to train.
    seed:
        RNG seed controlling cluster-to-partition assignment.

    Returns
    -------
    :class:`SplitResult` with the three partitions and the cluster mapping.
    """
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    unique = pairs[[id_col, sequence_col]].drop_duplicates().set_index(id_col)
    seq_map = unique[sequence_col].to_dict()
    clusters = mmseqs_cluster(seq_map, identity=identity, workdir=workdir)
    clusters = clusters.rename(columns={"sequence_id": id_col})

    rng = np.random.default_rng(seed)
    cluster_ids = clusters.cluster_id.unique()
    rng.shuffle(cluster_ids)

    n = len(cluster_ids)
    n_test = max(1, int(round(n * test_size)))
    n_val = max(1, int(round(n * val_size)))
    test_clusters = set(cluster_ids[:n_test])
    val_clusters = set(cluster_ids[n_test : n_test + n_val])

    def _split_tag(cid: str) -> str:
        if cid in test_clusters:
            return "test"
        if cid in val_clusters:
            return "val"
        return "train"

    clusters["split"] = clusters["cluster_id"].map(_split_tag)
    merged = pairs.merge(clusters, on=id_col, how="left")

    if merged["split"].isna().any():
        missing = merged[merged["split"].isna()][id_col].unique()
        raise RuntimeError(
            f"{len(missing)} ids from ``pairs`` were not covered by the clustering"
        )

    train = merged[merged.split == "train"].drop(columns=["split"]).reset_index(drop=True)
    val = merged[merged.split == "val"].drop(columns=["split"]).reset_index(drop=True)
    test = merged[merged.split == "test"].drop(columns=["split"]).reset_index(drop=True)
    return SplitResult(train=train, val=val, test=test, clusters=clusters)


def leakage_report(result: SplitResult, id_col: str = "phage_id") -> dict[str, int]:
    """Sanity-check that no ``id_col`` appears in more than one partition."""
    train_ids = set(result.train[id_col])
    val_ids = set(result.val[id_col])
    test_ids = set(result.test[id_col])
    return {
        "train_n_ids": len(train_ids),
        "val_n_ids": len(val_ids),
        "test_n_ids": len(test_ids),
        "train_val_overlap": len(train_ids & val_ids),
        "train_test_overlap": len(train_ids & test_ids),
        "val_test_overlap": len(val_ids & test_ids),
    }
