# MIT License. See LICENSE in repository root.
"""Phage-level leak-free cluster-aware split (F0).

The upstream E5 implementation clusters each phage's *first* RBP sequence
(see :mod:`src.data.split`).  That is already an entity-level split, but it
silently ignores the remaining RBPs of multi-RBP phages: two different
phages that share a secondary RBP can still land on the same side of
train/test even if their primary RBPs are dissimilar.

This module tightens the guarantee: we union-cluster every RBP sequence a
phage carries, then assign *phages* (not individual sequences) to
train/val/test such that any two phages sharing a RBP cluster end up in the
same partition.  Specifically:

1. Run MMseqs2 over all RBPs, yielding RBP -> cluster mapping.
2. Build a graph where phages are nodes and edges connect phages whose RBPs
   fall in the same cluster.
3. Take connected components of that graph as "phage groups".
4. Assign whole groups to partitions.

This guarantees zero RBP-cluster overlap between partitions at the phage
granularity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split import SplitResult, mmseqs_cluster


@dataclass(frozen=True)
class PhageSplitReport:
    """Diagnostic summary of the phage-level split."""

    n_rbps: int
    n_rbp_clusters: int
    n_phages: int
    n_phage_components: int
    largest_component: int
    median_component: float


def _connected_components(edges: dict[str, set[str]]) -> list[set[str]]:
    """Return connected components over the undirected adjacency map."""
    seen: set[str] = set()
    comps: list[set[str]] = []
    for node in edges:
        if node in seen:
            continue
        stack = [node]
        comp: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            stack.extend(n for n in edges.get(cur, ()) if n not in comp)
        seen |= comp
        comps.append(comp)
    return comps


def phage_level_split(
    interactions: pd.DataFrame,
    rbps: pd.DataFrame,
    identity: float = 0.5,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    workdir: Path | None = None,
) -> tuple[SplitResult, PhageSplitReport]:
    """Cluster every RBP, then split *phages* respecting component closure.

    Parameters
    ----------
    interactions:
        Tidy long-form table with columns ``host_id``, ``phage_id``, ``label``.
    rbps:
        RBP table (post :func:`src.data.phlearn.load_rbps`) with columns
        ``phage_id``, ``protein_id``, ``sequence``.
    identity:
        MMseqs2 ``--min-seq-id`` threshold.
    val_size, test_size:
        Target fractions over *connected components*.
    seed:
        Shuffle seed controlling component-to-partition assignment.
    workdir:
        Optional override for MMseqs2 scratch directory.
    """
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    # ---- 1. cluster every RBP sequence ------------------------------------
    rbps = rbps.copy().reset_index(drop=True)
    rbps["rbp_key"] = rbps["phage_id"].astype(str) + "::" + rbps["protein_id"].astype(str)
    seq_map = dict(zip(rbps["rbp_key"], rbps["sequence"], strict=True))
    clusters = mmseqs_cluster(seq_map, identity=identity, workdir=workdir)
    cluster_by_rbp = dict(zip(clusters["sequence_id"], clusters["cluster_id"], strict=True))

    # ---- 2. phage -> set of cluster ids -----------------------------------
    phage_clusters: dict[str, set[str]] = {}
    cluster_phages: dict[str, set[str]] = {}
    for _, row in rbps.iterrows():
        phage = str(row["phage_id"])
        cid = cluster_by_rbp.get(row["rbp_key"])
        if cid is None:
            continue
        phage_clusters.setdefault(phage, set()).add(cid)
        cluster_phages.setdefault(cid, set()).add(phage)

    # ---- 3. phage adjacency via shared clusters ---------------------------
    adjacency: dict[str, set[str]] = {p: set() for p in phage_clusters}
    for peers in cluster_phages.values():
        peers_l = list(peers)
        for i in range(len(peers_l)):
            for j in range(i + 1, len(peers_l)):
                adjacency[peers_l[i]].add(peers_l[j])
                adjacency[peers_l[j]].add(peers_l[i])

    components = _connected_components(adjacency)
    phage_to_component: dict[str, int] = {}
    for comp_id, members in enumerate(components):
        for phage in members:
            phage_to_component[phage] = comp_id

    # ---- 4. partition components ------------------------------------------
    rng = np.random.default_rng(seed)
    comp_ids = np.asarray(list(range(len(components))), dtype=int)
    rng.shuffle(comp_ids)
    n_comp = len(comp_ids)
    n_test = max(1, int(round(n_comp * test_size)))
    n_val = max(1, int(round(n_comp * val_size)))
    test_comps = set(comp_ids[:n_test].tolist())
    val_comps = set(comp_ids[n_test : n_test + n_val].tolist())

    def _split_tag(p: str) -> str:
        comp = phage_to_component.get(p)
        if comp is None:
            return "train"  # fallback; should not happen once all phages are covered
        if comp in test_comps:
            return "test"
        if comp in val_comps:
            return "val"
        return "train"

    # ---- 5. assemble DataFrames -------------------------------------------
    ix = interactions.copy()
    ix["split"] = ix["phage_id"].astype(str).map(_split_tag)

    # Phages that exist in interactions but not in rbps (missing RBP) are
    # placed in train so that they are not silently dropped; E1 / E2 already
    # skip such rows inside their embedding matrix builders.
    ix["split"] = ix["split"].fillna("train")

    cluster_report = pd.DataFrame(
        [
            {"phage_id": p, "cluster_id": f"comp{phage_to_component[p]}"}
            for p in phage_clusters
        ]
    )

    train = ix[ix.split == "train"].drop(columns=["split"]).reset_index(drop=True)
    val = ix[ix.split == "val"].drop(columns=["split"]).reset_index(drop=True)
    test = ix[ix.split == "test"].drop(columns=["split"]).reset_index(drop=True)

    split = SplitResult(train=train, val=val, test=test, clusters=cluster_report)
    comp_sizes = np.asarray([len(c) for c in components], dtype=int)
    report = PhageSplitReport(
        n_rbps=len(rbps),
        n_rbp_clusters=clusters["cluster_id"].nunique(),
        n_phages=len(phage_clusters),
        n_phage_components=len(components),
        largest_component=int(comp_sizes.max() if comp_sizes.size else 0),
        median_component=float(np.median(comp_sizes) if comp_sizes.size else 0.0),
    )
    return split, report
