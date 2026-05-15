# MIT License. See LICENSE in repository root.
"""Integrity tests for the phage-level connected-component split (F0).

These tests address reviewer M1 comment C3-2: ``src/data/phage_split.py``
performs connected-component decomposition over a phage -> RBP-cluster
graph, and any bug in that decomposition or in the partition assignment
could silently re-introduce leakage between train and test.

To keep the suite runnable without the external MMseqs2 binary, we
exercise :func:`src.data.phage_split._connected_components` directly with
small synthetic adjacency maps, plus an end-to-end test that monkey-patches
the MMseqs2 clustering step with a hand-crafted cluster table.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.phage_split import _connected_components, phage_level_split


def test_connected_components_singletons_and_pairs() -> None:
    """Two disconnected islands plus a singleton -> three components."""
    edges: dict[str, set[str]] = {
        "a": {"b"},
        "b": {"a"},
        "c": {"d"},
        "d": {"c"},
        "e": set(),
    }
    components = _connected_components(edges)
    as_sets = sorted([frozenset(c) for c in components], key=sorted)
    expected = sorted(
        [
            frozenset({"a", "b"}),
            frozenset({"c", "d"}),
            frozenset({"e"}),
        ],
        key=sorted,
    )
    assert as_sets == expected


def test_connected_components_transitive_closure() -> None:
    """a - b - c chain must collapse into one component."""
    edges: dict[str, set[str]] = {
        "a": {"b"},
        "b": {"a", "c"},
        "c": {"b"},
    }
    components = _connected_components(edges)
    assert len(components) == 1
    assert components[0] == {"a", "b", "c"}


def test_connected_components_empty_graph() -> None:
    assert _connected_components({}) == []


def test_phage_split_respects_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any two phages sharing an RBP cluster must land in the same partition.

    We monkey-patch :func:`src.data.phage_split.mmseqs_cluster` with a fake
    clustering that maps specific (phage_id, protein_id) pairs to cluster
    ids; this keeps the test hermetic (no MMseqs2 binary required).
    """

    # Phage A and C share cluster C1 via a secondary RBP; phage B and D are
    # isolated singletons.  The connected-component closure therefore has
    # three components: {A, C}, {B}, {D}.
    rbps = pd.DataFrame(
        [
            {"phage_id": "A", "protein_id": "A1", "sequence": "M" * 30},
            {"phage_id": "A", "protein_id": "A2", "sequence": "M" * 30},
            {"phage_id": "B", "protein_id": "B1", "sequence": "M" * 30},
            {"phage_id": "C", "protein_id": "C1", "sequence": "M" * 30},
            {"phage_id": "C", "protein_id": "C2", "sequence": "M" * 30},
            {"phage_id": "D", "protein_id": "D1", "sequence": "M" * 30},
        ]
    )

    cluster_table = pd.DataFrame(
        [
            # A1 / C1 share cluster C1 -> creates the A--C edge
            {"sequence_id": "A::A1", "cluster_id": "C1"},
            {"sequence_id": "C::C1", "cluster_id": "C1"},
            # A2 unique
            {"sequence_id": "A::A2", "cluster_id": "C2"},
            # C2 unique
            {"sequence_id": "C::C2", "cluster_id": "C3"},
            # B and D each get their own singleton cluster
            {"sequence_id": "B::B1", "cluster_id": "C4"},
            {"sequence_id": "D::D1", "cluster_id": "C5"},
        ]
    )

    def fake_mmseqs_cluster(*args: object, **kwargs: object) -> pd.DataFrame:
        return cluster_table

    monkeypatch.setattr("src.data.phage_split.mmseqs_cluster", fake_mmseqs_cluster)

    # Interactions: every phage * every host, labels arbitrary.
    interactions = pd.DataFrame(
        [
            {"phage_id": p, "host_id": h, "label": (idx + j) % 2}
            for idx, p in enumerate(["A", "B", "C", "D"])
            for j, h in enumerate(["H1", "H2", "H3"])
        ]
    )

    split, report = phage_level_split(
        interactions=interactions,
        rbps=rbps,
        identity=0.5,
        val_size=0.25,
        test_size=0.25,
        seed=42,
    )

    # (a) Component closure is respected: A and C always co-locate.
    split_of: dict[str, str] = {}
    for partition_name, frame in (
        ("train", split.train),
        ("val", split.val),
        ("test", split.test),
    ):
        for p in frame["phage_id"].astype(str).unique():
            split_of[p] = partition_name
    assert split_of["A"] == split_of["C"], (
        f"A and C share a cluster but ended up in {split_of['A']} / {split_of['C']}"
    )

    # (b) Zero phage overlap between any two partitions.
    for name_a, name_b in (("train", "val"), ("train", "test"), ("val", "test")):
        phages_a = set(getattr(split, name_a)["phage_id"].astype(str).tolist())
        phages_b = set(getattr(split, name_b)["phage_id"].astype(str).tolist())
        assert phages_a.isdisjoint(phages_b), (
            f"phage overlap between {name_a} and {name_b}: "
            f"{phages_a & phages_b}"
        )

    # (c) Report counts match the synthetic graph.
    assert report.n_rbps == 6
    assert report.n_phages == 4
    assert report.n_phage_components == 3


def test_phage_split_rejects_bad_partition_sizes() -> None:
    """val + test must be < 1.0 (otherwise train would be empty)."""
    rbps = pd.DataFrame(
        [{"phage_id": "A", "protein_id": "A1", "sequence": "M" * 30}]
    )
    interactions = pd.DataFrame(
        [{"phage_id": "A", "host_id": "H1", "label": 0}]
    )
    with pytest.raises(ValueError):
        phage_level_split(
            interactions=interactions,
            rbps=rbps,
            val_size=0.6,
            test_size=0.5,
        )
