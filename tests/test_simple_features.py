"""Tests for the simple-feature module."""
from __future__ import annotations

import math

import numpy as np

from src.features.simple_features import (
    AMINO_ACIDS,
    amino_acid_composition,
    dipeptide_composition,
    protparam_features,
    summarise_sequence,
)


def test_amino_acid_composition_sums_to_one() -> None:
    seq = "ACDEFGHIKLMNPQRSTVWY" * 3
    comp = amino_acid_composition(seq)
    assert comp.shape == (20,)
    assert math.isclose(comp.sum(), 1.0, rel_tol=1e-5)


def test_amino_acid_composition_handles_empty() -> None:
    assert np.allclose(amino_acid_composition(""), np.zeros(20))


def test_amino_acid_composition_ignores_non_standard() -> None:
    seq = "AAAXXBB"
    comp = amino_acid_composition(seq)
    # After cleaning we expect AAA (3x A) -> 100 % A.
    assert comp[AMINO_ACIDS.index("A")] == 1.0


def test_dipeptide_composition_length_and_sum() -> None:
    seq = "AAAA"
    dipep = dipeptide_composition(seq)
    assert dipep.shape == (400,)
    assert math.isclose(dipep.sum(), 1.0, rel_tol=1e-5)


def test_protparam_features_shape() -> None:
    seq = "ACDEFGHIKLMNPQRSTVWY"
    feats = protparam_features(seq)
    assert feats.shape == (5,)


def test_summarise_sequence_dimensionality() -> None:
    seq = "ACDEFGHIKLMNPQRSTVWY"
    out = summarise_sequence(seq)
    assert out.shape == (425,)
