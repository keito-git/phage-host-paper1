# MIT License. See LICENSE in repository root.
"""Tests for the k-mer Markov baseline (WIsH-style)."""

from __future__ import annotations

from src.baselines.kmer_markov import MarkovModel, train


def test_train_produces_valid_model() -> None:
    seqs = ["MACDEFGHI", "MACDEFGHI", "MACEFGHIK"]
    model = train(seqs, k=2)
    assert isinstance(model, MarkovModel)
    # transitions from "MA" must be a valid distribution.
    ma = model.transitions.get("MA")
    assert ma is not None
    assert abs(sum(ma.values()) - 1.0) < 1e-6


def test_log_likelihood_for_in_distribution_beats_noise() -> None:
    seqs = ["MACDEFGHIKLMNPQRSTVWY"] * 20
    model = train(seqs, k=2)
    in_dist = model.log_likelihood("MACDEFGHIKLMN")
    off_dist = model.log_likelihood("YYYYYYYYYYYYY")
    assert in_dist > off_dist
