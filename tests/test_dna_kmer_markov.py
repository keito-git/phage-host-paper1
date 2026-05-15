# MIT License. See LICENSE in repository root.
"""Tests for the DNA-level k-mer Markov baseline (Phase 2.6-C)."""

from __future__ import annotations

import math

from src.baselines.dna_kmer_markov import DNAMarkov, _clean, train


def test_clean_drops_non_acgt() -> None:
    assert _clean("AcgTxNryACGT") == "ACGTACGT"
    assert _clean("") == ""


def test_train_produces_valid_model() -> None:
    seqs = ["ACGTACGTACGT", "ACGTACGTACGT", "ACGTACGTACGA"]
    model = train(seqs, k=2)
    assert isinstance(model, DNAMarkov)
    # The AC transition distribution should sum to 1.
    ac = model.transitions.get("AC")
    assert ac is not None
    assert abs(sum(ac.values()) - 1.0) < 1e-6


def test_log_likelihood_for_in_distribution_beats_noise() -> None:
    seqs = ["ACGTACGTACGTACGT"] * 20
    model = train(seqs, k=3)
    in_dist = model.mean_log_likelihood("ACGTACGTAC")
    off_dist = model.mean_log_likelihood("TTTTTTTTTT")
    assert in_dist > off_dist


def test_short_sequence_returns_neg_inf() -> None:
    model = train(["ACGTACGTACGT"], k=3)
    # Sequence shorter than k (or equal) must return -inf by the contract.
    assert math.isinf(model.mean_log_likelihood("AC"))


def test_unseen_context_backs_off_to_uniform() -> None:
    model = train(["AAAAAAAA"], k=3)  # trained only on AAA contexts
    ll = model.mean_log_likelihood("GCGCGCGC")
    # Unseen contexts use uniform backoff, so ll should be finite and
    # roughly log(1/4) on average.
    assert math.isfinite(ll)
    assert ll < 0
    # log(1/4) = -1.386; allow some slack for the starts term.
    assert -3.5 < ll < -1.0
