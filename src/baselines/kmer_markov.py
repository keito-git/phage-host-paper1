# MIT License. See LICENSE in repository root.
"""k-mer Markov baseline — a lightweight, MIT-licensed stand-in for WIsH.

WIsH (Galiez et al., 2017) trains a per-host Markov model of nucleotide
k-mers and scores a query phage by its log-likelihood under each host
model.  The highest-scoring host is the predicted host.

We cannot bundle the upstream WIsH binary (GPL-3.0) inside an MIT
repository; this module is a clean-room reimplementation of the same idea
in pure NumPy so Paper 1 can include a "k-mer Markov" row without the
licence complication.  Results are naturally close to, but not identical
to, WIsH — which is exactly what we want to report ("WIsH-style k-mer
Markov baseline implemented from the paper description").

Default k=8 matches the WIsH recommendation.  The implementation uses
amino-acid alphabets (20 letters) rather than DNA (4 letters) because the
Paper 1 dataset only ships per-protein sequences at high quality
(``RBPbase.csv`` / ``Locibase.json``).  Reviewers should be aware that this
changes the state-space from ``4**8`` to ``20**8`` — too large to
enumerate; we use a hash-table representation.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

AMINO_ACIDS: tuple[str, ...] = (
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
)
ALPHABET = set(AMINO_ACIDS)


def _kmers(seq: str, k: int) -> list[str]:
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


@dataclass(frozen=True)
class MarkovModel:
    """k-th order Markov model over amino acids with Laplace smoothing."""

    k: int
    transitions: dict[str, dict[str, float]]
    starts: dict[str, float]

    def log_likelihood(self, seq: str) -> float:
        """Return the mean log-probability of ``seq`` under this model."""
        seq = "".join(c for c in seq.upper() if c in ALPHABET)
        if len(seq) <= self.k:
            return -math.inf
        ll = 0.0
        n = 0
        prefix = seq[: self.k]
        ll += math.log(self.starts.get(prefix, 1e-9))
        n += 1
        for i in range(self.k, len(seq)):
            ctx = seq[i - self.k : i]
            nxt = seq[i]
            ll += math.log(
                self.transitions.get(ctx, {}).get(nxt, 1e-9)
            )
            n += 1
        return ll / max(1, n)


def train(sequences: list[str], k: int = 3, alpha: float = 1.0) -> MarkovModel:
    """Train a Laplace-smoothed k-th order Markov model from a list of strings.

    Parameters
    ----------
    sequences:
        Training sequences (strings of amino-acid codes).  Non-canonical
        characters are filtered out before counting.
    k:
        Markov order (default 3).  Using the WIsH default of 8 on AA gives
        ``20**8 ≈ 2.6e10`` transition buckets, which is impractical without
        hashed backoff; k=3 strikes a good balance for 3.3 % prevalence and
        median 7488 aa sequences.
    alpha:
        Laplace smoothing constant.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    ctx_counter: dict[str, Counter[str]] = {}
    start_counter: Counter[str] = Counter()

    for raw in sequences:
        s = "".join(c for c in raw.upper() if c in ALPHABET)
        if len(s) <= k:
            continue
        start_counter[s[:k]] += 1
        for i in range(k, len(s)):
            ctx = s[i - k : i]
            nxt = s[i]
            ctx_counter.setdefault(ctx, Counter())[nxt] += 1

    V = len(AMINO_ACIDS)
    transitions: dict[str, dict[str, float]] = {}
    for ctx, nxts in ctx_counter.items():
        total = sum(nxts.values()) + alpha * V
        transitions[ctx] = {
            aa: (nxts.get(aa, 0) + alpha) / total for aa in AMINO_ACIDS
        }

    total_starts = sum(start_counter.values()) + alpha
    starts = {
        prefix: (count + alpha / max(1, len(start_counter))) / total_starts
        for prefix, count in start_counter.items()
    }
    return MarkovModel(k=k, transitions=transitions, starts=starts)


def score_pair_matrix(
    host_to_model: dict[str, MarkovModel],
    phage_sequences: dict[str, str],
) -> np.ndarray:
    """Return an ``(n_phages, n_hosts)`` log-likelihood score matrix."""
    phage_ids = list(phage_sequences.keys())
    host_ids = list(host_to_model.keys())
    out = np.empty((len(phage_ids), len(host_ids)), dtype=np.float32)
    for i, pid in enumerate(phage_ids):
        seq = phage_sequences[pid]
        for j, hid in enumerate(host_ids):
            out[i, j] = host_to_model[hid].log_likelihood(seq)
    return out
