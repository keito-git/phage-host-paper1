# MIT License. See LICENSE in repository root.
"""DNA-level k-mer Markov baseline (WIsH-style), supplementary.

Purpose
-------
The existing ``src/baselines/kmer_markov.py`` operates on *amino-acid*
sequences at k=3 (20**3 = 8000 contexts).  WIsH (Galiez et al., 2017)
operates on *DNA* sequences at k=6 or 8 (4**6 = 4096, 4**8 = 65536
contexts).  This module adds the DNA variant so Paper 1 can legitimately
claim to have tested a "WIsH-style k-mer Markov" baseline, strengthening
the "k-mer-based method category" row in the main table.

Design decisions
----------------
- Alphabet: {A, C, G, T}, N dropped.
- Order k: 6 (default, per the WIsH paper body).  k=8 also available.
- Smoothing: Laplace (add-1), as in the AA variant.
- Only phage-side DNA is available in PhageHostLearn (``RBPbase.csv``
  column ``dna_sequence``).  Host K-locus is shipped as protein (no
  DNA).  We therefore follow a *pragmatic WIsH analog*:
    * Train a per-host Markov model by concatenating the DNA of all
      RBPs that positively interact with that host in TRAIN data
      (no host DNA is required — the phage DNA is used as a surrogate
      host-specific signal, matching WIsH's one-host-one-model idea
      when read in reverse).
    * Score a candidate (phage, host) pair by the mean log-likelihood
      of the phage's concatenated RBP DNA under the host's model.
- This is a clean-room MIT implementation and is *not* identical to
  WIsH; we label it clearly as such in reports to avoid over-claiming.

Clean-room statement
--------------------
Implemented from the WIsH paper description only.  Upstream WIsH code
was never read while writing this file.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

DNA_ALPHABET: tuple[str, ...] = ("A", "C", "G", "T")
_DNA_SET = set(DNA_ALPHABET)


def _clean(seq: str) -> str:
    return "".join(c for c in seq.upper() if c in _DNA_SET)


@dataclass(frozen=True)
class DNAMarkov:
    """k-th order Markov model over DNA 4-letter alphabet with Laplace smoothing."""

    k: int
    transitions: dict[str, dict[str, float]]
    starts: dict[str, float]

    def mean_log_likelihood(self, seq: str) -> float:
        """Return mean log-probability of ``seq`` under this model.

        Uses Laplace-style backoff for unseen contexts (uniform next-symbol
        distribution) so rare-but-valid contexts never collapse the score
        to ``-inf``.
        """
        s = _clean(seq)
        if len(s) <= self.k:
            return -math.inf
        ll = 0.0
        n = 0
        prefix = s[: self.k]
        ll += math.log(self.starts.get(prefix, 1.0 / (4**self.k)))
        n += 1
        uniform = 1.0 / 4
        for i in range(self.k, len(s)):
            ctx = s[i - self.k : i]
            nxt = s[i]
            row = self.transitions.get(ctx)
            p = row.get(nxt, uniform) if row is not None else uniform
            ll += math.log(p)
            n += 1
        return ll / max(1, n)


def train(sequences: list[str], k: int = 6, alpha: float = 1.0) -> DNAMarkov:
    """Train a Laplace-smoothed k-th order DNA Markov model.

    Parameters
    ----------
    sequences:
        DNA strings (A/C/G/T; other letters are dropped).
    k:
        Markov order.  Default 6, matching WIsH paper body.
    alpha:
        Laplace smoothing constant.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    ctx_counter: dict[str, Counter[str]] = {}
    start_counter: Counter[str] = Counter()

    for raw in sequences:
        s = _clean(raw)
        if len(s) <= k:
            continue
        start_counter[s[:k]] += 1
        for i in range(k, len(s)):
            ctx = s[i - k : i]
            nxt = s[i]
            ctx_counter.setdefault(ctx, Counter())[nxt] += 1

    V = 4  # DNA alphabet size
    transitions: dict[str, dict[str, float]] = {}
    for ctx, nxts in ctx_counter.items():
        total = sum(nxts.values()) + alpha * V
        transitions[ctx] = {
            b: (nxts.get(b, 0) + alpha) / total for b in DNA_ALPHABET
        }

    total_starts = sum(start_counter.values()) + alpha * max(1, len(start_counter))
    starts = {
        prefix: (count + alpha) / total_starts
        for prefix, count in start_counter.items()
    }
    return DNAMarkov(k=k, transitions=transitions, starts=starts)


def score_pair_matrix(
    host_to_model: dict[str, DNAMarkov],
    phage_sequences: dict[str, str],
) -> np.ndarray:
    """Return an ``(n_phages, n_hosts)`` mean-log-likelihood matrix."""
    phage_ids = list(phage_sequences.keys())
    host_ids = list(host_to_model.keys())
    out = np.empty((len(phage_ids), len(host_ids)), dtype=np.float32)
    for i, pid in enumerate(phage_ids):
        seq = phage_sequences[pid]
        for j, hid in enumerate(host_ids):
            out[i, j] = host_to_model[hid].mean_log_likelihood(seq)
    return out
