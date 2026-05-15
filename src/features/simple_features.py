# MIT License. See LICENSE in repository root.
"""Classical protein-sequence features (Experiment 6 sanity baseline).

These features avoid any pretrained model and therefore provide a lower bound
on what a downstream classifier can do without learned embeddings.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AMINO_ACIDS: tuple[str, ...] = (
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
)


def _clean_sequence(seq: str) -> str:
    """Keep only the 20 canonical amino-acid codes.

    ``ProteinAnalysis`` raises on anything outside this alphabet; the raw
    dataset occasionally contains ``X`` / ``*``.
    """
    allowed = set(AMINO_ACIDS)
    return "".join(c for c in seq.upper() if c in allowed)


def amino_acid_composition(seq: str) -> np.ndarray:
    """Return the 20-dim fractional composition of the input sequence."""
    seq = _clean_sequence(seq)
    if not seq:
        return np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    counts = Counter(seq)
    total = float(len(seq))
    return np.array([counts.get(aa, 0) / total for aa in AMINO_ACIDS], dtype=np.float32)


def protparam_features(seq: str) -> np.ndarray:
    """Return a 5-dim physical / chemical descriptor vector.

    Components:
        0. molecular weight (divided by 1e4 to keep magnitudes reasonable)
        1. isoelectric point
        2. aromaticity
        3. instability index
        4. GRAVY (grand average of hydropathy)
    """
    seq = _clean_sequence(seq)
    if not seq:
        return np.zeros(5, dtype=np.float32)
    analysis = ProteinAnalysis(seq)
    return np.array(
        [
            analysis.molecular_weight() / 1e4,
            analysis.isoelectric_point(),
            analysis.aromaticity(),
            analysis.instability_index(),
            analysis.gravy(),
        ],
        dtype=np.float32,
    )


def dipeptide_composition(seq: str) -> np.ndarray:
    """Return the 400-dim relative-frequency dipeptide vector."""
    seq = _clean_sequence(seq)
    dim = len(AMINO_ACIDS) * len(AMINO_ACIDS)
    if len(seq) < 2:
        return np.zeros(dim, dtype=np.float32)
    index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    counts = np.zeros(dim, dtype=np.float32)
    total = float(len(seq) - 1)
    for a, b in zip(seq[:-1], seq[1:], strict=True):
        counts[index[a] * len(AMINO_ACIDS) + index[b]] += 1.0
    return counts / total


def summarise_sequence(seq: str) -> np.ndarray:
    """Concatenate AAC (20) + dipeptide (400) + protparam (5) = 425 features."""
    return np.concatenate(
        [
            amino_acid_composition(seq),
            dipeptide_composition(seq),
            protparam_features(seq),
        ]
    )


def summarise_frame(seqs: pd.Series) -> np.ndarray:
    """Apply :func:`summarise_sequence` row-by-row.

    Returns a 2D array of shape ``(len(seqs), 425)``.
    """
    return np.vstack([summarise_sequence(s) for s in seqs])
