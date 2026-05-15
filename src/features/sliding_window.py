# MIT License. See LICENSE in repository root.
"""Sliding-window pooled ESM-2 embeddings for long K-locus concatenations (F1).

Motivation
----------
The ESM-2 position embedding is capped at 1022 tokens.  For this project the
K-locus concatenation has median length 7488 aa and p95 8688 aa (E4 EDA),
which means truncation wipes out a non-trivial share of the sequence.

A simple and popular fix in protein-LM work is to slide a fixed-size window
over the sequence, embed each window independently, and pool the per-window
CLS-or-mean vectors.  This module provides that helper on top of the
existing :class:`src.features.esm_embedding.ESMEmbedder` cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowConfig:
    """Configuration for sliding-window pooling."""

    window_size: int = 1022
    stride: int = 511  # 50% overlap by default
    pooling: str = "mean"  # "mean" or "max"

    def validate(self) -> None:
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError("window_size and stride must be positive")
        if self.pooling not in ("mean", "max"):
            raise ValueError(f"unsupported pooling: {self.pooling}")


def iter_windows(seq: str, cfg: WindowConfig) -> list[str]:
    """Slice ``seq`` into overlapping windows.

    Short sequences (``len(seq) <= window_size``) return a single window
    equal to the input sequence so callers can treat short and long cases
    uniformly.
    """
    cfg.validate()
    n = len(seq)
    if n <= cfg.window_size:
        return [seq]
    windows: list[str] = []
    start = 0
    while start < n:
        end = min(start + cfg.window_size, n)
        windows.append(seq[start:end])
        if end == n:
            break
        start += cfg.stride
    return windows


def pool_window_embeddings(
    windows: list[np.ndarray],
    pooling: str = "mean",
) -> np.ndarray:
    """Pool a list of per-window embeddings into a single vector.

    Parameters
    ----------
    windows:
        List of per-window ``(D,)`` arrays.
    pooling:
        ``"mean"`` (the default) or ``"max"``.
    """
    if not windows:
        raise ValueError("windows must be non-empty")
    stack = np.stack(windows, axis=0)
    if pooling == "mean":
        return stack.mean(axis=0).astype(np.float32)
    if pooling == "max":
        return stack.max(axis=0).astype(np.float32)
    raise ValueError(f"unsupported pooling: {pooling}")


def sliding_window_embed(
    embedder_embed_many: object,
    sequences: dict[str, str],
    cfg: WindowConfig | None = None,
) -> dict[str, np.ndarray]:
    """Mean-pool sliding-window embeddings over long sequences.

    ``embedder_embed_many`` is a callable (typically
    :meth:`ESMEmbedder.embed_many`) that takes a ``dict[str, str]`` of
    ``id -> sequence`` and returns ``dict[str, np.ndarray]``.  This indirection
    keeps the sliding-window code itself torch-free so it can be unit-tested
    without downloading ESM-2.

    Returns
    -------
    Mapping from the original ``id`` to a ``(D,)`` pooled vector.
    """
    cfg = cfg or WindowConfig()
    cfg.validate()

    # Build a flat window table so the embedder sees a single batched call.
    flat_seqs: dict[str, str] = {}
    window_index: dict[str, list[str]] = {}
    for name, seq in sequences.items():
        windows = iter_windows(seq, cfg)
        window_index[name] = []
        for w_idx, w_seq in enumerate(windows):
            key = f"{name}__w{w_idx:03d}"
            flat_seqs[key] = w_seq
            window_index[name].append(key)

    per_window = embedder_embed_many(flat_seqs)  # type: ignore[operator]

    pooled: dict[str, np.ndarray] = {}
    for name, keys in window_index.items():
        vecs = [per_window[k] for k in keys if k in per_window]
        if not vecs:
            continue
        pooled[name] = pool_window_embeddings(vecs, pooling=cfg.pooling)
    return pooled
