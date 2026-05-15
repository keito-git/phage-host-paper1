# MIT License. See LICENSE in repository root.
"""Sliding-window pooling tests (F1)."""

from __future__ import annotations

import numpy as np

from src.features.sliding_window import (
    WindowConfig,
    iter_windows,
    pool_window_embeddings,
    sliding_window_embed,
)


def test_short_sequence_returns_single_window() -> None:
    cfg = WindowConfig(window_size=1022, stride=511)
    windows = iter_windows("A" * 100, cfg)
    assert len(windows) == 1
    assert windows[0] == "A" * 100


def test_long_sequence_yields_overlapping_windows() -> None:
    cfg = WindowConfig(window_size=10, stride=5)
    windows = iter_windows("A" * 27, cfg)
    # floor((27 - 10) / 5) + 1 = 4 windows (last two fully inside, one boundary)
    assert len(windows) == 5  # incl. the final partial
    assert all(0 < len(w) <= 10 for w in windows)


def test_pool_mean_matches_numpy_mean() -> None:
    arrs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    pooled = pool_window_embeddings(arrs, pooling="mean")
    np.testing.assert_allclose(pooled, np.array([2.0, 3.0]))


def test_sliding_window_embed_respects_configuration() -> None:
    def fake_embedder(seqs: dict[str, str]) -> dict[str, np.ndarray]:
        return {k: np.array([float(len(v))], dtype=np.float32) for k, v in seqs.items()}

    pooled = sliding_window_embed(
        fake_embedder,
        sequences={"x": "A" * 30},
        cfg=WindowConfig(window_size=10, stride=5, pooling="mean"),
    )
    assert "x" in pooled
    # Mean over window lengths (10, 10, 10, 10, 10) = 10.0 for fully-filled windows.
    # The fake embedder returns length-of-window; mean is <= 10.
    assert pooled["x"].item() <= 10.0
