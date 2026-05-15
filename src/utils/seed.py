# MIT License. See LICENSE in repository root.
"""Deterministic seeding utilities.

Every training / evaluation script calls :func:`set_global_seed` at startup
to make randomness reproducible across `random`, ``numpy``, ``torch`` (CPU /
MPS / CUDA).
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed all the RNGs we use.

    Parameters
    ----------
    seed:
        Non-negative integer seed.

    Notes
    -----
    We also set ``PYTHONHASHSEED`` so that dict ordering of set literals etc.
    is deterministic when the script is launched fresh.  ``torch`` is imported
    lazily so that this module can be used in environments that lack torch.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch is optional for some scripts (e.g. simple feature baselines)
        pass
