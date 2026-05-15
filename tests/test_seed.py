"""Tests for src.utils.seed."""
from __future__ import annotations

import random

import numpy as np
import pytest

from src.utils.seed import set_global_seed


def test_set_global_seed_is_deterministic() -> None:
    set_global_seed(42)
    a_random = random.random()
    a_numpy = np.random.rand()

    set_global_seed(42)
    b_random = random.random()
    b_numpy = np.random.rand()

    assert a_random == b_random
    assert a_numpy == b_numpy


def test_set_global_seed_rejects_negative() -> None:
    with pytest.raises(ValueError):
        set_global_seed(-1)
