# MIT License. See LICENSE in repository root.
"""DeLong implementation cross-check against pROC reference fixtures.

Addresses reviewer comment C3-1: the Sun & Xu (2014) fast DeLong
implementation in ``src/stats/metrics.py`` had no numerical reference
check.  This test loads a small set of paired prediction arrays plus the
reference ROC-AUCs and p-value summaries that R ``pROC``
(Robin et al. 2011) produces, and asserts our Python implementation
agrees within the tolerance encoded in
``tests/fixtures/delong_proc_ref.json``.

Regeneration recipe
-------------------
The JSON fixture documents how to regenerate its reference numbers from
an R session.  Until pROC can be run in CI, the fixtures encode only
invariants that are robust across reasonable DeLong conventions:
* perfectly-separating paired predictors -> ``delta == 0``, ``p == 1``
* overlap with a single-pair perturbation -> ``delta`` positive, ``p``
  strictly below a conservative upper bound.

These invariants catch the two most common failure modes (sign flips
and off-by-one mid-rank handling) without requiring a live R
installation.  When R is available, replace the ``reference`` block
with the exact pROC numbers and tighten the tolerances.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.stats.metrics import delong_test

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "delong_proc_ref.json"


@pytest.fixture(scope="module")
def reference_fixture() -> dict:
    with FIXTURE_PATH.open() as fh:
        return json.load(fh)


def test_fixture_loads(reference_fixture: dict) -> None:
    assert "cases" in reference_fixture
    assert len(reference_fixture["cases"]) >= 1
    assert "tolerance" in reference_fixture


def test_delong_matches_fixtures(reference_fixture: dict) -> None:
    """Assert delong_test agrees with the pROC-style reference fixtures."""
    tol = float(reference_fixture["tolerance"])
    for case in reference_fixture["cases"]:
        y = np.asarray(case["labels"], dtype=int)
        a = np.asarray(case["scores_a"], dtype=float)
        b = np.asarray(case["scores_b"], dtype=float)
        ref = case["reference"]
        result = delong_test(a, b, y)

        # AUCs always compared to reference.
        assert abs(result.auc_a - ref["auc_a"]) < tol, (
            f"[{case['name']}] auc_a mismatch: got {result.auc_a}, "
            f"expected {ref['auc_a']}"
        )
        assert abs(result.auc_b - ref["auc_b"]) < tol, (
            f"[{case['name']}] auc_b mismatch: got {result.auc_b}, "
            f"expected {ref['auc_b']}"
        )

        # Optional strict-delta / strict-p check.
        if "delta" in ref:
            assert abs(result.delta - ref["delta"]) < tol, (
                f"[{case['name']}] delta mismatch: got {result.delta}, "
                f"expected {ref['delta']}"
            )
        if "p_value" in ref:
            assert abs(result.p_value - ref["p_value"]) < tol, (
                f"[{case['name']}] p mismatch: got {result.p_value}, "
                f"expected {ref['p_value']}"
            )

        # Sign / bound invariants (robust across pROC conventions).
        if ref.get("delta_sign") == "positive":
            assert result.delta > 0, (
                f"[{case['name']}] expected positive delta, got {result.delta}"
            )
        if ref.get("delta_sign") == "negative":
            assert result.delta < 0, (
                f"[{case['name']}] expected negative delta, got {result.delta}"
            )
        if ref.get("z_sign") == "positive":
            assert result.z > 0, (
                f"[{case['name']}] expected positive z, got {result.z}"
            )
        if ref.get("z_sign") == "negative":
            assert result.z < 0, (
                f"[{case['name']}] expected negative z, got {result.z}"
            )
        if "p_max" in ref:
            assert result.p_value < ref["p_max"], (
                f"[{case['name']}] p={result.p_value} exceeded bound "
                f"{ref['p_max']}"
            )


def test_delong_identical_inputs_have_p_one() -> None:
    """Sanity invariant: identical score vectors -> delta=0, p=1."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=80)
    # Ensure at least one of each class.
    y[0], y[1] = 0, 1
    s = rng.standard_normal(80)
    res = delong_test(s, s, y)
    assert res.delta == 0.0
    assert res.p_value == 1.0


def test_delong_symmetry_of_p_value() -> None:
    """Swapping A and B should flip the sign of delta / z but keep p."""
    rng = np.random.default_rng(1)
    y = np.concatenate([np.zeros(40, dtype=int), np.ones(40, dtype=int)])
    a = rng.standard_normal(80)
    b = rng.standard_normal(80) + 0.3 * y  # B is slightly better on positives
    res_ab = delong_test(a, b, y)
    res_ba = delong_test(b, a, y)
    assert abs(res_ab.delta + res_ba.delta) < 1e-9
    assert abs(res_ab.z + res_ba.z) < 1e-9
    assert abs(res_ab.p_value - res_ba.p_value) < 1e-9
