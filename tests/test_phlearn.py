"""Smoke tests for the PhageHostLearn tidy-table loader.

These tests are skipped if the raw files are missing so the suite stays
functional on CI installs where data is not available.
"""
from __future__ import annotations

import pytest

from src.config import RAW_DIR
from src.data.phlearn import load_all

pytestmark = pytest.mark.skipif(
    not (RAW_DIR / "Locibase.json").exists(),
    reason="PhageHostLearn raw data not present; run e1 download first",
)


def test_load_all_drops_missing_host_rows() -> None:
    tables = load_all(restrict_to_loci=True)
    # All interaction host_ids must be present in loci once filtered.
    inter_hosts = set(tables.interactions.host_id.unique())
    loci_hosts = set(tables.loci.host_id.unique())
    assert inter_hosts.issubset(loci_hosts), inter_hosts - loci_hosts


def test_overview_is_self_consistent() -> None:
    tables = load_all()
    stats = tables.overview()
    assert stats["num_interactions"] == (
        stats["num_positive_pairs"] + stats["num_negative_pairs"]
    )
    assert stats["num_phages"] > 0 and stats["num_hosts"] > 0
