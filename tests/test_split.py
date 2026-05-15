"""Tests for the split module.

The MMseqs2 invocation itself is integration-tested separately; here we only
validate the pure-Python bookkeeping.
"""
from __future__ import annotations

import pandas as pd

from src.data.split import SplitResult, leakage_report


def _make_pairs(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"phage_id": ids, "host_id": ["h"] * len(ids), "label": [1] * len(ids)})


def test_leakage_report_counts_overlap() -> None:
    result = SplitResult(
        train=_make_pairs(["p1", "p2"]),
        val=_make_pairs(["p2", "p3"]),
        test=_make_pairs(["p4"]),
        clusters=pd.DataFrame(columns=["phage_id", "cluster_id", "split"]),
    )
    report = leakage_report(result)
    assert report["train_val_overlap"] == 1
    assert report["train_test_overlap"] == 0
    assert report["val_test_overlap"] == 0
