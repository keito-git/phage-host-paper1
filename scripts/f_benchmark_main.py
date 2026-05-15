# MIT License. See LICENSE in repository root.
"""Unified Phase 2 benchmark driver.

Runs every available method against every split kind × every seed and
emits ``predictions_*.parquet`` under ``data/processed/predictions/``.  The
file is the ground truth for F10 aggregation, F2 bootstrap CI, and F3 ECE.

Methods included
----------------
* ``simple_logreg`` / ``simple_xgb`` — E6 classical features (AAC + dipep + ProtParam).
* ``esm650_concat_logreg`` / ``esm650_concat_xgb`` — E1 upstream Zenodo embeddings.
* ``homology_1nn`` — E3 MMseqs2 nearest-neighbour.
* ``kmer_markov`` — MIT-licensed WIsH-style reimplementation.

Split kinds
-----------
* ``rbp_cluster`` — the E5 split (``split_id{id}_seed{seed}.parquet``).
* ``phage_component`` — the F0 phage-level split (``phage_split_id{id}_seed{seed}.parquet``).

Per-row output
--------------
``predictions_{method}_{split_kind}_seed{seed}.parquet`` with columns
``[method, split_kind, seed, host_id, phage_id, label, score]``.
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.homology import mmseqs_all_vs_all, predict_by_nearest_neighbour
from src.baselines.kmer_markov import train
from src.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs
from src.data.phlearn import flatten_loci, load_all
from src.features.simple_features import summarise_sequence
from src.models.classifiers import make_logistic, make_xgboost
from src.utils.seed import set_global_seed

warnings.filterwarnings("ignore", category=UserWarning)

PREDICTIONS_DIR = PROCESSED_DIR / "predictions"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_split(split_kind: str, identity: float, seed: int) -> pd.DataFrame | None:
    if split_kind == "rbp_cluster":
        path = PROCESSED_DIR / "splits" / f"split_id{identity}_seed{seed}.parquet"
    elif split_kind == "phage_component":
        path = PROCESSED_DIR / "splits" / f"phage_split_id{identity}_seed{seed}.parquet"
    else:
        raise ValueError(f"unknown split kind: {split_kind}")
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df


def _load_esm_phlearn() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rbp_df = pd.read_csv(RAW_DIR / "esm2_embeddings_rbp.csv")
    feature_cols = [c for c in rbp_df.columns if c.isdigit()]
    phage_vec = {
        pid: sub[feature_cols].mean(axis=0).to_numpy(dtype=np.float32)
        for pid, sub in rbp_df.groupby("phage_ID")
    }
    loc_df = pd.read_csv(RAW_DIR / "esm2_embeddings_loci.csv")
    loc_cols = [c for c in loc_df.columns if c.isdigit()]
    host_vec = {
        row["accession"]: np.asarray([row[c] for c in loc_cols], dtype=np.float32)
        for _, row in loc_df.iterrows()
    }
    return phage_vec, host_vec


def _save_preds(
    method: str,
    split_kind: str,
    seed: int,
    test_df: pd.DataFrame,
    scores: np.ndarray,
) -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = test_df[["host_id", "phage_id", "label"]].copy()
    out["method"] = method
    out["split_kind"] = split_kind
    out["seed"] = seed
    out["score"] = scores.astype(np.float32)
    path = PREDICTIONS_DIR / f"predictions_{method}_{split_kind}_seed{seed}.parquet"
    out.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# per-method runners
# ---------------------------------------------------------------------------

def run_simple_features(
    split_df: pd.DataFrame,
    rbp_map: dict[str, str],
    loci_concat: dict[str, str],
    seed: int,
) -> dict[str, np.ndarray]:
    """Return ``{model_name -> scores}`` for logreg and XGBoost on classical feats."""
    set_global_seed(seed)

    def build(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        rows_X: list[np.ndarray] = []
        rows_y: list[int] = []
        for row in df.itertuples(index=False):
            rbp_seq = rbp_map.get(row.phage_id)
            host_seq = loci_concat.get(row.host_id)
            if rbp_seq is None or host_seq is None:
                rows_X.append(np.zeros(425 + 425, dtype=np.float32))
                rows_y.append(int(row.label))
                continue
            vec = np.concatenate(
                [summarise_sequence(rbp_seq), summarise_sequence(host_seq)]
            )
            rows_X.append(vec.astype(np.float32))
            rows_y.append(int(row.label))
        return np.vstack(rows_X), np.asarray(rows_y, dtype=np.int64)

    train_df = split_df[split_df.split == "train"]
    test_df = split_df[split_df.split == "test"]

    X_tr, y_tr = build(train_df)
    X_te, _ = build(test_df)

    scale = max(1.0, (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))

    out: dict[str, np.ndarray] = {}
    for name, clf in [
        ("simple_logreg", make_logistic(seed)),
        ("simple_xgb", make_xgboost(seed, scale_pos_weight=scale)),
    ]:
        clf.fit(X_tr, y_tr)
        out[name] = clf.predict_proba(X_te)[:, 1]
    return out


def run_esm_concat(
    split_df: pd.DataFrame,
    phage_vec: dict[str, np.ndarray],
    host_vec: dict[str, np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    set_global_seed(seed)
    train_df = split_df[split_df.split == "train"].reset_index(drop=True)
    test_df = split_df[split_df.split == "test"].reset_index(drop=True)

    def build(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        rows_X: list[np.ndarray] = []
        rows_y: list[int] = []
        for row in df.itertuples(index=False):
            pv = phage_vec.get(row.phage_id)
            hv = host_vec.get(row.host_id)
            if pv is None or hv is None:
                # zero-vector fallback so we keep alignment with the split
                d_p = next(iter(phage_vec.values())).shape[0]
                d_h = next(iter(host_vec.values())).shape[0]
                rows_X.append(np.zeros(d_p + d_h, dtype=np.float32))
            else:
                rows_X.append(np.concatenate([pv, hv]))
            rows_y.append(int(row.label))
        return np.vstack(rows_X), np.asarray(rows_y, dtype=np.int64)

    X_tr, y_tr = build(train_df)
    X_te, _ = build(test_df)
    scale = max(1.0, (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))

    out: dict[str, np.ndarray] = {}
    for name, clf in [
        ("esm650_logreg", make_logistic(seed)),
        ("esm650_xgb", make_xgboost(seed, scale_pos_weight=scale)),
    ]:
        clf.fit(X_tr, y_tr)
        out[name] = clf.predict_proba(X_te)[:, 1]
    return out


def run_homology_1nn(
    split_df: pd.DataFrame,
    rbp_map: dict[str, str],
    seed: int,
) -> np.ndarray:
    from src.config import CACHE_DIR

    set_global_seed(seed)
    train_df = split_df[split_df.split == "train"]
    test_df = split_df[split_df.split == "test"]
    needed_ids = set(train_df.phage_id) | set(test_df.phage_id)
    seq_subset = {pid: rbp_map[pid] for pid in needed_ids if pid in rbp_map}
    if len(seq_subset) < 2:
        return np.full(len(test_df), train_df.label.mean(), dtype=np.float32)

    workdir = CACHE_DIR / f"mmseqs_search_bench_seed{seed}"
    similarity = mmseqs_all_vs_all(seq_subset, workdir=workdir)
    scores = predict_by_nearest_neighbour(
        train_pairs=train_df,
        test_pairs=test_df,
        similarity=similarity,
        id_col="phage_id",
        host_col="host_id",
        label_col="label",
    )
    return scores


def run_kmer_markov(
    split_df: pd.DataFrame,
    rbp_map: dict[str, str],
    loci_concat: dict[str, str],
    seed: int,
) -> np.ndarray:
    """WIsH-style per-host Markov model on RBP sequences.

    Training: for each host in train with >= 1 positive, fit a Markov model
    on the RBP sequences of phages that lyse it.  Scoring: test pair score
    = log-likelihood of the test phage's RBP under that host's model.
    Hosts absent from train fall back to global prevalence.
    """
    set_global_seed(seed)
    train_df = split_df[split_df.split == "train"]
    test_df = split_df[split_df.split == "test"]

    # Build host -> list[training_rbp_seq]
    host_models: dict[str, object] = {}
    for host, sub in train_df.groupby("host_id"):
        pos = sub[sub.label == 1]
        seqs = [
            rbp_map[pid]
            for pid in pos.phage_id
            if pid in rbp_map and len(rbp_map[pid]) > 20
        ]
        if len(seqs) >= 2:
            host_models[host] = train(seqs, k=3, alpha=1.0)

    prevalence = float(train_df.label.mean())
    scores = np.full(len(test_df), prevalence, dtype=np.float32)
    for i, row in enumerate(test_df.itertuples(index=False)):
        model = host_models.get(row.host_id)
        seq = rbp_map.get(row.phage_id)
        if model is None or seq is None:
            continue
        scores[i] = model.log_likelihood(seq)  # type: ignore[attr-defined]
    return scores


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_all(identity: float, seeds: list[int]) -> dict:
    ensure_dirs()
    tables = load_all()
    rbp_map = dict(zip(tables.rbps["phage_id"], tables.rbps["sequence"], strict=True))
    loci_concat = dict(
        zip(
            flatten_loci(tables.loci)["host_id"],
            flatten_loci(tables.loci)["k_locus_concat"],
            strict=True,
        )
    )
    phage_vec, host_vec = _load_esm_phlearn()

    summary: dict = {"runs": [], "skipped": []}

    for split_kind in ("rbp_cluster", "phage_component"):
        for seed in seeds:
            split_df = _load_split(split_kind, identity, seed)
            if split_df is None:
                summary["skipped"].append(
                    {"split_kind": split_kind, "seed": seed, "reason": "split_missing"}
                )
                continue
            if "split" not in split_df.columns:
                # phage_split has 'split' already; safety net
                summary["skipped"].append(
                    {"split_kind": split_kind, "seed": seed, "reason": "no_split_column"}
                )
                continue

            test_df = split_df[split_df.split == "test"].reset_index(drop=True)
            if test_df.empty or (test_df.label == 1).sum() < 2 or (test_df.label == 0).sum() < 2:
                summary["skipped"].append(
                    {
                        "split_kind": split_kind,
                        "seed": seed,
                        "reason": "test_set_insufficient",
                        "n_test": int(len(test_df)),
                        "n_pos": int((test_df.label == 1).sum()),
                    }
                )
                continue

            # E6 classical
            try:
                simple = run_simple_features(split_df, rbp_map, loci_concat, seed)
                for name, sc in simple.items():
                    _save_preds(name, split_kind, seed, test_df, sc)
                    summary["runs"].append(
                        {"method": name, "split_kind": split_kind, "seed": seed}
                    )
            except Exception as e:  # noqa: BLE001 — we record & continue
                summary["skipped"].append(
                    {"method": "simple_features", "split_kind": split_kind, "seed": seed,
                     "reason": f"exception: {type(e).__name__}: {e}"}
                )

            # E1 ESM-2 650M
            try:
                esm_out = run_esm_concat(split_df, phage_vec, host_vec, seed)
                for name, sc in esm_out.items():
                    _save_preds(name, split_kind, seed, test_df, sc)
                    summary["runs"].append(
                        {"method": name, "split_kind": split_kind, "seed": seed}
                    )
            except Exception as e:  # noqa: BLE001
                summary["skipped"].append(
                    {"method": "esm650", "split_kind": split_kind, "seed": seed,
                     "reason": f"exception: {type(e).__name__}: {e}"}
                )

            # E3 homology
            try:
                sc = run_homology_1nn(split_df, rbp_map, seed)
                _save_preds("homology_1nn", split_kind, seed, test_df, sc)
                summary["runs"].append(
                    {"method": "homology_1nn", "split_kind": split_kind, "seed": seed}
                )
            except Exception as e:  # noqa: BLE001
                summary["skipped"].append(
                    {"method": "homology_1nn", "split_kind": split_kind, "seed": seed,
                     "reason": f"exception: {type(e).__name__}: {e}"}
                )

            # k-mer Markov (WIsH-style)
            try:
                sc = run_kmer_markov(split_df, rbp_map, loci_concat, seed)
                _save_preds("kmer_markov", split_kind, seed, test_df, sc)
                summary["runs"].append(
                    {"method": "kmer_markov", "split_kind": split_kind, "seed": seed}
                )
            except Exception as e:  # noqa: BLE001
                summary["skipped"].append(
                    {"method": "kmer_markov", "split_kind": split_kind, "seed": seed,
                     "reason": f"exception: {type(e).__name__}: {e}"}
                )

    (REPORTS_DIR / "f_benchmark_main.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    out = run_all(args.identity, list(args.seeds))
    print(json.dumps({"n_runs": len(out["runs"]), "n_skipped": len(out["skipped"])}))
