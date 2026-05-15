# MIT License. See LICENSE in repository root.
"""Experiment 2: locally-computed ESM-2 embedding + lightweight classifier.

Contrast with E1: here we *regenerate* the embeddings on this machine with a
small-ish ESM-2 variant (default ``esm2_t12_35M_UR50D``) and run the
classifier on the identical E5 split.  The point is to verify that the
whole pipeline is reproducible end-to-end on a commodity workstation and to
show how much performance we give up vs. the upstream 650M embeddings.
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import MULTI_SEEDS, PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from src.data.phlearn import flatten_loci, load_all, load_loci, pair_with_first_rbp
from src.features.esm_embedding import ESMEmbedder
from src.models.classifiers import evaluate, make_logistic, make_xgboost
from src.utils.seed import set_global_seed


def _load_split(identity: float, seed: int) -> pd.DataFrame:
    path = PROCESSED_DIR / "splits" / f"split_id{identity}_seed{seed}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Split missing: {path}. Run E5 first.")
    return pd.read_parquet(path)


def _sequence_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Return canonical ``phage_id -> RBP seq`` and ``host_id -> K-locus seq``.

    For the K-locus we use the flattened concatenation (``*``-joined) from
    :func:`flatten_loci`; the ESM tokenizer ignores ``*`` so it plays the
    role of a soft separator without polluting the representation.
    """
    tables = load_all()
    pairs = pair_with_first_rbp(tables.interactions, tables.rbps)
    phage_seq = (
        pairs[["phage_id", "rbp_sequence"]]
        .drop_duplicates("phage_id")
        .set_index("phage_id")["rbp_sequence"]
        .to_dict()
    )
    loci_flat = flatten_loci(load_loci())
    host_seq = loci_flat.set_index("host_id")["k_locus_concat"].to_dict()
    return phage_seq, host_seq


def _build_matrix(
    df: pd.DataFrame,
    phage_vec: dict[str, np.ndarray],
    host_vec: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for row in df.itertuples(index=False):
        pv = phage_vec.get(row.phage_id)
        hv = host_vec.get(row.host_id)
        if pv is None or hv is None:
            continue
        rows.append(np.concatenate([pv, hv]))
        labels.append(int(row.label))
    return (
        np.vstack(rows).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def main(
    identity: float,
    seeds: list[int],
    esm_model: str,
    batch_size: int,
    limit_loci_length: int,
    device: str,
) -> None:
    ensure_dirs()

    phage_seq, host_seq = _sequence_maps()

    # Truncate extremely long K-locus concatenations to keep inference on M4
    # predictable (pooled representation uses the first ``limit_loci_length``
    # residues; ESM-2's positional limit is ~1022).
    host_seq = {h: s[:limit_loci_length] for h, s in host_seq.items()}

    # "auto" lets ESMEmbedder pick mps > cuda > cpu.  Users who hit MPS
    # instability (observed occasionally on M4 Max for this exact workload)
    # can override with ``--device cpu`` for a slower but reliable run.
    dev: torch.device | None = None if device == "auto" else torch.device(device)
    embedder = ESMEmbedder(model_name=esm_model, batch_size=batch_size, device=dev)
    phage_vec = embedder.embed_many(phage_seq)
    host_vec = embedder.embed_many(host_seq)

    per_seed: dict[int, dict] = {}
    for s in seeds:
        set_global_seed(s)
        df = _load_split(identity, s)
        train = df[df.split == "train"].reset_index(drop=True)
        test = df[df.split == "test"].reset_index(drop=True)

        X_train, y_train = _build_matrix(train, phage_vec, host_vec)
        X_test, y_test = _build_matrix(test, phage_vec, host_vec)

        scale_pos = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))

        run: dict[str, dict] = {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        }
        for name, clf in [
            ("logreg", make_logistic(s)),
            ("xgboost", make_xgboost(s, scale_pos_weight=scale_pos)),
        ]:
            clf.fit(X_train, y_train)
            scores = clf.predict_proba(X_test)[:, 1]
            m = evaluate(y_test, scores)
            run[name] = {
                "roc_auc": m.roc_auc,
                "pr_auc": m.pr_auc,
                "best_f1": m.best_f1,
            }
        per_seed[s] = run

    aggregated: dict[str, dict[str, float]] = {}
    for model in ("logreg", "xgboost"):
        for metric in ("roc_auc", "pr_auc", "best_f1"):
            vals = [per_seed[s][model][metric] for s in seeds]
            aggregated.setdefault(model, {})[f"{metric}_mean"] = float(statistics.mean(vals))
            aggregated[model][f"{metric}_std"] = float(statistics.pstdev(vals))

    summary = {
        "identity": identity,
        "seeds": list(seeds),
        "esm_model": esm_model,
        "batch_size": batch_size,
        "limit_loci_length": limit_loci_length,
        "phage_vec_dim": int(next(iter(phage_vec.values())).shape[0]),
        "host_vec_dim": int(next(iter(host_vec.values())).shape[0]),
        "per_seed": per_seed,
        "aggregated": aggregated,
    }
    render_report(REPORTS_DIR / "experiment_02_esm_embed.md", summary)
    print(json.dumps(summary, indent=2))


def render_report(path: Path, summary: dict) -> None:
    lines = [
        "# Experiment 2 -- Local ESM-2 embeddings + lightweight classifier",
        "",
        "**Goal.** Reproduce the embedding + classifier pipeline entirely on",
        "this workstation using a small ESM-2 variant, so the whole setup is",
        "runnable on a single M4 Max laptop.",
        "",
        f"- ESM-2 checkpoint: `{summary['esm_model']}`",
        f"- Batch size: `{summary['batch_size']}`",
        f"- Loci length cap: `{summary['limit_loci_length']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- RBP embedding dimensionality: {summary['phage_vec_dim']}",
        f"- K-locus embedding dimensionality: {summary['host_vec_dim']}",
        "",
        "## Aggregated metrics",
        "",
        "| Model | ROC-AUC | PR-AUC | best F1 |",
        "|---|---|---|---|",
    ]
    for model, vals in summary["aggregated"].items():
        lines.append(
            f"| {model} | {vals['roc_auc_mean']:.3f} +/- {vals['roc_auc_std']:.3f} | "
            f"{vals['pr_auc_mean']:.3f} +/- {vals['pr_auc_std']:.3f} | "
            f"{vals['best_f1_mean']:.3f} +/- {vals['best_f1_std']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Compare ROC-AUC / PR-AUC to E1 (upstream 650M embeddings) and E6",
        "(classical features).  A large gap above E6 with only a small gap",
        "below E1 would justify prioritising locally-trained small models in",
        "subsequent, larger experiments.",
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(MULTI_SEEDS))
    parser.add_argument("--esm_model", type=str, default="esm2_t12_35M_UR50D")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--limit_loci_length", type=int, default=1022)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu", "cuda"],
        help="Override device selection (default: auto).",
    )
    args = parser.parse_args()
    main(
        identity=args.identity,
        seeds=list(args.seeds),
        esm_model=args.esm_model,
        batch_size=args.batch_size,
        limit_loci_length=args.limit_loci_length,
        device=args.device,
    )
