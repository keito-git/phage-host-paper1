# MIT License. See LICENSE in repository root.
"""Experiment alpha: label-shuffle negative control.

Re-runs the E6 pipeline with randomised labels.  A correctly implemented
model must collapse to prevalence-level performance on shuffled labels;
anything better than that would signal data leakage or implementation bugs.
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

from src.config import MULTI_SEEDS, PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from src.data.phlearn import flatten_loci, load_loci
from src.features.simple_features import summarise_frame
from src.models.classifiers import evaluate, make_logistic, make_xgboost
from src.utils.seed import set_global_seed


def _load_split(identity: float, seed: int) -> pd.DataFrame:
    path = PROCESSED_DIR / "splits" / f"split_id{identity}_seed{seed}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Split missing: {path}. Run E5 first.")
    return pd.read_parquet(path)


def _shuffle_labels(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = y.copy()
    rng.shuffle(shuffled)
    return shuffled


def run_once(df: pd.DataFrame, loci_flat: pd.DataFrame, seed: int) -> dict:
    set_global_seed(seed)

    train = df[df.split == "train"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)

    # Shuffle train labels inside the train partition only; keep test labels
    # intact so the scoring reflects the classifier's ability to *find*
    # structure in random labels (it should not).
    y_train = _shuffle_labels(train.label.to_numpy(), seed=seed)
    y_test = test.label.to_numpy()

    rbp_train = summarise_frame(train["rbp_sequence"])
    rbp_test = summarise_frame(test["rbp_sequence"])
    loc_train = summarise_frame(
        train.merge(loci_flat, on="host_id", how="left")["k_locus_concat"]
    )
    loc_test = summarise_frame(
        test.merge(loci_flat, on="host_id", how="left")["k_locus_concat"]
    )
    X_train = np.hstack([rbp_train, loc_train]).astype(np.float32)
    X_test = np.hstack([rbp_test, loc_test]).astype(np.float32)

    scale_pos = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))
    results: dict[str, dict] = {}
    for name, clf in [
        ("logreg_shuffled", make_logistic(seed)),
        ("xgboost_shuffled", make_xgboost(seed, scale_pos_weight=scale_pos)),
    ]:
        clf.fit(X_train, y_train)
        scores = clf.predict_proba(X_test)[:, 1]
        m = evaluate(y_test, scores)
        results[name] = {
            "roc_auc": m.roc_auc,
            "pr_auc": m.pr_auc,
            "best_f1": m.best_f1,
            "test_prevalence": float(y_test.mean()),
        }
    return results


def main(identity: float, seeds: list[int]) -> None:
    ensure_dirs()
    loci_flat = flatten_loci(load_loci())

    per_seed: dict[int, dict] = {}
    for s in seeds:
        df = _load_split(identity, s)
        per_seed[s] = run_once(df, loci_flat, seed=s)

    aggregated: dict[str, dict[str, float]] = {}
    for model in per_seed[seeds[0]]:
        for metric in ("roc_auc", "pr_auc", "best_f1"):
            vals = [per_seed[s][model][metric] for s in seeds]
            aggregated.setdefault(model, {})[f"{metric}_mean"] = float(statistics.mean(vals))
            aggregated[model][f"{metric}_std"] = float(statistics.pstdev(vals))

    summary = {
        "identity": identity,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "aggregated": aggregated,
    }
    render_report(REPORTS_DIR / "experiment_alpha_label_shuffle.md", summary)
    print(json.dumps(summary, indent=2))


def render_report(path: Path, summary: dict) -> None:
    lines = [
        "# Experiment alpha -- Label-shuffle negative control",
        "",
        "**Goal.** Confirm that the simple-feature pipeline collapses to",
        "prevalence-level performance once training labels are randomly",
        "shuffled.  A well-specified experiment should yield ROC-AUC ~= 0.5",
        "and PR-AUC ~= test prevalence here.  Anything higher suggests a leak.",
        "",
        f"- Identity: `{summary['identity']}`, Seeds: `{summary['seeds']}`",
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
        "If shuffled-label ROC-AUC is markedly above 0.5 the pipeline is",
        "leaking information via features (e.g. host ID acting as a lookup).",
        "If the shuffled PR-AUC sits close to the test prevalence the setup",
        "is sound.",
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(MULTI_SEEDS))
    args = parser.parse_args()
    main(identity=args.identity, seeds=list(args.seeds))
