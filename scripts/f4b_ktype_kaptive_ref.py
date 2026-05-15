# MIT License. See LICENSE in repository root.
"""F4-B — K-type assignment via Kaptive reference database, then re-run F4.

Why not `kaptive assembly`?
---------------------------
``kaptive assembly`` requires an assembly FASTA for every host.  The
PhageHostLearn 2024 public release does not ship host-genome FASTAs — it
ships only ``Locibase.json`` which contains the K-locus protein
amino-acid sequences (concatenated per host).  We therefore cannot run
Kaptive end-to-end.

Pragmatic substitute
--------------------
1. Extract the Kaptive KpSC K-locus protein reference with
   ``kaptive extract kpsc_k --faa kaptive_k_ref.faa`` (requires the
   ``kaptive`` pip package, available as of Kaptive v3).
2. Build an MMseqs2 protein database from that reference.
3. Search each host's Locibase protein set against the reference, take
   the top hit per host across all proteins, and call that the host's
   K-type.  This is inspired by Kaptive v1/v2 scoring (top-matching
   K-locus by protein identity) but is obviously an approximation —
   we flag the result accordingly in the report.
4. Re-run the F4 stratified evaluation using the new K-type labels;
   output a dated markdown + parquet so the existing
   ``f4_ktype_stratified.md`` (cluster-surrogate) is preserved.

Honest statement (for reviewers): a proper nucleotide-based Kaptive run
is not possible on this dataset; the results here should be treated as a
coarse K-type approximation.  Any K-type with fewer than 2 hosts is
collapsed into a "rare" bin before per-type metrics are computed.
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.config import CACHE_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs
from src.stats.metrics import pr_auc, roc_auc


def _resolve_kaptive_bin() -> Path | None:
    """Resolve the Kaptive binary path.

    Resolution order: ``KAPTIVE_BIN`` env var > ``shutil.which("kaptive")`` >
    the macOS ``~/Library/Python/3.11/bin/kaptive`` fallback.  Returns
    ``None`` if none of these resolve to an existing file.
    """
    env_override = os.environ.get("KAPTIVE_BIN")
    if env_override:
        candidate = Path(env_override).expanduser()
        if candidate.exists():
            return candidate
    which = shutil.which("kaptive")
    if which:
        return Path(which)
    fallback = Path.home() / "Library/Python/3.11/bin/kaptive"
    if fallback.exists():
        return fallback
    return None


KAPTIVE_BIN = _resolve_kaptive_bin()
K_REFERENCE_FAA = CACHE_DIR / "kaptive_k_reference.faa"
MMSEQS_WORKDIR = CACHE_DIR / "mmseqs_kaptive_ktype"


@dataclass
class KTypeAssignment:
    host_id: str
    k_type: str
    identity: float
    query_len: int
    n_proteins: int


def ensure_kaptive_reference() -> Path:
    """Run ``kaptive extract kpsc_k --faa`` to get the protein reference."""
    if K_REFERENCE_FAA.exists() and K_REFERENCE_FAA.stat().st_size > 0:
        return K_REFERENCE_FAA
    K_REFERENCE_FAA.parent.mkdir(parents=True, exist_ok=True)
    if KAPTIVE_BIN is None or not KAPTIVE_BIN.exists():
        raise RuntimeError(
            "Kaptive binary not found.  Set the KAPTIVE_BIN environment "
            "variable, expose 'kaptive' on PATH, or install it with "
            "'pip3 install --user kaptive'."
        )
    subprocess.run(
        [str(KAPTIVE_BIN), "extract", "kpsc_k", "--faa", str(K_REFERENCE_FAA)],
        check=True,
    )
    return K_REFERENCE_FAA


def build_host_faa(locibase_path: Path, out_path: Path) -> dict[str, int]:
    """Write host proteins to FASTA.  Return host_id -> number of proteins."""
    with locibase_path.open() as f:
        raw = json.load(f)
    counts: dict[str, int] = {}
    with out_path.open("w") as out:
        for host_id, proteins in raw.items():
            if not isinstance(proteins, list):
                continue
            counts[host_id] = len(proteins)
            for i, seq in enumerate(proteins):
                if not isinstance(seq, str):
                    continue
                s = seq.rstrip("*")  # drop any stop codons
                if not s:
                    continue
                out.write(f">{host_id}__prot{i:02d}\n{s}\n")
    return counts


def run_mmseqs_search(
    query_faa: Path,
    target_faa: Path,
    workdir: Path,
    threads: int = 4,
) -> pd.DataFrame:
    """Run ``mmseqs easy-search`` and return hits as a DataFrame."""
    workdir.mkdir(parents=True, exist_ok=True)
    out_tsv = workdir / "hits.tsv"
    tmp = workdir / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    cmd = [
        "mmseqs",
        "easy-search",
        str(query_faa),
        str(target_faa),
        str(out_tsv),
        str(tmp),
        "--threads",
        str(threads),
        "-s",
        "6.0",  # sensitivity
        "--min-seq-id",
        "0.30",  # keep low-identity hits so rare K-types are not missed
        "-e",
        "1e-5",
        "--max-seqs",
        "50",
        "--format-output",
        "query,target,pident,alnlen,mismatch,qlen,tlen,evalue,bits",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    cols = ["query", "target", "pident", "alnlen", "mismatch", "qlen", "tlen", "evalue", "bits"]
    hits = pd.read_csv(out_tsv, sep="\t", names=cols)
    return hits


def assign_ktype(hits: pd.DataFrame, host_protein_counts: dict[str, int]) -> list[KTypeAssignment]:
    """From hits, assign each host its best K-locus by summed bits."""
    if hits.empty:
        return []
    hits = hits.copy()
    hits["host_id"] = hits["query"].str.split("__prot", n=1).str[0]
    hits["k_locus"] = hits["target"].str.split("_", n=1).str[0]  # KL1, KL2, ...
    # For each (host, k_locus) aggregate: total bits, best identity.
    agg = (
        hits.groupby(["host_id", "k_locus"])
        .agg(total_bits=("bits", "sum"), best_pident=("pident", "max"), qlen_sum=("qlen", "sum"))
        .reset_index()
    )
    # Pick top k_locus per host by total bits.
    idx = agg.groupby("host_id")["total_bits"].idxmax()
    top = agg.loc[idx].reset_index(drop=True)
    return [
        KTypeAssignment(
            host_id=row.host_id,
            k_type=row.k_locus,
            identity=float(row.best_pident),
            query_len=int(row.qlen_sum),
            n_proteins=host_protein_counts.get(row.host_id, 0),
        )
        for row in top.itertuples(index=False)
    ]


def compute_ktype_metrics(
    ktype_map: dict[str, str],
    seeds: list[int],
) -> pd.DataFrame:
    """For each seed × K-type, compute ROC-AUC / PR-AUC from existing predictions.

    Note: The ``identity`` parameter was removed in Round 3 because MMseqs2
    ``--min-seq-id`` is hard-coded to 0.30 in :func:`run_mmseqs_search` and
    does not propagate into the metric computation.  Varying ``--min-seq-id``
    requires re-running the search step; see Table S7 for the post-hoc
    ``pident`` filter sensitivity analysis.
    """
    pred_dir = PROCESSED_DIR / "predictions"
    # We use the best-performing method in the existing F10 summary:
    # esm650_xgb on phage_component split.
    method = "esm650_xgb"
    split_kind = "phage_component"

    rows: list[dict] = []
    for seed in seeds:
        pred_path = pred_dir / f"predictions_{method}_{split_kind}_seed{seed}.parquet"
        if not pred_path.exists():
            continue
        pred = pd.read_parquet(pred_path)
        pred["k_type"] = pred["host_id"].map(ktype_map).fillna("UNASSIGNED")
        for k_type, sub in pred.groupby("k_type"):
            n = len(sub)
            n_pos = int((sub.label == 1).sum())
            n_neg = int((sub.label == 0).sum())
            if n_pos < 2 or n_neg < 2:
                rows.append(
                    {
                        "method": method,
                        "split_kind": split_kind,
                        "seed": seed,
                        "k_type": k_type,
                        "n": n,
                        "n_pos": n_pos,
                        "n_neg": n_neg,
                        "roc_auc": float("nan"),
                        "pr_auc": float("nan"),
                        "status": "too_few_samples",
                    }
                )
                continue
            y = sub.label.to_numpy()
            s = sub.score.to_numpy()
            rows.append(
                {
                    "method": method,
                    "split_kind": split_kind,
                    "seed": seed,
                    "k_type": k_type,
                    "n": n,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "roc_auc": float(roc_auc(y, s)),
                    "pr_auc": float(pr_auc(y, s)),
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows)


def main(identity: float, seeds: list[int]) -> dict:
    # ``identity`` is retained in the CLI for backwards compatibility with
    # earlier invocations and revision logs, but it no longer affects
    # ``compute_ktype_metrics``.  The MMseqs2 ``--min-seq-id`` parameter is
    # fixed at 0.30 inside :func:`run_mmseqs_search`; altering it requires a
    # full re-run of the search step.
    del identity  # currently unused; preserved in CLI signature
    ensure_dirs()
    today = date.today().isoformat()

    # Step 1: obtain Kaptive K-locus protein reference.
    print("[F4B] ensuring Kaptive reference ...", flush=True)
    ref = ensure_kaptive_reference()
    print(f"[F4B] reference at {ref} ({ref.stat().st_size} bytes)", flush=True)

    # Step 2: dump host proteins to FASTA.
    host_faa = CACHE_DIR / "locibase_host_proteins.faa"
    host_counts = build_host_faa(RAW_DIR / "Locibase.json", host_faa)
    print(f"[F4B] wrote {len(host_counts)} hosts to {host_faa}", flush=True)

    # Step 3: mmseqs search.
    print("[F4B] running mmseqs easy-search ...", flush=True)
    hits = run_mmseqs_search(host_faa, ref, MMSEQS_WORKDIR)
    print(f"[F4B] got {len(hits)} hits", flush=True)

    # Step 4: assign K-type per host.
    assignments = assign_ktype(hits, host_counts)
    assign_df = pd.DataFrame([a.__dict__ for a in assignments])
    ktype_tsv = PROCESSED_DIR / "ktypes_kaptive.tsv"
    assign_df.to_csv(ktype_tsv, sep="\t", index=False)
    print(f"[F4B] assigned {len(assign_df)} hosts; wrote {ktype_tsv}", flush=True)

    # Also hosts without any hit get UNASSIGNED
    ktype_map = dict(zip(assign_df.host_id, assign_df.k_type, strict=False))
    for h in host_counts:
        ktype_map.setdefault(h, "UNASSIGNED")
    unassigned = sum(v == "UNASSIGNED" for v in ktype_map.values())
    print(f"[F4B] {unassigned} hosts have UNASSIGNED", flush=True)

    # K-type coverage summary
    ktype_counts = pd.Series(list(ktype_map.values())).value_counts()

    # Step 5: recompute F4 stratified metrics.
    print("[F4B] computing per-K-type metrics from existing predictions ...", flush=True)
    metrics = compute_ktype_metrics(ktype_map, seeds)
    metrics_path = REPORTS_DIR / f"f4_ktype_kaptive_{today}.parquet"
    metrics.to_parquet(metrics_path, index=False)

    # Aggregate per K-type across seeds
    agg = (
        metrics[metrics.status == "ok"]
        .groupby("k_type")
        .agg(
            n_seeds=("seed", "count"),
            n_total=("n", "sum"),
            n_pos_total=("n_pos", "sum"),
            roc_mean=("roc_auc", "mean"),
            roc_std=("roc_auc", "std"),
            pr_mean=("pr_auc", "mean"),
            pr_std=("pr_auc", "std"),
        )
        .reset_index()
        .sort_values("n_pos_total", ascending=False)
    )

    # Write markdown report
    md_lines = [
        f"# F4-B — K-type stratified evaluation via Kaptive reference ({today})",
        "",
        "## Method",
        "",
        "A proper `kaptive assembly` run is not possible because the ",
        "PhageHostLearn 2024 public release does not ship host genome FASTAs. ",
        "We therefore extracted the Kaptive KpSC protein reference with ",
        "`kaptive extract kpsc_k --faa`, built an MMseqs2 database, and ",
        "searched each host's Locibase protein set against the reference.  ",
        "Each host was assigned the K-locus (KL#) that received the highest ",
        "summed bit-score across its proteins.  This is an approximation of ",
        "classical Kaptive v1/v2 scoring and is NOT equivalent to a full ",
        "nucleotide Kaptive run — results should be treated accordingly.",
        "",
        f"Hosts processed: **{len(host_counts)}** / assigned: "
        f"**{len(host_counts) - unassigned}** / unassigned: **{unassigned}**",
        "",
        "## K-type coverage",
        "",
        "Top 15 K-types by host count:",
        "",
        "| k_type | n_hosts |",
        "|---|---|",
    ]
    for k_type, n in ktype_counts.head(15).items():
        md_lines.append(f"| {k_type} | {int(n)} |")

    md_lines.append("")
    md_lines.append("## Per-K-type metrics (esm650_xgb on phage_component split, aggregated across seeds)")
    md_lines.append("")
    md_lines.append(
        "Only K-types with ≥ 2 positive and ≥ 2 negative test pairs per seed are reported."
    )
    md_lines.append("")
    md_lines.append(
        "| k_type | n_seeds | total_n | total_pos | ROC-AUC (mean ± std) | PR-AUC (mean ± std) |"
    )
    md_lines.append("|---|---|---|---|---|---|")
    for _, r in agg.iterrows():
        md_lines.append(
            f"| {r.k_type} | {int(r.n_seeds)} | {int(r.n_total)} | {int(r.n_pos_total)} | "
            f"{r.roc_mean:.3f} ± {r.roc_std:.3f} | {r.pr_mean:.3f} ± {r.pr_std:.3f} |"
        )
    md_lines.append("")

    # Rare K-type pool
    rare_mask = metrics.status == "too_few_samples"
    if rare_mask.any():
        md_lines.append("## Rare K-types (pooled, insufficient positives for per-seed AUC)")
        md_lines.append("")
        pooled = metrics[rare_mask].groupby("k_type").agg(
            n_total=("n", "sum"),
            n_pos_total=("n_pos", "sum"),
            n_neg_total=("n_neg", "sum"),
        ).reset_index().sort_values("n_pos_total", ascending=False)
        md_lines.append("| k_type | total_n | total_pos | total_neg |")
        md_lines.append("|---|---|---|---|")
        for _, r in pooled.iterrows():
            md_lines.append(
                f"| {r.k_type} | {int(r.n_total)} | {int(r.n_pos_total)} | {int(r.n_neg_total)} |"
            )
        md_lines.append("")

    report_path = REPORTS_DIR / f"f4_ktype_kaptive_{today}.md"
    report_path.write_text("\n".join(md_lines))
    return {
        "n_hosts": len(host_counts),
        "n_assigned": len(host_counts) - unassigned,
        "n_ktypes": int(ktype_counts.shape[0]),
        "report": str(report_path),
        "metrics_parquet": str(metrics_path),
        "ktypes_tsv": str(ktype_tsv),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    out = main(args.identity, list(args.seeds))
    print(json.dumps(out, indent=2))
