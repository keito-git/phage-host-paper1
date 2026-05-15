# MIT License. See LICENSE in repository root.
"""One-command reproducer for Paper 1 Phase 2 + Phase 2.6 experiments.

Runs in order:
    1. f0_phage_level_split.py (generates the phage-level splits)
    2. e5_cluster_split.py x seeds (RBP-level splits)
    3. f_benchmark_main.py (all methods × 2 splits × 5 seeds)
    4. f_plus_survey.py (availability probe for external methods)
    5. f4_ktype_stratified.py (K-type stratification, cluster surrogate)
    6. f9_shap_e6.py (SHAP on classical features)
    7. f10_aggregate.py (main tables and figures)

Phase 2.6 additions (optional; pass --with-phase26 to enable):
    8. f1b_esm650_sliding_gpu.py (run manually on GPU 2 before step 9)
    9. f1b_classifier_esm650.py (classifier on downloaded 650M embeddings)
    10. f_plus_dna_kmer_markov.py (DNA k=6 WIsH-style supplementary)
    11. f4b_ktype_kaptive_ref.py (Kaptive-reference K-typing & re-evaluation)

Step 8 requires:
    - SSH access to the university GPU server (GPU 2 only).
    - Manual upload of data/raw/Locibase.json + the script, run of the
      script under ``CUDA_VISIBLE_DEVICES=2``, and download of the two
      parquet outputs + ``run_meta.json`` into
      ``data/processed/f1_cache/``.  See the README for commands.
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import subprocess
import sys
from pathlib import Path


def _run(step: str, args: list[str]) -> None:
    script = Path(__file__).resolve().parent / step
    print(f"\n=== running {step} {args} ===\n", flush=True)
    subprocess.run([sys.executable, str(script), *args], check=True)


def main(seeds: list[int], identity: float, with_phase26: bool = False) -> None:
    seed_args = [str(s) for s in seeds]
    _run("f0_phage_level_split.py", ["--identity", str(identity), "--seeds", *seed_args])
    for s in seeds:
        _run("e5_cluster_split.py", ["--identity", str(identity), "--seed", str(s)])
    _run("f_benchmark_main.py", ["--identity", str(identity), "--seeds", *seed_args])
    _run("f_plus_survey.py", [])
    _run("f4_ktype_stratified.py", [])
    _run("f9_shap_e6.py", ["--identity", str(identity), "--seeds", *seed_args,
                            "--split-kind", "phage_component"])
    _run("f10_aggregate.py", [])

    if with_phase26:
        # Step 8 (GPU 2, manual): f1b_esm650_sliding_gpu.py — see script docstring.
        # Step 9: classifier on the GPU-computed embeddings (M4 Max).
        _run("f1b_classifier_esm650.py",
             ["--identity", str(identity), "--seeds", *seed_args])
        _run("f_plus_dna_kmer_markov.py",
             ["--identity", str(identity), "--seeds", *seed_args, "--k", "6"])
        _run("f4b_ktype_kaptive_ref.py",
             ["--identity", str(identity), "--seeds", *seed_args])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--with-phase26",
        action="store_true",
        help="Also run Phase 2.6 additional experiments (requires GPU pre-step)",
    )
    args = parser.parse_args()
    main(list(args.seeds), args.identity, args.with_phase26)
