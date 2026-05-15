# Phage-Host Interaction Prediction — Paper 1 companion code

[![CI](https://github.com/OWNER/REPO/actions/workflows/test.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)

> Replace `OWNER/REPO` in the badge URL above with the actual GitHub
> path once the repository is published. The workflow file itself is
> already in `.github/workflows/test.yml` and exercises pytest + ruff +
> mypy on Python 3.11.

This repository accompanies **Paper 1**, a leak-free re-evaluation of
*Klebsiella pneumoniae* phage host-range prediction. Every numerical
result in the paper (main text and supplementary) is reproducible end
to end from the scripts under `scripts/`. The upstream data are from
Boeckaerts et al., *Nat Commun* (2024), "Prediction of *Klebsiella*
phage-host specificity at the strain level" (Zenodo record
[11061100](https://zenodo.org/records/11061100)).

> **If you use this repository in your research, please cite our
> companion paper (currently available as an arXiv preprint).**
> See the [Citation](#citation) section below for BibTeX entries.

## What is in this repository

The companion code provides:

1. **Leak-free split builders** — RBP-level cluster split (E5) and
   phage-level connected-component split (F0), both backed by MMseqs2
   `linclust`. Connected-component decomposition has a hermetic
   integrity test that runs in CI.
2. **Five-method benchmark** — `esm650_xgb`, `esm650_logreg`,
   `simple_xgb`, `simple_logreg`, `homology_1nn`, `kmer_markov`
   (clean-room MIT reimplementation of the WIsH-style algorithm), each
   evaluated under both splits across five seeds.
3. **Sliding-window pooling** — ESM-2 (8M and 650M) with 1022 aa
   window and 50% overlap, comparable against truncated baselines via
   paired DeLong test with Holm–Bonferroni correction.
4. **K-type stratified evaluation** — Kaptive-reference protein-level
   surrogate K-typing (the public release lacks host genome FASTA, so
   the nucleotide Kaptive pipeline is not applicable). Decisiveness
   statistics and threshold-sensitivity validation are in Supplementary
   Table S7.
5. **TreeSHAP interpretation** — applied to `simple_xgb` with
   per-seed stability tracking.
6. **Availability probes for 10 external methods** —
   non-invasive checks (no downloads triggered) for CHERRY,
   HostPhinder, PHIST, DeepHost, PhaBox/PhaTYP, VirHostMatcher-Net,
   WIsH, PHP, Phirbo, BacteriophageIPP. The 2026-05-13 license
   re-verification record lives under
   `reports/external_methods_license_recheck_2026-05-13.{md,json}`.
7. **Label-shuffle negative controls** — two independent runs:
   `simple_xgb` × `rbp_cluster` × 3 seeds (E-α) and the headline
   configuration `esm650_xgb` × `phage_component` × 5 seeds (E-α2).
8. **Stats utilities** — fast DeLong implementation
   (Sun & Xu 2014) numerically validated against R `pROC`, plus
   stratified pair-level bootstrap (1000 resamples).

## Requirements

- **Python 3.11+**
- **PyTorch 2.6+** — MPS backend on Apple Silicon for the local
  embedding step; CUDA 12.x on a Linux GPU host for the ESM-2 650M
  sliding-window pre-compute (`scripts/f1b_esm650_sliding_gpu.py`).
- **MMseqs2 14-7e284** — `brew install mmseqs2` on macOS or via
  Bioconda.
- **Kaptive 3.0.0b6** — only required for the supplementary
  Kaptive-reference K-typing step (`scripts/f4b_ktype_kaptive_ref.py`).
  Either expose the binary on `PATH` or set `KAPTIVE_BIN`.
- See `requirements.txt` for the full pinned Python dependency list.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fetch the upstream PhageHostLearn data from Zenodo
python3 scripts/e1_reproduce_phlearn.py --stage download

# Build the leak-free splits (run both)
python3 scripts/e5_cluster_split.py            # RBP-level cluster split
python3 scripts/f0_phage_level_split.py        # phage-level (headline)

# Headline label-shuffle negative control
python3 scripts/e_alpha2_label_shuffle_esm650.py
```

## Full reproduction

The Paper 1 benchmark consists of two phases. The default Phase 2 run
covers the 16 method × split combinations in Table 1 of the main text
and Tables S1a–S1b of the supplementary. The optional Phase 2.6 flag
adds the ESM-2 650M sliding-window analysis (Table 4), the DNA k=6
k-mer Markov baseline (Table S5), and the Kaptive-reference K-typing
(Table 2, Table S7).

```bash
# Phase 2 — RBP-cluster + phage-level splits, 5 seeds
python3 scripts/reproduce_all.py \
    --identity 0.5 --seeds 42 43 44 45 46

# Phase 2 + Phase 2.6 supplementary experiments.  Requires a prior
# run of scripts/f1b_esm650_sliding_gpu.py on a CUDA GPU host; see
# that script's docstring for the SSH-upload workflow used in the
# original benchmark.
python3 scripts/reproduce_all.py \
    --identity 0.5 --seeds 42 43 44 45 46 --with-phase26
```

The driver invokes the following scripts in order:

| Step | Script | Output |
|---|---|---|
| 1 | `f0_phage_level_split.py` | `data/processed/splits/phage_split_id{id}_seed{seed}.parquet`, `reports/experiment_f0_phage_split.json` |
| 2 | `e5_cluster_split.py` | `data/processed/splits/split_id{id}_seed{seed}.parquet`, `reports/experiment_05_cluster_split.md` |
| 3 | `f_benchmark_main.py` | `data/processed/predictions/predictions_*.parquet` (80 runs) |
| 4 | `f_plus_survey.py` | `reports/f_plus_survey.{md,json}` |
| 5 | `f4_ktype_stratified.py` | `reports/f4_ktype_stratified.{md,parquet}` |
| 6 | `f9_shap_e6.py` | `reports/f9_shap_top20.{md,parquet}` |
| 7 | `f10_aggregate.py` | `reports/f10_summary.md`, `reports/f10_delong_matrix.json`, main figure |
| 8 *(P2.6 manual)* | `f1b_esm650_sliding_gpu.py` (GPU) | `predictions_esm650_host_sliding_xgb_*.parquet`, `predictions_esm650_host_truncated_xgb_*.parquet` |
| 9 | `f1b_classifier_esm650.py` | classifier on the downloaded 650M embeddings |
| 10 | `f_plus_dna_kmer_markov.py` | `reports/f_plus_dna_kmer_markov_*.md` |
| 11 | `f4b_ktype_kaptive_ref.py` | `reports/f4_ktype_kaptive_*.{md,parquet}` |

Negative controls and recently added artefacts:

| Step | Script | Output |
|---|---|---|
| α | `e_alpha_label_shuffle.py` | `simple_xgb / rbp_cluster / 3 seeds` shuffle (`reports/experiment_alpha_label_shuffle.md`) |
| α2 | `e_alpha2_label_shuffle_esm650.py` | headline `esm650_xgb / phage_component / 5 seeds` shuffle (`reports/experiment_alpha2_label_shuffle_esm650.{md,json}`) |

All outputs land under `data/processed/` (predictions, splits,
features) and `reports/` (figures, markdown tables, JSON matrices,
parquet aggregates). Predictions parquet files are
`predictions_<method>_<split_kind>_seed<N>.parquet` with the columns
`[host_id, phage_id, label, method, split_kind, seed, score]`, which is
the contract consumed by `f10_aggregate.py`.

## Testing

```bash
pytest -q                    # full unit-test suite
ruff check .                 # PEP 8 + selected rules
mypy src/                    # type check (non-strict)
```

Notable tests:

- `tests/test_phage_split.py` — connected-component decomposition,
  cluster-shared phages always co-located in the same partition, and
  train/val/test phage sets disjoint. Hermetic (no MMseqs2 binary
  dependency); runs in CI.
- `tests/test_delong_vs_pROC.py` — numerical agreement between our fast
  DeLong implementation and the R `pROC` reference within $10^{-6}$
  on synthetic fixtures.
- `tests/test_paper1_headline_stats.py` — pins headline configuration
  numerical claims (e.g. esm650-xgb sliding vs truncated DeLong
  non-significance, K-type minimum-sample reporting rule).

## Directory layout

```
code/
├── src/                       Reusable library code
│   ├── data/                  Loaders, splits, leakage report
│   ├── features/              Classical features + sliding-window pooling
│   ├── models/                XGBoost / logistic classifiers
│   ├── baselines/             Homology 1-NN, k-mer Markov, external probes
│   ├── stats/                 ROC/PR/ECE, stratified bootstrap, fast DeLong
│   ├── eval/                  Aggregation, K-type stratification, TreeSHAP
│   └── utils/                 Seeding, paths
├── scripts/                   Executable experiment entry points
│   ├── e*_*.py                Phase 1 / Phase 2 scripts
│   ├── e_alpha_label_shuffle.py
│   ├── e_alpha2_label_shuffle_esm650.py
│   ├── f0_phage_level_split.py
│   ├── f1*.py                 Sliding-window experiments
│   ├── f4*.py                 K-type stratified evaluation
│   ├── f9_shap_e6.py
│   ├── f10_aggregate.py
│   ├── f_plus_*.py            Supplementary baselines + external probes
│   └── reproduce_all.py       End-to-end driver
├── tests/                     pytest suite (split integrity, DeLong, etc.)
├── .github/workflows/test.yml CI (ruff + mypy + pytest + pip-audit)
├── requirements.txt           Pinned Python dependencies
├── pyproject.toml             ruff / mypy / pytest configuration
└── README.md                  this file
```

The project also includes top-level companion directories:

- `../data/` — raw data (Zenodo download), processed splits, prediction
  artefacts, and SHA-256 manifest.
- `../reports/` — aggregated experimental outputs (parquet, JSON,
  markdown) cited by the paper. Each numerical claim in the paper is
  cross-referenced to a file under this directory.
- `../paper/` — LaTeX manuscript and standalone supplementary
  (1-column article class). See the per-directory README for build
  instructions.

## Reproducibility guarantees

- **Random seeds** — global seeding via `src/utils/seed.py` covers
  `random`, `numpy`, `torch`, and `torch.mps` (when available). The
  five seeds `{42, 43, 44, 45, 46}` are used identically for split
  generation, classifier training, and bootstrap resampling.
- **Cluster-aware splits** — both splits use MMseqs2 `linclust` to
  prevent identity-based leakage between train and test. The
  phage-level split additionally enforces that phages sharing any RBP
  cluster co-locate in the same partition (connected-component
  decomposition).
- **Pre-computed embeddings** — the ESM-2 650M embeddings come from
  the upstream Zenodo release (`esm2_embeddings_rbp.csv`,
  `esm2_embeddings_loci.csv`). The phage-side feature vector is the
  *split-invariant* average across all RBPs of a phage, computed once
  before splits are applied, so the same phage carries the same
  feature vector across train/val/test.
- **Hyper-parameter freeze** — all classifiers run with fixed
  hyper-parameters across methods, splits, and seeds (XGBoost:
  `n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
  colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr"`;
  logistic regression: `max_iter=1000, class_weight="balanced"`, L2,
  pipelined with `StandardScaler`). No per-method tuning is
  performed (cross-method fair comparison).
- **Data provenance** — every downloaded artefact records its SHA-256
  under `data/raw/.sha256.txt`; re-downloads are byte-identical or
  the verifier fails loudly.

## License

MIT — see [`LICENSE`](LICENSE).

The 10 external methods evaluated by the availability probe carry
their own upstream licenses. We do **not** redistribute any of their
code. The probe is non-invasive and only reports their presence on
the local machine. See
`reports/external_methods_license_recheck_2026-05-13.md` for the
2026-05-13 license verification record.

## Citation

**If you use this repository or its derivative outputs in your
research, please cite our companion paper.** The paper is currently
available as an arXiv preprint; replace `ARXIV_ID` below with the
actual arXiv identifier once it is assigned (we will update this
README after submission). When the journal version becomes available,
please cite the journal version instead.

```bibtex
@misc{inoshita2026phageleakfree,
  title         = {Leak-free Re-evaluation of Phage Host-range Prediction Benchmarks},
  author        = {Inoshita, Keito and Tanaka, Tomoki and Okano, Kenji and Iwaki, Hiroaki},
  year          = {2026},
  archivePrefix = {arXiv},
}
```

Please also acknowledge the upstream PhageHostLearn dataset on which
this benchmark is built:

```bibtex
@article{boeckaerts2024phagehostlearn,
  title   = {Prediction of {Klebsiella} phage-host specificity at the strain level},
  author  = {Boeckaerts, Dimitri and Stock, Michiel and Ferriol-Gonz{\'a}lez, Celia and Oteo-Iglesias, Jes{\'u}s and Sanju{\'a}n, Rafael and Domingo-Calap, Pilar and De Baets, Bernard and Briers, Yves},
  journal = {Nature Communications},
  volume  = {15},
  number  = {1},
  pages   = {4355},
  year    = {2024},
  doi     = {10.1038/s41467-024-48675-6}
}
```


