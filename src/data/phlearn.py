# MIT License. See LICENSE in repository root.
"""Parsers and loaders for the PhageHostLearn (Zenodo 11061100) dataset.

The raw dataset ships with a wide interaction matrix (rows = host strains,
columns = phages), a set of phage receptor-binding protein (RBP) sequences,
and K-locus protein lists per host strain.  The functions below convert these
heterogeneous files into tidy long-form frames that the downstream
experiments can consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR


@dataclass(frozen=True)
class PhLearnTables:
    """Tidy PhageHostLearn tables used by every experiment."""

    interactions: pd.DataFrame  # columns: [host_id, phage_id, label]
    rbps: pd.DataFrame          # columns: [phage_id, protein_id, sequence, dna, xgb_score]
    loci: pd.DataFrame          # columns: [host_id, sequences] (list[str])

    def overview(self) -> dict[str, int]:
        """Return dataset sizes for logging / reports."""
        return {
            "num_interactions": len(self.interactions),
            "num_positive_pairs": int((self.interactions.label == 1).sum()),
            "num_negative_pairs": int((self.interactions.label == 0).sum()),
            "num_phages": self.interactions.phage_id.nunique(),
            "num_hosts": self.interactions.host_id.nunique(),
            "num_rbp_proteins": len(self.rbps),
            "num_phages_in_rbps": self.rbps.phage_id.nunique(),
            "num_hosts_in_loci": len(self.loci),
        }


def load_interactions(path: Path | None = None) -> pd.DataFrame:
    """Return the interaction matrix in long form.

    Parameters
    ----------
    path:
        Optional override for the CSV location; defaults to
        ``RAW_DIR / 'phage_host_interactions.csv'``.

    Returns
    -------
    DataFrame with columns ``host_id``, ``phage_id``, ``label``; rows with
    missing labels (NaN) are dropped so the resulting frame only contains
    observed pairs.
    """
    src = path or (RAW_DIR / "phage_host_interactions.csv")
    wide = pd.read_csv(src, index_col=0)
    long = wide.stack().reset_index()
    long.columns = ["host_id", "phage_id", "label"]
    long["label"] = long["label"].astype(int)
    return long


def load_rbps(path: Path | None = None) -> pd.DataFrame:
    """Load RBPbase.csv and return it with canonical column names."""
    src = path or (RAW_DIR / "RBPbase.csv")
    df = pd.read_csv(src)
    return df.rename(
        columns={
            "phage_ID": "phage_id",
            "protein_ID": "protein_id",
            "protein_sequence": "sequence",
            "dna_sequence": "dna",
        }
    )


def load_loci(path: Path | None = None) -> pd.DataFrame:
    """Load Locibase.json and return a DataFrame of host_id -> list of sequences."""
    src = path or (RAW_DIR / "Locibase.json")
    raw = json.loads(Path(src).read_text())
    rows = [{"host_id": host, "sequences": seqs} for host, seqs in raw.items()]
    return pd.DataFrame(rows)


def load_all(
    raw_dir: Path = RAW_DIR, restrict_to_loci: bool = True
) -> PhLearnTables:
    """Load every tidy table in one shot.

    Parameters
    ----------
    raw_dir:
        Folder holding the Zenodo artefacts.
    restrict_to_loci:
        When ``True`` (default) rows of the interaction matrix whose ``host_id``
        is **not** present in Locibase are dropped.  The PhageHostLearn release
        of Zenodo 11061100 contains 15 such rows (``K2`` / ``K21`` / ...) that
        reference generic K-typing references rather than specific strains, so
        they have no corresponding K-locus amino-acid sequence in the JSON
        dump.  Downstream feature extractors need a sequence, hence the
        default filter.
    """
    interactions = load_interactions(raw_dir / "phage_host_interactions.csv")
    rbps = load_rbps(raw_dir / "RBPbase.csv")
    loci = load_loci(raw_dir / "Locibase.json")

    if restrict_to_loci:
        host_ids_with_loci = set(loci.host_id)
        before = len(interactions)
        interactions = interactions[
            interactions.host_id.isin(host_ids_with_loci)
        ].reset_index(drop=True)
        dropped = before - len(interactions)
        if dropped:
            # We deliberately keep a visible side-effect rather than silently
            # shrinking the data.  The count is small and deterministic.
            import warnings

            warnings.warn(
                f"Dropped {dropped} interaction rows whose host_id has no "
                "K-locus sequence in Locibase.json (e.g. generic K-typing "
                "references like 'K2').",
                stacklevel=2,
            )

    return PhLearnTables(interactions=interactions, rbps=rbps, loci=loci)


def pair_with_first_rbp(
    interactions: pd.DataFrame, rbps: pd.DataFrame
) -> pd.DataFrame:
    """Attach the first-listed RBP sequence of each phage to every interaction.

    When a phage has multiple candidate RBPs, PhageHostLearn concatenates them
    at inference time.  For the lightweight baselines in this pilot we collapse
    to the representative sequence with the highest ``xgb_score``.
    """
    best = rbps.sort_values("xgb_score", ascending=False).drop_duplicates("phage_id")
    merged = interactions.merge(
        best[["phage_id", "protein_id", "sequence"]],
        on="phage_id",
        how="inner",
    )
    return merged.rename(columns={"sequence": "rbp_sequence"})


def flatten_loci(loci: pd.DataFrame) -> pd.DataFrame:
    """Concatenate per-host K-locus sequences with ``*`` separators.

    The separator token preserves protein boundaries when the string is later
    fed to MMseqs2 (``*`` terminates a sequence) or any embedding model that
    splits on stop symbols.
    """
    loci = loci.copy()
    loci["k_locus_concat"] = loci["sequences"].apply(lambda seqs: "*".join(seqs))
    return loci[["host_id", "k_locus_concat"]]
