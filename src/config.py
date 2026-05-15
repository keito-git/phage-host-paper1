# MIT License. See LICENSE in repository root.
"""Project-wide paths and constants.

All filesystem paths are resolved relative to the repository root, so the code
can be checked out and run in any location without editing this file.
"""

from __future__ import annotations

from pathlib import Path

# Repository layout ----------------------------------------------------------
# <project_root>/
#   code/     <- this file lives at code/src/config.py
#   data/
#   reports/
CODE_DIR: Path = Path(__file__).resolve().parent.parent
PROJECT_ROOT: Path = CODE_DIR.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
CACHE_DIR: Path = DATA_DIR / "cache"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# Upstream data --------------------------------------------------------------
# PhageHostLearn (Boeckaerts et al. 2024) - Zenodo record 11061100
PHLEARN_ZENODO_RECORD_ID: str = "11061100"
PHLEARN_ZENODO_API_URL: str = (
    f"https://zenodo.org/api/records/{PHLEARN_ZENODO_RECORD_ID}"
)

# Determinism ----------------------------------------------------------------
DEFAULT_SEED: int = 42
MULTI_SEEDS: tuple[int, ...] = (42, 43, 44)


def ensure_dirs() -> None:
    """Create all known data / report directories if missing."""
    for d in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
