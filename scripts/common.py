"""Shared boilerplate for experiment scripts.

Every ``e*_.py`` entry point imports :func:`ensure_path` so it can be run
directly from the ``scripts/`` folder without having to set ``PYTHONPATH``
manually.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_path() -> None:
    """Make ``src`` importable when the script is run as a file."""
    code_dir = Path(__file__).resolve().parent.parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
