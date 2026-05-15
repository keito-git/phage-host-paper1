"""SHA-256 bookkeeping for downloaded artefacts.

Each call to :func:`record_sha256` appends one line to the registry file so we
can later verify that a given data file has not been modified on disk.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex-encoded SHA-256 digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def record_sha256(path: Path, registry: Path) -> str:
    """Compute the hash of ``path`` and append it to ``registry``.

    The registry file is a newline-delimited ``<hex_digest>  <relative_path>``
    list.  Duplicate entries for the same path are overwritten.
    """
    digest = sha256_of_file(path)
    rel = path.resolve().as_posix()

    lines: list[str] = []
    if registry.exists():
        for line in registry.read_text().splitlines():
            if not line.strip():
                continue
            _digest, _rel = line.split("  ", 1)
            if _rel != rel:
                lines.append(line)
    lines.append(f"{digest}  {rel}")
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("\n".join(lines) + "\n")
    return digest
