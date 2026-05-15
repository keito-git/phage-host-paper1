"""Download helpers for the PhageHostLearn dataset (Zenodo record 11061100)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import PHLEARN_ZENODO_API_URL, RAW_DIR
from src.utils.hashing import record_sha256


@dataclass(frozen=True)
class RemoteFile:
    """A file entry as returned by the Zenodo metadata endpoint."""

    key: str
    url: str
    size: int | None
    checksum: str | None

    @classmethod
    def from_record(cls, entry: dict) -> RemoteFile:
        links = entry.get("links", {})
        url = links.get("self") or links.get("content") or ""
        return cls(
            key=entry["key"],
            url=url,
            size=entry.get("size"),
            checksum=entry.get("checksum"),
        )


def list_zenodo_files(record_url: str = PHLEARN_ZENODO_API_URL) -> list[RemoteFile]:
    """Return the list of files exposed by the Zenodo record."""
    response = requests.get(record_url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    return [RemoteFile.from_record(entry) for entry in payload.get("files", [])]


def download_file(
    remote: RemoteFile, dest_dir: Path = RAW_DIR, force: bool = False
) -> Path:
    """Download ``remote`` into ``dest_dir`` and record its SHA-256.

    Skips the download if the file already exists and ``force`` is ``False``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / remote.key
    if out.exists() and not force:
        record_sha256(out, dest_dir / ".sha256.txt")
        return out

    with requests.get(remote.url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", remote.size or 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=remote.key)
        with out.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))
        bar.close()

    record_sha256(out, dest_dir / ".sha256.txt")
    return out


def download_all(force: bool = False) -> list[Path]:
    """Download every file referenced by the PhageHostLearn Zenodo record."""
    remotes = list_zenodo_files()
    return [download_file(r, force=force) for r in remotes]
