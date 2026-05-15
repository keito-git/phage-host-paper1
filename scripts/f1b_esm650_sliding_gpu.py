# MIT License. See LICENSE in repository root.
"""F1-B — Compute ESM-2 650M sliding-window host K-locus embeddings on GPU.

This is a *standalone* script intended to run inside
``~/phage_paper1_esm650`` on the university GPU server.  It depends only on
``torch + transformers + numpy + pandas`` and does NOT import
``src/...`` so it can run without installing the full repository.

GPU policy (2026-04-21 PI decision)
-----------------------------------
- **GPU 2 ONLY**.  This script sets ``CUDA_VISIBLE_DEVICES='2'`` **before**
  ``import torch`` and asserts ``torch.cuda.device_count() == 1`` — if the
  server is configured so GPU 2 is not visible, the script aborts rather
  than silently falling back to another GPU.
- Do not touch files outside ``~/phage_paper1_esm650/``.

Inputs (placed next to this script)
-----------------------------------
- ``Locibase.json``                host_id -> list[str] of K-locus proteins
- (nothing else — we handle only the host side here)

Outputs (written next to this script)
-------------------------------------
- ``host_esm650_sliding.parquet``  host_id, embedding (list[float], D=1280)
- ``host_esm650_truncated.parquet`` host_id, embedding (list[float], D=1280)
- ``run_meta.json``                model name, sha256 of inputs, device info

Run
---
::

    CUDA_VISIBLE_DEVICES=2 python f1b_esm650_sliding_gpu.py

We intentionally use the smallest useful batch size (1) because memory is
not the bottleneck — each window is only 1022 tokens — while time is, so
we mostly just want to stream the ~3200 host windows through quickly.
"""

from __future__ import annotations

# CRITICAL: GPU policy — pin to GPU 2 BEFORE torch import.  The university
# server has 3 GPUs (0, 1, 2) and PI allows GPU 2 only.
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

HERE: Path = Path(__file__).resolve().parent
MODEL_ID: str = "facebook/esm2_t33_650M_UR50D"
WINDOW_SIZE: int = 1022  # ESM-2 position embedding upper bound
STRIDE: int = 511  # 50% overlap, matches existing F1 8M design
BATCH_SIZE: int = 1  # single-window batches; sequences are length-diverse


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_loci_concat(loci_path: Path) -> dict[str, str]:
    """Read Locibase.json and return host_id -> concatenated K-locus protein sequence."""
    with loci_path.open() as f:
        raw = json.load(f)
    # Locibase.json is either {host_id: [seq, seq, ...]} or already concatenated.
    result: dict[str, str] = {}
    for host_id, value in raw.items():
        if isinstance(value, list):
            result[host_id] = "".join(v for v in value if isinstance(v, str))
        elif isinstance(value, str):
            result[host_id] = value
        else:
            raise TypeError(f"Unexpected type for host {host_id}: {type(value)}")
    return result


def iter_windows(seq: str, window_size: int, stride: int) -> list[str]:
    """Slice ``seq`` into overlapping windows (same semantics as src/features/sliding_window.py)."""
    n = len(seq)
    if n <= window_size:
        return [seq]
    windows: list[str] = []
    start = 0
    while start < n:
        end = min(start + window_size, n)
        windows.append(seq[start:end])
        if end == n:
            break
        start += stride
    return windows


@torch.no_grad()
def embed_window(tokenizer, model, device: torch.device, seq: str) -> np.ndarray:
    """Return a mean-pooled ESM-2 embedding (D,) for one window."""
    enc = tokenizer(
        seq,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=WINDOW_SIZE + 2,  # reserve CLS + EOS
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    hidden = model(**enc).last_hidden_state  # (1, L, D)
    attention = enc["attention_mask"].bool()
    # drop CLS and EOS
    special = torch.zeros_like(attention)
    special[:, 0] = True
    last_idx = attention.sum(dim=1) - 1
    for i, idx in enumerate(last_idx.tolist()):
        special[i, idx] = True
    mask = attention & ~special
    mask_f = mask.unsqueeze(-1).float()
    pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
    return pooled.detach().float().cpu().numpy().astype(np.float32)[0]


def main() -> None:
    # GPU policy self-check — fail loudly if GPU 2 is not visible.
    if not torch.cuda.is_available():
        print("[F1B] CUDA not available — aborting (GPU-only script)", flush=True)
        sys.exit(1)
    if torch.cuda.device_count() != 1:
        print(
            f"[F1B] Expected exactly 1 visible GPU (via CUDA_VISIBLE_DEVICES=2), "
            f"got {torch.cuda.device_count()} — aborting",
            flush=True,
        )
        sys.exit(1)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[F1B] device={device}, gpu_name={gpu_name}", flush=True)

    loci_path = HERE / "Locibase.json"
    if not loci_path.exists():
        print(f"[F1B] missing input: {loci_path}", flush=True)
        sys.exit(1)

    loci_concat = load_loci_concat(loci_path)
    print(f"[F1B] loaded {len(loci_concat)} hosts, median length = "
          f"{int(np.median([len(v) for v in loci_concat.values()]))} aa", flush=True)

    print(f"[F1B] loading model {MODEL_ID} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # fp16 to halve memory & speed up; ESM-2 650M mean pooling is not sensitive
    # to fp16 rounding in practice (we are averaging 1280-D vectors across
    # dozens of windows).
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model = model.to(device).eval()
    print(f"[F1B] model loaded in {time.time() - t0:.1f}s", flush=True)

    # --- truncated mode -----------------------------------------------------
    truncated_rows: list[dict] = []
    t0 = time.time()
    for i, (host_id, seq) in enumerate(sorted(loci_concat.items())):
        vec = embed_window(tokenizer, model, device, seq[:WINDOW_SIZE])
        truncated_rows.append({"host_id": host_id, "embedding": vec.tolist()})
        if (i + 1) % 20 == 0:
            print(f"[F1B] truncated {i + 1}/{len(loci_concat)}", flush=True)
    print(f"[F1B] truncated done in {time.time() - t0:.1f}s", flush=True)
    pd.DataFrame(truncated_rows).to_parquet(HERE / "host_esm650_truncated.parquet", index=False)

    # --- sliding mode -------------------------------------------------------
    sliding_rows: list[dict] = []
    t0 = time.time()
    total_windows = 0
    for i, (host_id, seq) in enumerate(sorted(loci_concat.items())):
        windows = iter_windows(seq, WINDOW_SIZE, STRIDE)
        vecs = [embed_window(tokenizer, model, device, w) for w in windows]
        total_windows += len(windows)
        pooled = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
        sliding_rows.append({"host_id": host_id, "embedding": pooled.tolist()})
        if (i + 1) % 20 == 0:
            print(f"[F1B] sliding {i + 1}/{len(loci_concat)} (cum windows={total_windows})",
                  flush=True)
    print(f"[F1B] sliding done in {time.time() - t0:.1f}s, total windows={total_windows}",
          flush=True)
    pd.DataFrame(sliding_rows).to_parquet(HERE / "host_esm650_sliding.parquet", index=False)

    meta = {
        "model_id": MODEL_ID,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "n_hosts": len(loci_concat),
        "total_sliding_windows": total_windows,
        "gpu_name": gpu_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "locibase_sha256": sha256(loci_path),
    }
    (HERE / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[F1B] wrote meta: {meta}", flush=True)


if __name__ == "__main__":
    main()
