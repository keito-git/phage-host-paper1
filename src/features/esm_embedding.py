# MIT License. See LICENSE in repository root.
"""ESM-2 embedding helpers.

Mean-pooled ESM-2 embeddings via Hugging Face ``transformers``.  We picked
``transformers`` over ``fair-esm`` because the upstream pickle hosted by the
former is fetched from ``huggingface.co`` where the SSL chain is reliably
trusted by ``certifi`` on macOS (``fair-esm`` reaches ``dl.fbaipublicfiles``
which occasionally fails TLS on stock Python).

The embeddings are cached under ``data/cache/esm_embeddings/`` keyed by the
SHA-1 of the (truncated) sequence plus the model name, so re-running an
experiment with the same model is close to free.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.config import CACHE_DIR

# Map the short fair-esm style names to the canonical Hugging Face IDs.
_HF_MODEL_MAP: dict[str, str] = {
    "esm2_t6_8M_UR50D": "facebook/esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D": "facebook/esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D": "facebook/esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D": "facebook/esm2_t33_650M_UR50D",
}


def _resolve_hf_id(name: str) -> str:
    return _HF_MODEL_MAP.get(name, name)


def pick_device() -> torch.device:
    """Return the best available device (mps > cuda > cpu)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class ESMEmbedder:
    """Run mean-pooled ESM-2 embedding on a list of sequences.

    Example
    -------
    >>> embedder = ESMEmbedder(model_name="esm2_t12_35M_UR50D")
    >>> matrix = embedder.embed_many({"phage1": "MSEQ...", "phage2": "MSEQ..."})
    """

    model_name: str = "esm2_t12_35M_UR50D"
    max_length: int = 1022  # ESM-2 position embedding upper bound
    batch_size: int = 4
    device: torch.device | None = None
    cache_dir: Path = CACHE_DIR / "esm_embeddings"

    def __post_init__(self) -> None:
        self.device = self.device or pick_device()
        hf_id = _resolve_hf_id(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        # The ESM-2 HF model is a masked-LM head + backbone; we only need the
        # last hidden state, so ``AutoModel`` (which returns ``BaseModelOutput``
        # with ``last_hidden_state``) is sufficient and slightly lighter.
        self.model = AutoModel.from_pretrained(hf_id).to(self.device).eval()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, seq: str) -> Path:
        digest = hashlib.sha1(seq.encode()).hexdigest()
        return self.cache_dir / f"{self.model_name}__{digest}.npy"

    def _load_cached(self, seq: str) -> np.ndarray | None:
        path = self._cache_key(seq)
        if path.exists():
            return np.load(path)
        return None

    def _store_cached(self, seq: str, vec: np.ndarray) -> None:
        np.save(self._cache_key(seq), vec)

    @torch.no_grad()
    def _embed_batch(self, items: list[tuple[str, str]]) -> dict[str, np.ndarray]:
        """Embed a batch of ``(id, seq)`` tuples that were NOT in cache."""
        names = [n for n, _ in items]
        seqs = [s for _, s in items]
        enc = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length + 2,  # reserve CLS / EOS
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.model(**enc).last_hidden_state  # (B, L, D)
        # Mean-pool over *attended* positions only, dropping special tokens.
        attention = enc["attention_mask"].bool()
        # Mark CLS and EOS positions as non-attended for pooling.
        special = torch.zeros_like(attention)
        special[:, 0] = True
        # EOS is at the last attended index of each row.
        last_idx = attention.sum(dim=1) - 1
        for i, idx in enumerate(last_idx.tolist()):
            special[i, idx] = True
        mask = attention & ~special
        mask_f = mask.unsqueeze(-1).float()
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        arr = pooled.detach().float().cpu().numpy().astype(np.float32)
        return {name: arr[i] for i, name in enumerate(names)}

    def embed_many(self, sequences: dict[str, str]) -> dict[str, np.ndarray]:
        """Embed all sequences, populating the on-disk cache as a side effect."""
        pending: list[tuple[str, str]] = []
        cached: dict[str, np.ndarray] = {}

        for name, seq in sequences.items():
            truncated = seq[: self.max_length]
            hit = self._load_cached(truncated)
            if hit is None:
                pending.append((name, truncated))
            else:
                cached[name] = hit

        for i in tqdm(range(0, len(pending), self.batch_size), desc=f"ESM {self.model_name}"):
            batch = pending[i : i + self.batch_size]
            embeddings = self._embed_batch(batch)
            for name, vec in embeddings.items():
                cached[name] = vec
                # reconstruct the canonicalised sequence for caching
                seq_for_id = dict(batch)[name]
                self._store_cached(seq_for_id, vec)
        return cached
