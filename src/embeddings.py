"""Sentence embeddings for queries / replies.

Uses all-MiniLM-L6-v2 (a small, fast bi-encoder, ~22M params). Embeddings are
cached to disk so repeated runs are fast and reproducible.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
EMBED_CACHE = CACHE_DIR / "embeddings"
EMBED_CACHE.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(CACHE_DIR))
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str], cache_name: str | None = None) -> np.ndarray:
        """Embed a list of texts, caching the result under `cache_name`."""
        if cache_name:
            key = hashlib.md5(f"{self.model_name}:{cache_name}".encode()).hexdigest()
            cache_file = EMBED_CACHE / f"{cache_name}__{key}.npy"
            meta_file = cache_file.with_suffix(".meta.json")
            if cache_file.exists() and meta_file.exists():
                meta = json.loads(meta_file.read_text())
                if meta.get("n") == len(texts):
                    return np.load(cache_file)
        embs = self.model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        arr = np.asarray(embs, dtype=np.float32)
        if cache_name:
            np.save(cache_file, arr)
            meta_file.write_text(json.dumps({"n": len(texts)}))
        return arr


if __name__ == "__main__":
    e = Embedder()
    from .data import load_cached_or_fetch
    from .pairs import to_pairs

    convs = load_cached_or_fetch()
    pairs = to_pairs(convs)
    print("pairs:", len(pairs))
    sample = [p["query"] for p in pairs[:5]]
    embs = e.embed(sample, cache_name="selftest")
    print("emb shape:", embs.shape, "norm:", float(np.linalg.norm(embs[0])))
