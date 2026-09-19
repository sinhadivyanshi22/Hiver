"""Sample a held-out golden set of conversations for hand-labelling.

The golden set is excluded from the classifier's self-training pool AND from the
reply-retrieval index, so the agent answers each golden query using only other
(historical) conversations. This is the only honest setup: at inference time the
agent does not get to retrieve the exact pair it is being asked about.
"""
from __future__ import annotations

import csv
import random

import numpy as np

from .data import load_cached_or_fetch
from .pairs import to_pairs
from .embeddings import Embedder

GOLDEN_PATH = "data/golden_unlabeled.csv"
N_GOLDEN = 200
SEED = 20240913

# Hard queries (low embedding similarity to any other pair) are likely to be
# edge cases, so we deliberately include extra-low-similarity ones.
N_HARD = 40


def build_golden():
    convs = load_cached_or_fetch()
    pairs = to_pairs(convs)
    rng = random.Random(SEED)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)

    # Take a small extra slice to pick hard cases from.
    probe = indices[:2000]
    emb = Embedder()
    E = emb.embed([pairs[i]["query"] for i in probe], cache_name="pool_queries")
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sims = En @ En.T
    np.fill_diagonal(sims, -2)
    nn_sim = sims.max(axis=1)
    order = np.argsort(nn_sim)
    hard_local = [probe[order[i]] for i in range(min(N_HARD, len(probe)))]

    rest = [i for i in indices if i not in set(hard_local)]
    rest_sample = rest[: N_GOLDEN - N_HARD]
    golden = hard_local + rest_sample
    rng.shuffle(golden)

    with open(GOLDEN_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "query", "reference_reply"])
        for i in golden:
            w.writerow([i, pairs[i]["query"], pairs[i]["reference_reply"]])
    print(f"Wrote {len(golden)} golden candidates to {GOLDEN_PATH}")
    print(f"train_pool size: {len(pairs) - len(golden)}")
    return golden


if __name__ == "__main__":
    build_golden()
