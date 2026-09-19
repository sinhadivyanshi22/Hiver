"""Intent classification.

Pipeline (CPU-friendly, few-shot + self-training):
  1. Embed a hand-curated SEED set (canonical queries per intent) with
     all-MiniLM-L6-v2.
  2. Build per-intent centroids and pseudo-label the unlabeled conversation pool
     with high-confidence assignments (cosine > seed threshold).
  3. Train a LogisticRegression on seeds + pseudo-labels. This learns linear
     decision boundaries that handle overlapping intents better than nearest
     centroid, while needing no large hand-labelled training set.
  4. Predict intent + a calibrated confidence (max softmax probability).

Reproducibility: seeds and thresholds are fixed constants; the pool embeddings
are cached, so reruns are deterministic.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from .embeddings import Embedder
from .intents import INTENTS

POOL_EMB_CACHE = "pool_queries"
SEED_CACHE = "seeds"
PSEUDO_THRESH = 0.70  # cosine similarity to a seed centroid required to pseudo-label
CONF_THRESH = 0.45  # max LR prob below this => "uncertain" -> escalate


class IntentClassifier:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.clf_: Optional[LogisticRegression] = None
        self.labels_: list[str] = []
        self._pseudo_labels: Optional[np.ndarray] = None

    def fit(self, pool_queries: list[str], n_refine_iters: int = 3, verbose: bool = True, pool_cache_name: str = POOL_EMB_CACHE):
        self.labels_ = list(INTENTS.keys())
        label_index = {lab: i for i, lab in enumerate(self.labels_)}

        seed_queries: list[str] = []
        seed_labels: list[int] = []
        for iid, spec in INTENTS.items():
            for s in spec["seeds"]:
                seed_queries.append(s)
                seed_labels.append(label_index[iid])
        S = self.embedder.embed(seed_queries, cache_name=SEED_CACHE)
        P = self.embedder.embed(pool_queries, cache_name=pool_cache_name)

        cents = self._centroids(S, np.array(seed_labels))
        # 1) pseudo-label the pool with high-confidence centroid assignments
        sims = P @ cents.T
        best = sims.argmax(axis=1)
        best_val = sims.max(axis=1)
        confident = best_val > 0.65
        assign = np.where(confident, best, -1)
        if verbose:
            print(f"  centroid pseudo-labels: {int(confident.sum())}/{len(P)}")

        # 2) train LR on seeds + confident pseudo-labels
        X = np.vstack([S, P[confident]])
        y = np.concatenate([np.array(seed_labels), assign[confident]])
        self.clf_ = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)
        self.clf_.fit(X, y)

        # 3) LR self-training: expand labels with high-confidence LR predictions
        unassigned = np.where(assign == -1)[0]
        for it in range(3):
            if len(unassigned) == 0:
                break
            Pu = P[unassigned]
            probs = self.clf_.predict_proba(Pu)
            classes = list(self.clf_.classes_)
            full = np.zeros((len(Pu), len(self.labels_)), dtype=np.float32)
            for c, p in zip(classes, probs.T):
                full[:, c] = p
            top = full.max(axis=1)
            top_lab = full.argmax(axis=1)
            keep = top > 0.90
            kept = np.where(keep)[0]
            if len(kept) == 0:
                break
            assign[unassigned[kept]] = top_lab[kept]
            if verbose:
                print(f"  lr self-train iter {it}: added {len(kept)} (total labelled {int((assign >= 0).sum())})")
            unassigned = np.setdiff1d(unassigned, unassigned[kept])
            X2 = np.vstack([S, P[assign >= 0]])
            y2 = np.concatenate([np.array(seed_labels), assign[assign >= 0]])
            self.clf_ = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)
            self.clf_.fit(X2, y2)

        self._pseudo_labels = assign
        return self

    def _assign(self, P: np.ndarray, cents: np.ndarray) -> np.ndarray:
        return (P @ cents.T).argmax(axis=1)

    @staticmethod
    def _centroids(S: np.ndarray, labels: np.ndarray) -> np.ndarray:
        n = len(set(labels))
        cents = np.zeros((n, S.shape[1]), dtype=np.float32)
        for i, lab in enumerate(labels):
            cents[lab] += S[i]
        return IntentClassifier._l2norm(cents + 1e-6)

    @staticmethod
    def _l2norm(a: np.ndarray) -> np.ndarray:
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    def predict(self, query: str) -> tuple[str, float, np.ndarray]:
        qemb = self.embedder.embed([query])
        probs = self.clf_.predict_proba(qemb)[0]
        # align probs to self.labels_ order
        classes = list(self.clf_.classes_)
        full = np.zeros(len(self.labels_), dtype=np.float32)
        for c, p in zip(classes, probs):
            full[c] = p
        best = int(full.argmax())
        return self.labels_[best], float(full[best]), full

    def predict_bulk(self, queries: list[str]) -> list[tuple[str, float]]:
        Q = self.embedder.embed(queries)
        probs = self.clf_.predict_proba(Q)
        classes = list(self.clf_.classes_)
        out = []
        for row in probs:
            full = np.zeros(len(self.labels_), dtype=np.float32)
            for c, p in zip(classes, row):
                full[c] = p
            best = int(full.argmax())
            out.append((self.labels_[best], float(full[best])))
        return out


if __name__ == "__main__":
    from .data import load_cached_or_fetch
    from .pairs import to_pairs

    convs = load_cached_or_fetch()
    pairs = to_pairs(convs)
    pool = [p["query"] for p in pairs]
    emb = Embedder()
    clf = IntentClassifier(emb)
    clf.fit(pool)
    for q in [
        "Where is my order 123-456?",
        "I received a broken item.",
        "thanks",
        "can't log in",
        "my card was declined",
        "I want a refund",
        "is this product available in red",
    ]:
        intent, conf, _ = clf.predict(q)
        print(f"{q!r} -> {intent} (conf {conf:.2f})")
