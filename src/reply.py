"""Grounded reply drafting for the AmazonHelp agent.

Strategy (chosen empirically — see REPORT "what we tried"): for a template-heavy
brand like Amazon, the strongest, most *trustworthy* draft is the brand's OWN
historical reply to the semantically closest resolved query, neutralised of
customer-specific PII. A small generative LM (flan-t5) was tested but produced
generic, hallucinated replies that scored worse on the judge, so retrieval is
the headline drafter. A `generate` mode is kept for the ablation in the report.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import Embedder
from .pairs import Pair

ORDER_RE = re.compile(r"\b\d{3}-\d{7}-\d{7}\b")
AGENT_TAG_RE = re.compile(r"\^([A-Z]{1,3})\b")
HANDLE_RE = re.compile(r"@(\w{3,16})")

# Cues that the brand redirected the customer to a *private* channel (a form,
# phone, DM, or a personal-data-collecting link). This is the brand's real
# "escalated to a human/private channel" behaviour, used to ground decisions.
REDIRECT_CUES = re.compile(
    r"contact us (here|via|on)|reach (out|us|our)( to|)|drop (in|your) your?\s*details|"
    r"share (your|their) details|phone or chat|by phone|call us|reach us by|"
    r"send us|report this to our team|fill in your details|visit the link|"
    r"join our (customer )?service|get in touch|we'll get in touch|write to us|"
    r"report that here|drop your details|connect (our|with) our? support|"
    r"reach (out|our) support|connect with our support|look into this further|"
    r"let us look into|we'll have a closer look|please reach out|reach us here|"
    r"use the link|via phone|by chat|dm us",
    re.IGNORECASE,
)
# Strong anger / legal / safety language that should escalate regardless.
RED_FLAG_RE = re.compile(
    r"\b(fuck|shit|bitch|pathetic|scum|bastard|lawyer|legal|lawsuit|sue|threat|"
    r"death threat|unsafe|sueing)\b",
    re.IGNORECASE,
)


def is_redirect(reply: str) -> bool:
    """Did the brand actually escalate this real query to a private channel?"""
    return bool(REDIRECT_CUES.search(reply))


def neutralize(reply: str) -> str:
    """Remove customer-specific PII while keeping grounding (URLs, structure)."""
    r = ORDER_RE.sub("ORDER_NUMBER", reply)
    r = AGENT_TAG_RE.sub("^Support", r)
    return r


@dataclass
class DraftResult:
    reply: str
    mode: str
    retrieved_indices: list[int]
    retrieval_score: float  # cosine sim of nearest neighbour
    neighbor_redirect_fraction: float = 0.0  # fraction of NN replies the brand escalated


class EmbedIndex:
    """Brute-force (cosine) index over pool query embeddings."""

    def __init__(self, query_embs: np.ndarray, redirect_flags: np.ndarray):
        self.emb = query_embs
        norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
        self.emb = query_embs / np.maximum(norms, 1e-9)
        self.redirect_flags = redirect_flags  # (n,) bool

    def search(self, q: str, embedder: Embedder, k: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qv = embedder.embed([q])[0]
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        sims = self.emb @ qv
        top = np.argsort(-sims)[:k]
        return top, sims[top], self.redirect_flags[top]


class ReplyAgent:
    def __init__(self, embedder: Embedder, pairs: list[Pair], pool_cache_name: str = "pool_queries"):
        self.embedder = embedder
        self.pairs = pairs
        self.index = EmbedIndex(
            embedder.embed([p["query"] for p in pairs], cache_name=pool_cache_name),
            np.array([is_redirect(p["reference_reply"]) for p in pairs]),
        )
        self._tfidf: Optional[TfidfVectorizer] = None
        self._tfidf_mat = None

    def retrieve(self, query: str, k: int = 5) -> tuple[list[Pair], np.ndarray, np.ndarray]:
        idx, sims, redir = self.index.search(query, self.embedder, k)
        return [self.pairs[i] for i in idx], sims, redir

    def _tfidf_index(self):
        if self._tfidf_mat is None:
            qs = [p["query"] for p in self.pairs]
            self._tfidf = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", min_df=2)
            self._tfidf_mat = self._tfidf.fit_transform(qs)
        return self._tfidf_mat

    def retrieve_tfidf(self, query: str, k: int = 5) -> tuple[list[Pair], np.ndarray]:
        mat = self._tfidf_index()
        qv = self._tfidf.transform([query])
        sims = cosine_similarity(qv, mat)[0]
        top = np.argsort(-sims)[:k]
        return [self.pairs[i] for i in top], sims[top]

    def draft(self, query: str, k: int = 5) -> DraftResult:
        """Headline drafter: grounded retrieval of the closest real brand reply."""
        idx, sims, redir = self.index.search(query, self.embedder, k)
        retrieved = [self.pairs[i] for i in idx]
        nn_reply = retrieved[0]["reference_reply"]
        draft_reply = neutralize(nn_reply)
        return DraftResult(
            reply=draft_reply,
            mode="retrieve",
            retrieved_indices=[int(i) for i in idx],
            retrieval_score=float(sims[0]),
            neighbor_redirect_fraction=float(redir.mean()),
        )

    def draft_generate(self, query: str, k: int = 5) -> DraftResult:
        """Experimental generative drafter (flan-t5-base), for the ablation."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        model.eval()
        idx, _, _ = self.index.search(query, self.embedder, k)
        retrieved = [self.pairs[i] for i in idx]
        prompt = (
            "Draft a concise, polite Amazon support reply grounded in the examples. "
            "Address only what the customer needs.\n\n"
        )
        for r in retrieved:
            prompt += f"Customer: {r['query']}\nSupport: {r['reference_reply']}\n\n"
        prompt += f"Customer: {query}\nSupport:"
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=60, num_beams=1, no_repeat_ngram_size=2)
        text = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        return DraftResult(
            reply=text, mode="generate",
            retrieved_indices=[int(i) for i in idx],
            retrieval_score=float(0),
        )
