"""Baseline agents for comparison.

trivial: always classify as general_acknowledgement, always reply with a generic
         template, always escalate. (The "do nothing smart" floor.)
simple: keyword-rule intent + TF-IDF nearest-neighbour verbatim brand reply +
        threshold escalation. (The "obvious engineering" ceiling.)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .pairs import Pair
from .reply import neutralize

GENERIC_REPLY = "Thank you for contacting Amazon. We'll look into this and get back to you."


@dataclass
class BaselineResponse:
    intent: str
    reply: str
    should_escalate: bool
    reason: str


_INTENT_RULES = [
    ("return_refund", ["return", "refund", "returned", "exchange"]),
    ("account_access", ["log in", "login", "password", "account", "prime", "sign up", "signin"]),
    ("billing_payment", ["card", "payment", "charged", "declin", "billing", "balance", "pay"]),
    ("damage_defective", ["broken", "damage", "defect", "cracked", "used item", "wrong item", "missing"]),
    ("technical_issue", ["error", "crash", "load", "download", "not work", "bug", "frozen", "glitch", "checkout"]),
    ("order_status", ["order", "shipped", "ship", "deliver", "tracking", "arrive", "delay", "package"]),
    ("complaint_feedback", ["worst", "terrible", "pathetic", "frustrat", "disappoint", "scam", "fraud", "awful", "angry"]),
    ("product_inquiry", ["ship to", "available", "stock", "cost", "price", "carry", "ingredient", "have"]),
    ("general_acknowledgement", ["thank", "thx", "great", "ok", "thanks", "appreciate", "grateful"]),
]


def keyword_intent(query: str) -> str:
    q = query.lower()
    for intent, kws in _INTENT_RULES:
        if any(k in q for k in kws):
            return intent
    return "general_acknowledgement"


class TrivialBaseline:
    def __init__(self, pairs: list[Pair]):
        self.pairs = pairs

    def respond(self, query: str) -> BaselineResponse:
        return BaselineResponse(
            intent="general_acknowledgement",
            reply=GENERIC_REPLY,
            should_escalate=True,
            reason="trivial baseline escalates all messages to be safe",
        )


class SimpleBaseline:
    """Keyword intent + TF-IDF verbatim retrieval + threshold escalation."""

    def __init__(self, pairs: list[Pair]):
        self.pairs = pairs
        qs = [p["query"] for p in pairs]
        self.tfidf = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", min_df=2)
        self.mat = self.tfidf.fit_transform(qs)
        self.queries = qs

    def respond(self, query: str) -> BaselineResponse:
        intent = keyword_intent(query)
        qv = self.tfidf.transform([query])
        sims = cosine_similarity(qv, self.mat)[0]
        j = int(sims.argmax())
        sim = float(sims[j])
        reply = neutralize(self.pairs[j]["reference_reply"])
        sensitive = {"account_access", "billing_payment", "damage_defective",
                     "order_status", "return_refund", "complaint_feedback"}
        red = re.search(r"\b(fuck|shit|pathetic|worst|scam|fraud|sue|legal|urgent)\b", query.lower())
        if intent in sensitive or sim < 0.30 or red:
            esc, reason = True, "sensitive topic or weak retrieval match"
        else:
            esc, reason = False, "factual inquiry with a known reply"
        return BaselineResponse(intent=intent, reply=reply, should_escalate=esc, reason=reason)