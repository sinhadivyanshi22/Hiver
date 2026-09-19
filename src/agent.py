"""The AmazonHelp AI support agent.

Orchestrates:
  1. intent classification (embedding centroid + LR, self-trained),
  2. grounded reply drafting (retrieval of the brand's real historical reply),
  3. auto-handle vs escalate decision with a stated reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .classifier import IntentClassifier
from .decide import decide
from .embeddings import Embedder
from .pairs import Pair
from .reply import ReplyAgent


@dataclass
class AgentResponse:
    query: str
    intent: str
    confidence: float
    reply: str
    should_escalate: bool
    reason: str
    retrieval_score: float
    neighbor_redirect_fraction: float = 0.0
    retrieved_context: list[str] = field(default_factory=list)


class Agent:
    def __init__(self, embedder: Embedder, classifier: IntentClassifier, reply_agent: ReplyAgent):
        self.embedder = embedder
        self.classifier = classifier
        self.reply_agent = reply_agent

    def respond(self, query: str, k: int = 5) -> AgentResponse:
        intent, conf, _ = self.classifier.predict(query)
        draft = self.reply_agent.draft(query, k=k)
        decision = decide(intent, conf, query, draft.retrieval_score, draft.neighbor_redirect_fraction)
        retrieved_ctx = [self.reply_agent.pairs[i]["reference_reply"] for i in draft.retrieved_indices]
        return AgentResponse(
            query=query,
            intent=intent,
            confidence=conf,
            reply=draft.reply,
            should_escalate=decision.should_escalate,
            reason=decision.reason,
            retrieval_score=draft.retrieval_score,
            neighbor_redirect_fraction=draft.neighbor_redirect_fraction,
            retrieved_context=retrieved_ctx,
        )

    @classmethod
    def build(cls, pairs: list[Pair], pool_cache_name: str = "pool_queries") -> "Agent":
        embedder = Embedder()
        classifier = IntentClassifier(embedder)
        classifier.fit([p["query"] for p in pairs], pool_cache_name=pool_cache_name)
        reply_agent = ReplyAgent(embedder, pairs, pool_cache_name=pool_cache_name)
        return cls(embedder, classifier, reply_agent)
