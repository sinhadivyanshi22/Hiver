"""Decide whether to auto-handle or escalate, with a stated reason.

Policy (grounded in the brand's real behaviour on similar past queries):
  The brand resolves a query inline on Twitter only when it can do so WITHOUT
  needing account / order / billing details and WITHOUT human empathy. We mirror
  that by looking at the top-k nearest historical replies:
    * if similar past queries were usually redirected to a private channel/form
      by the brand -> escalate (private details needed);
    * if angry / legal / urgent language is present -> escalate (human
      de-escalation / legal risk);
    * if the intent classifier is uncertain -> escalate (unclear intent);
    * if there is no close historical precedent -> escalate (no known pattern);
    * otherwise -> auto-handle with the grounded draft.
"""
from __future__ import annotations

from dataclasses import dataclass

from .intents import INTENTS
from .reply import RED_FLAG_RE

CONF_THRESH = 0.30
REDIRECT_FRAC_THRESH = 0.50  # majority of nearest neighbours were brand-redirected
RETRIEVAL_THRESH = 0.35


@dataclass
class Decision:
    should_escalate: bool
    reason: str
    confidence: float
    intent: str


def decide(intent: str, confidence: float, query: str, retrieval_score: float,
           neighbor_redirect_fraction: float = 0.0) -> Decision:
    reasons: list[str] = []

    # 1. explicit anger / legal / safety language -> escalate
    if RED_FLAG_RE.search(query):
        reasons.append("red-flag language needs human / legal handling")

    # 2. behaviour-grounded escalation: similar past queries were redirected
    if neighbor_redirect_fraction >= REDIRECT_FRAC_THRESH:
        reasons.append(
            f"brand typically redirects similar queries (NN redirect {neighbor_redirect_fraction:.0%})"
        )

    # 3. intent that the brand almost always privatises
    if intent in ("account_access", "billing_payment", "damage_defective"):
        reasons.append(INTENTS[intent]["name"] + " requires private/account details")

    # 4. classifier uncertainty
    if confidence < CONF_THRESH:
        reasons.append(f"low intent confidence ({confidence:.2f})")

    # 5. no close historical precedent
    if retrieval_score < RETRIEVAL_THRESH:
        reasons.append(f"no close historical reply (sim {retrieval_score:.2f})")

    should_escalate = bool(reasons)
    if should_escalate:
        reason = "; ".join(reasons)
    else:
        reason = "clear query with a known inline resolution"
    return Decision(
        should_escalate=should_escalate,
        reason=reason,
        confidence=confidence,
        intent=intent,
    )
