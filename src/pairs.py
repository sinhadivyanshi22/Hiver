"""Convert cached conversations into (customer query, brand reply) pairs.

For each conversation we take the LAST customer message as the "incoming query"
(the thing the agent must respond to) and the LAST support message as the
ground-truth / reference reply the brand actually gave. This gives one realistic
turn of the agent's job per conversation.
"""
from __future__ import annotations

from typing import TypedDict


class Pair(TypedDict):
    conversation_id: str
    brand: str
    query: str  # last Customer message
    reference_reply: str  # last Support message (ground truth)
    full_turns: list[dict]  # original turns for context
    query_index: int  # index of the query turn in full_turns


def to_pairs(convs: list[dict]) -> list[Pair]:
    pairs: list[Pair] = []
    for c in convs:
        turns = c["turns"]
        if len(turns) < 2:
            continue
        # Find the last Customer turn that is followed (later) by a Support turn.
        last_customer_idx = max(i for i, t in enumerate(turns) if t["speaker"] == "Customer")
        support_indices = [i for i, t in enumerate(turns) if t["speaker"] == "Support" and i > last_customer_idx]
        if not support_indices:
            # No support reply after the last customer message -> unresolved.
            continue
        reply_idx = support_indices[-1]
        pair: Pair = {
            "conversation_id": c["conversation_id"],
            "brand": c["company"],
            "query": turns[last_customer_idx]["text"],
            "reference_reply": turns[reply_idx]["text"],
            "full_turns": turns,
            "query_index": last_customer_idx,
        }
        pairs.append(pair)
    return pairs


def clean_text(s: str) -> str:
    """Lower-case + strip whitespace for lexical baselines."""
    return " ".join(s.lower().split())
