"""Data loading and parsing for the Customer-Support-on-Twitter dataset.

The source dataset is `TNE-AI/customer-support-on-twitter-conversation` on
HuggingFace (a conversation-structured mirror of the Kaggle
"Customer Support on Twitter" corpus by Thought Vector). Each row has a
`conversation` field containing multi-turn text prefixed with
``Customer:`` / ``Support:``, plus a `company` tag.

This module:
  * fetches AmazonHelp conversations (the chosen brand) via streaming,
  * parses each conversation into structured turns,
  * keeps only English-dominant conversations,
  * caches the result as data/amazonhelp_convs.jsonl so the whole pipeline
    runs reproducibly without re-downloading the 217MB source.

Citation: Thought Vector / Stuart Axelbrooke, "Customer Support on Twitter",
Kaggle (CC BY-NC-SA 4.0). Mirror: TNE-AI/customer-support-on-twitter-conversation
(HF Datasets).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import load_dataset

try:
    from langdetect import detect as _detect
except Exception:  # pragma: no cover - langdetect is a declared dependency
    _detect = None

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_PATH = DATA_DIR / "amazonhelp_convs.jsonl"
TARGET_BRAND = "AmazonHelp"
N_CONVERSATIONS = 8000

_TURN_RE = re.compile(r"^(Customer|Support):\s?(.*)$", re.MULTILINE)


def _to_turns(conversation: str) -> list[dict]:
    """Split a conversation string into a list of {speaker, text} turns."""
    text = conversation.replace("\r", "")
    # Drop leading empty content before the first marker.
    turns: list[dict] = []
    for m in _TURN_RE.finditer(text):
        turns.append({"speaker": m.group(1), "text": m.group(2).strip()})
    return turns


def _is_english_dominant(text: str) -> bool:
    """Keep only English-dominant conversations (clean intent/re-labeling)."""
    if not text.strip():
        return False
    if _detect is not None:
        try:
            return _detect(text[:2000]) == "en"
        except Exception:
            pass
    # Fallback heuristic: >=92% of non-space chars must be ASCII.
    ascii_chars = sum(c.isascii() for c in text if not c.isspace())
    total = sum(1 for c in text if not c.isspace())
    return total > 0 and (ascii_chars / total) >= 0.92


def fetch_conversations(brand: str = TARGET_BRAND, n: int = N_CONVERSATIONS) -> list[dict]:
    """Stream the dataset, filter to one brand, return parsed English convs."""
    ds = load_dataset(
        "TNE-AI/customer-support-on-twitter-conversation",
        split="train",
        streaming=True,
    )
    out: list[dict] = []
    seen: set[str] = set()
    for ex in ds:
        if ex.get("company") != brand:
            continue
        cid = ex["conversation_id"]
        if cid in seen:
            continue
        seen.add(cid)
        turns = _to_turns(ex["conversation"])
        if not turns:
            continue
        conv_text = ex["conversation"]
        if not _is_english_dominant(conv_text):
            continue
        # Keep conversations that begin with a Customer message (brand replies to
        # a real inbound request) and contain at least one Support turn.
        speakers = [t["speaker"] for t in turns]
        if "Support" not in speakers or speakers[0] != "Customer":
            continue
        out.append(
            {
                "conversation_id": cid,
                "company": brand,
                "turns": turns,
                "summary": ex.get("summary", ""),
            }
        )
        if len(out) >= n:
            break
    return out


def load_cached_or_fetch(cache: Path = CACHE_PATH, brand=TARGET_BRAND, n=N_CONVERSATIONS) -> list[dict]:
    """Return cached parsed conversations, fetching+writing cache if absent."""
    if cache.exists():
        with cache.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    cache.parent.mkdir(parents=True, exist_ok=True)
    convs = fetch_conversations(brand=brand, n=n)
    with cache.open("w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return convs


if __name__ == "__main__":
    convs = load_cached_or_fetch()
    print(f"Wrote {len(convs)} conversations to {CACHE_PATH}")
    for c in convs[:5]:
        print("===", c["conversation_id"])
        for t in c["turns"][:4]:
            print(f"  [{t['speaker']}] {t['text'][:160]}")
