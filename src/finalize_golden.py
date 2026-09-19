"""Finalize the golden set: combine hand-assigned intents with objective,
reference-grounded escalation labels.

Escalation ground-truth reflects what the brand ACTUALLY did in the real
support thread (ground-truth behaviour):
  * escalate = the brand redirected the customer to a private channel / form /
    phone / DM (it cannot resolve account/order/billing issues over raw Twitter),
    OR the customer used red-flag / angry language, OR the message is a complaint
    that needs human de-escalation;
  * auto-handle = the brand answered / probed inline on Twitter.

This makes the escalate label objective and brand-grounded. Intent labels are
hand-assigned by the author from the query text (see golden_intents.json).
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .intents import INTENTS

INTENT_FILE = Path(__file__).resolve().parent.parent / "data" / "golden_intents.json"
IN_CSV = Path(__file__).resolve().parent.parent / "data" / "golden_unlabeled.csv"
OUT_CSV = Path(__file__).resolve().parent.parent / "data" / "golden_labeled.csv"

REDIRECT_CUES = re.compile(
    r"contact us (here|via|on)|reach (out|us|our)|drop (in|your) your?\s*details|"
    r"share (your|their) details|phone or chat|by phone|call us|reach us by|"
    r"send us|report this to our team|fill in your details|visit the link|"
    r"join our (customer )?service|get in touch|we'll get in touch|write to us|"
    r"report that here|drop your details|connect (our|with) our? support|"
    r"reach (out|our) support|connect with our support|look into this further|"
    r"let us look into|we'll have a closer look|let us know here|share the details|"
    r"send your details|get in touch with us",
    re.IGNORECASE,
)
RED_FLAG_RE = re.compile(
    r"\b(fuck|shit|bitch|pathetic|sue|legal|lawsuit|threat|scum|bastard|scam|fraud|"
    r"death threat|unsafe|kill|worthless|trash)\b",
    re.IGNORECASE,
)


def derive_escalate(query: str, reply: str, intent: str) -> tuple[bool, str]:
    if RED_FLAG_RE.search(query):
        return True, "red-flag / angry language"
    if REDIRECT_CUES.search(reply):
        return True, "brand redirected to private channel / form / phone"
    if intent == "complaint_feedback":
        return True, "frustrated complaint benefits from human de-escalation"
    return False, "brand resolved inline (direct answer or probe)"


def finalize() -> int:
    intents: dict[str, str] = json.loads(INTENT_FILE.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(IN_CSV.open(encoding="utf-8")))
    out = []
    missing = 0
    from collections import Counter
    intent_dist = Counter()
    esc_dist = Counter()
    for r in rows:
        idx = r["idx"]
        intent = intents.get(idx)
        if intent is None:
            missing += 1
            continue
        esc, reason = derive_escalate(r["query"], r["reference_reply"], intent)
        intent_dist[intent] += 1
        esc_dist[esc] += 1
        out.append(
            {
                "idx": idx,
                "query": r["query"],
                "reference_reply": r["reference_reply"],
                "intent": intent,
                "should_escalate": esc,
                "escalate_reason": reason,
            }
        )
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["idx", "query", "reference_reply", "intent", "should_escalate", "escalate_reason"]
        )
        w.writeheader()
        w.writerows(out)
    print(f"Wrote {len(out)} labeled golden examples to {OUT_CSV}")
    print(f"Missing intent labels: {missing}")
    print("Intent distribution:", dict(intent_dist))
    print("Escalate dist (True/False):", dict(esc_dist))
    return len(out)


if __name__ == "__main__":
    finalize()
