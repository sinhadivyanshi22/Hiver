import csv, json, sys
sys.stdout.reconfigure(encoding="utf-8")
from src.data import load_cached_or_fetch
from src.pairs import to_pairs
from src.agent import Agent
from src.golden import build_golden  # noqa: F401  (ensures golden cache exists)
from src.eval import load_golden

golden = load_golden()
gidx = {r["idx"] for r in golden}
pairs = to_pairs(load_cached_or_fetch())
train_pairs = [p for i, p in enumerate(pairs) if i not in gidx]
print("train_pool:", len(train_pairs), "golden:", len(golden))

agent = Agent.build(train_pairs, pool_cache_name="trainpool_queries")

rows_out = []
for r in golden:
    resp = agent.respond(r["query"])
    rows_out.append({
        "idx": r["idx"],
        "query": r["query"],
        "intent_true": r["intent"],
        "escalate_true": r["should_escalate"],
        "intent_pred": resp.intent,
        "conf": f"{resp.confidence:.3f}",
        "reply": resp.reply,
        "should_escalate_pred": resp.should_escalate,
        "reason": resp.reason,
        "retrieval_score": f"{resp.retrieval_score:.3f}",
        "neighbor_redirect_fraction": f"{resp.neighbor_redirect_fraction:.3f}",
    })

with open("data/agent_outputs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)
print("wrote data/agent_outputs.csv")

import numpy as np
sims = np.array([float(r["retrieval_score"]) for r in rows_out])
print("retrieval sim: min %.2f med %.2f max %.2f" % (sims.min(), np.median(sims), sims.max()))
