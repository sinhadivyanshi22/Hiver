"""Derive a leakage-free escalation policy from the train pool's behaviour,
then evaluate it on the golden set."""
import csv, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from collections import defaultdict
from src.data import load_cached_or_fetch
from src.pairs import to_pairs
from src.agent import Agent
from src.reply import RED_FLAG_RE, is_redirect
from src.eval import load_golden
from src.eval import split_pairs

golden = load_golden()
train_pairs, _ = split_pairs(golden)

agent = Agent.build(train_pairs, pool_cache_name="trainpool_queries")
clf = agent.classifier

# Predict intents over the train pool and compute per-intent redirect rate.
pool_preds = clf.predict_bulk([p["query"] for p in train_pairs])
pool_intents = np.array([p[0] for p in pool_preds])
pool_redir = np.array([is_redirect(p["reference_reply"]) for p in train_pairs])

by_int = defaultdict(list)
for i, lab in enumerate(pool_intents):
    by_int[lab].append(pool_redir[i])
pool_esc_rate = {k: float(np.mean(v)) for k, v in by_int.items()}
print("Pool per-intent redirect rates:")
for k in sorted(pool_esc_rate):
    print(f"  {k:24s} {pool_esc_rate[k]:.2f}  (n={len(by_int[k])})")

def esc_rate_for(intent):
    return pool_esc_rate.get(intent, 0.5)

# Evaluate policy on golden using the AGENT (train-pool-trained)
rows = list(csv.DictReader(open("data/agent_outputs.csv", encoding="utf-8")))
et = np.array([r["escalate_true"] == "True" for r in rows])
pred = np.array([r["intent_pred"] for r in rows])
nr = np.array([float(r["neighbor_redirect_fraction"]) for r in rows])
conf = np.array([float(r["conf"]) for r in rows])
rs = np.array([float(r["retrieval_score"]) for r in rows])
rf = np.array([bool(RED_FLAG_RE.search(r["query"])) for r in rows])
er = np.array([esc_rate_for(i) for i in pred])

policies = {
    "neighbor>=0.5 OR redf": (nr >= 0.5) | rf,
    "pred-intent esc>=0.5": er >= 0.5,
    "pred-intent esc>=0.5 OR neighbor>=0.5 OR redflag OR conf<0.3 OR sim<0.35":
        (er >= 0.5) | (nr >= 0.5) | rf | (conf < 0.30) | (rs < 0.35),
    "pred-intent esc>=0.4 OR neighbor>=0.4 OR redflag OR conf<0.25 OR sim<0.35":
        (er >= 0.4) | (nr >= 0.4) | rf | (conf < 0.25) | (rs < 0.35),
    "pred-intent esc>=0.5 OR neighbor>=0.5 OR redflag (no conf/sim)":
        (er >= 0.5) | (nr >= 0.5) | rf,
}
from sklearn.metrics import accuracy_score, f1_score
print("\nGolden escalate rate:", et.mean())
for name, p in policies.items():
    print(f"  {name:55s} acc {accuracy_score(et,p):.3f} F1 {f1_score(et,p,zero_division=0):.3f} pred_esc {p.mean():.2f}")
