import csv, json, sys
sys.stdout.reconfigure(encoding="utf-8")
from src.data import load_cached_or_fetch
from src.pairs import to_pairs

ints = json.load(open("data/golden_intents.json"))
rows = list(csv.DictReader(open("data/golden_unlabeled.csv", encoding="utf-8")))
missing = [r["idx"] for r in rows if r["idx"] not in ints]
print("MISSING idx:", missing, "->")
for r in rows:
    if r["idx"] in missing:
        print("  Q:", r["query"][:200])
        print("  A:", r["reference_reply"][:160])

# find candidate account_access / billing_payment / return_refund queries in pool
convs = load_cached_or_fetch()
pairs = to_pairs(convs)
golden_idx = set(r["idx"] for r in rows)
pool = [p for p in pairs if str(pairs.index(p)) not in golden_idx]
# we lost idx mapping; rebuild with indices
pool = [(i, pairs[i]) for i in range(len(pairs)) if str(i) not in set(ints.keys())]
print("\n=== account_access candidates ===")
c = 0
for i, p in pool:
    q = p["query"].lower()
    if any(k in q for k in ["log in","login","password","account locked","prime","sign up","signin","unable to log","access my account","reset my password","forgot"]) and "login" not in q or "cannot log" in q:
        print(f"[{i}] {p['query'][:180]}")
        c += 1
        if c >= 12: break
print("\n=== billing_payment candidates ===")
c = 0
for i, p in pool:
    q = p["query"].lower()
    if any(k in q for k in ["card declined","charged","payment method","balance","declined","billing","price","charges","refund"]):
        print(f"[{i}] {p['query'][:180]}")
        c += 1
        if c >= 12: break
