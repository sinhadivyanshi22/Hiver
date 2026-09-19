import sys
sys.stdout.reconfigure(encoding="utf-8")
from src.data import load_cached_or_fetch
from src.pairs import to_pairs

pairs = to_pairs(load_cached_or_fetch())
golden_ids = set()
import csv
for r in csv.DictReader(open("data/golden_unlabeled.csv", encoding="utf-8")):
    golden_ids.add(int(r["idx"]))

ac = []
bill = []
ret = []
for i, p in enumerate(pairs):
    if i in golden_ids:
        continue
    q = p["query"].lower()
    if any(k in q for k in ["password is", "forgot my password", "can't log in", "cannot log in", "log in to my account", "account locked", "unable to log", "reset my password", "login"]) and not any(k in q for k in ["prime","deliver","delay","order"]):
        ac.append((i, p["query"][:170]))
    elif any(k in q for k in ["card was declin", "card declined", "payment method", "my card", "charged twice", "declined at checkout", "payment not working", "balance that", "recharged me", "billing"]):
        bill.append((i, p["query"][:170]))
    elif any(k in q for k in ["refund", "return this", "return my", "want a refund", "return policy", "return label"]):
        ret.append((i, p["query"][:170]))

print("=== ACCOUNT_ACCESS ===")
for i, q in ac[:12]:
    print(f"[{i}] {q}")
print("\n=== BILLING ===")
for i, q in bill[:12]:
    print(f"[{i}] {q}")
print("\n=== RETURN_REFUND ===")
for i, q in ret[:12]:
    print(f"[{i}] {q}")
