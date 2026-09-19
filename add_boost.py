import csv, json, sys
sys.stdout.reconfigure(encoding="utf-8")
from src.data import load_cached_or_fetch
from src.pairs import to_pairs

pairs = to_pairs(load_cached_or_fetch())

# (idx, intent) hand-verified labels for the stratified boost + the missing one.
BOOST = [
    (55, "account_access"), (1491, "account_access"), (2213, "account_access"),
    (2226, "account_access"), (2897, "account_access"), (2915, "account_access"),
    (2934, "account_access"), (2142, "account_access"), (1750, "account_access"),
    (453, "account_access"),
    (56, "billing_payment"), (825, "billing_payment"), (919, "billing_payment"),
    (1535, "billing_payment"), (3187, "billing_payment"), (3822, "billing_payment"),
    (5, "return_refund"), (52, "return_refund"), (132, "return_refund"),
    (88, "return_refund"),
    (556, "damage_defective"),
]

# load existing golden_intents and append
ints = json.load(open("data/golden_intents.json"))
for idx, intent in BOOST:
    ints[str(idx)] = intent
json.dump(ints, open("data/golden_intents.json", "w"), indent=2)

# append boosted rows to golden_unlabeled.csv
existing = [r["idx"] for r in csv.DictReader(open("data/golden_unlabeled.csv", encoding="utf-8"))]
new_indices = [str(idx) for idx, _ in BOOST if str(idx) not in existing]
with open("data/golden_unlabeled.csv", "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    for idx, q, ref in [(idx, pairs[idx]["query"], pairs[idx]["reference_reply"]) for idx in [b[0] for b in BOOST] if str(idx) in new_indices]:
        w.writerow([idx, q, ref])
print("added", len(new_indices), "rows to golden_unlabeled.csv")
print("total intent labels:", len(ints))
