import csv, sys
sys.stdout.reconfigure(encoding="utf-8")
rows = list(csv.DictReader(open("data/golden_unlabeled.csv", encoding="utf-8")))
batch = int(sys.argv[1]) if len(sys.argv) > 1 else 0
start = batch * 50
for r in rows[start:start+50]:
    print(f"[{r['idx']}] Q: {r['query']}")
    print(f"    A: {r['reference_reply']}")
    print()
