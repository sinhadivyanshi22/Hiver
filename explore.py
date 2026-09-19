"""Explore the customer-support-on-twitter-conversation dataset (streamed)."""
from collections import Counter
import textwrap

from datasets import load_dataset

ds = load_dataset("TNE-AI/customer-support-on-twitter-conversation", split="train", streaming=True)

# Count brands in a sample and find a target brand.
samples = []
counts = Counter()
target = "AmazonHelp"
TARGET = 4000

for i, ex in enumerate(ds):
    comp = ex["company"]
    counts[comp] += 1
    if comp == target:
        samples.append(ex)
        if len(samples) >= TARGET:
            break

print("=== TOP 20 BRANDS (in stream order, not final) ===")
for brand, c in counts.most_common(20):
    print(f"{c:6d}  {brand}")

print(f"\n=== Collected {len(samples)} {target} conversations ===")
print("\n=== SAMPLE PARSES ===")
for s in samples[:6]:
    conv = s["conversation"]
    print("---- conv_id", s["conversation_id"], "company", s["company"])
    print(textwrap.fill(conv, width=120))
    print()
