import sys
sys.stdout.reconfigure(encoding="utf-8")
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
m = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
m.eval()
print("models loaded")

rub = "Rate the support reply 1-5 on relevance+helpfulness+grounding.\n\n"
exs = [
    ("Customer: where is order 123-456?", "Support: Sorry, we can not help with that over Twitter.", "1"),
    ("Customer: where is order 123-456?", "Support: Here is a photo of a banana.", "1"),
    ("Customer: where is order 123-456?", "Support: Your order shipped tomorrow. Track at https://t.co/x", "5"),
    ("Customer: where is order 123-456?", "Support: Thank you for contacting Amazon.", "3"),
    ("Customer: where is order 123-456?", "Support: Your order 123-456 is in transit, arriving tomorrow.", "5"),
    ("Customer: where is order 123-456?", "Support: We can help. Pls share phone number.", "2"),
]
for q, a, exp in exs:
    p = rub + f"Customer: {q}\nSupport: {a}\nRating:"
    inp = tok(p, return_tensors="pt", truncation=True, max_length=256)
    # explicit args
    with torch.no_grad():
        logits = m(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]).logits
    scores = {}
    for v in ["1", "2", "3", "4", "5"]:
        vi = tok(v, add_special_tokens=False).input_ids[0]
        scores[v] = float(logits[0, -1, vi])
    r = max(scores, key=scores.get)
    sd = {k: round(v, 2) for k, v in scores.items()}
    print(f"exp={exp} got={r} {sd}")
