import sys
sys.stdout.reconfigure(encoding="utf-8")
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
m = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
m.eval()
print("models loaded", flush=True)

rub = (
    "Rate the support reply 1-5 on relevance+helpfulness+grounding. Rating: "
)
exs = [
    ("where is order 123-456?", "Sorry, we can not help with that over Twitter.", "1"),
    ("where is order 123-456?", "Here is a photo of a banana.", "1"),
    ("where is order 123-456?", "Your order shipped tomorrow. Track at https://t.co/x", "5"),
    ("where is order 123-456?", "Thank you for contacting Amazon.", "3"),
    ("where is order 123-456?", "We can help. Pls share phone number.", "2"),
    ("where is order 123-456?", "I am a banana.", "1"),
]
for q, a, exp in exs:
    p = rub + f"Customer: {q}\nSupport: {a}\nRating: "
    inp = tok(p, return_tensors="pt", truncation=True, max_length=256)
    ids = inp["input_ids"]
    attn = inp["attention_mask"]
    # T5 decoder starts with pad token (id=0) which is also BOS.
    dec = torch.tensor([[tok.pad_token_id]])
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=attn, decoder_input_ids=dec)
    logits = out.logits[0, -1]
    scores = {v: float(logits[tok(v, add_special_tokens=False).input_ids[0]]) for v in ["1", "2", "3", "4", "5"]}
    r = max(scores, key=scores.get)
    sd = {k: round(v, 2) for k, v in scores.items()}
    print(f"exp={exp} got={r} {sd}")
