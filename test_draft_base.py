import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.data import load_cached_or_fetch
from src.pairs import to_pairs
from src.embeddings import Embedder

tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
m = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
m.eval()

convs = load_cached_or_fetch()
pairs = to_pairs(convs)
emb = Embedder()
E = emb.embed([p["query"] for p in pairs], cache_name="pool_queries")

P = "Draft ONE concise reply. Ground it in the examples. Address the customer directly.\n"
def draft(q, retrieved):
    prompt = P
    for rq, ra in retrieved:
        prompt += f"Customer: {rq}\nSupport: {ra}\n\n"
    prompt += f"Customer: {q}\nSupport:"
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    t0 = time.time()
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=40, num_beams=1, no_repeat_ngram_size=2)
    dt = time.time() - t0
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip(), dt

rng = np.random.default_rng(3)
idx = rng.choice(len(pairs), size=6, replace=False)
for i in idx:
    q = pairs[i]["query"]; ref = pairs[i]["reference_reply"]
    sims = E @ E[i]; sims[i] = -2
    nn = np.argsort(-sims)[:3]
    retrieved = [(pairs[j]["query"], pairs[j]["reference_reply"]) for j in nn]
    d, dt = draft(q, retrieved)
    print("=== %.1fs ===" % dt)
    print("Q   :", q[:180])
    print("REF :", ref[:160])
    print("DRAFT:", d[:220])
    print()
