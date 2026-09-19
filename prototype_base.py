"""Compare retrieval vs flan-t5-base generation drafting."""
import time
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

from src.data import load_cached_or_fetch
from src.pairs import to_pairs
from src.embeddings import Embedder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def make_prompt(query, retrieved, ref_hint=None):
    p = (
        "Amazon customer support reply. Be concise, polite, grounded, and only "
        "say what the customer needs. Mirror the brand's helpful tone.\n"
    )
    for q, a in retrieved:
        p += f"Customer: {q}\nSupport: {a}\n\n"
    p += f"Customer: {query}\nSupport:"
    return p


def main():
    convs = load_cached_or_fetch()
    pairs = to_pairs(convs)
    emb = Embedder()
    E = emb.embed([p["query"] for p in pairs], cache_name="pool_queries")
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model.eval()

    rng = np.random.default_rng(7)
    idx = rng.choice(len(pairs), size=6, replace=False)
    for i in idx:
        q = pairs[i]["query"]
        sims = E @ E[i]; sims[i] = -2
        nn = np.argsort(-sims)[:3]
        retrieved = [(pairs[j]["query"], pairs[j]["reference_reply"]) for j in nn]
        ref = pairs[i]["reference_reply"]

        prompt = make_prompt(q, retrieved)
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        t0 = time.time()
        out = model.generate(**inp, max_new_tokens=70, num_beams=3, early_stopping=True, no_repeat_ngram_size=2)
        dt = time.time() - t0
        gen = tok.batch_decode(out, skip_special_tokens=True)[0]
        r1 = pairs[nn[0]]["reference_reply"]
        print("=== %d (%.1fs) ===" % (i, dt))
        print("Q     :", q[:180])
        print("REF   :", ref[:180])
        print("NN1   :", r1[:180])
        print("GEN   :", gen[:220])
        print()


if __name__ == "__main__":
    main()
