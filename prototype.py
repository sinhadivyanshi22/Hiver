"""Quick prototype test of the retrieval+generation drafter."""
import time
import numpy as np

from src.data import load_cached_or_fetch
from src.pairs import to_pairs
from src.embeddings import Embedder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def draft(query, retrieved):
    """Draft a grounded reply to `query` using retrieved (q, a) examples."""
    prompt = (
        "Amazon customer support. Reply concisely, politely, and only with what "
        "the customer needs. Ground the reply in the examples below.\n"
    )
    for q, a in retrieved:
        prompt += f"Customer: {q}\nSupport: {a}\n\n"
    prompt += f"Customer: {query}\nSupport:"
    return prompt


def main():
    convs = load_cached_or_fetch()
    pairs = to_pairs(convs)
    emb = Embedder()
    E = emb.embed([p["query"] for p in pairs], cache_name="pool_queries")
    # also embed reference replies for a retrieval index over queries
    import numpy as np
    index = E

    tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model.eval()

    rng = np.random.default_rng(1)
    test_idx = rng.choice(len(pairs), size=5, replace=False)
    for i in test_idx:
        q = pairs[i]["query"]
        # nearest neighbours among the pool (exclude itself)
        sims = index @ E[i]
        sims[i] = -2
        nn = np.argsort(-sims)[:3]
        retrieved = [(pairs[j]["query"], pairs[j]["reference_reply"]) for j in nn]

        prompt = draft(q, retrieved)
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        # time generation
        t0 = time.time()
        out = model.generate(**inp, max_new_tokens=64, num_beams=2, early_stopping=True)
        dt = time.time() - t0
        rep = tok.batch_decode(out, skip_special_tokens=True)[0]
        print("=== %d (%.1fs) ===" % (i, dt))
        print("Q :", q[:160])
        print("NN:", [pairs[j]["query"][:60] for j in nn])
        print("REF:", pairs[i]["reference_reply"][:160])
        print("DRAFT:", rep[:200])
        print()


if __name__ == "__main__":
    main()
