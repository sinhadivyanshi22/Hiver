import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

t0 = time.time()
tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
m = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
m.eval()
print("base load %.1fs" % (time.time() - t0), flush=True)

rub = (
    "Rate the Amazon support reply 1-5 on: (1) relevance to the customer's message, "
    "(2) helpfulness, (3) grounding in real Amazon support practice. "
    "Output ONLY a single integer from 1 to 5.\n\n"
    "Example:\nCustomer: where is order 1?\nSupport: Here is a photo of a banana.\nRating: 1\n\n"
    "Example:\nCustomer: where is order 1?\nSupport: Thank you for contacting Amazon.\nRating: 3\n\n"
    "Example:\nCustomer: where is order 1?\nSupport: Your order is in transit, arriving tomorrow. Track it at https://t.co/x\nRating: 5\n\n"
)
exs = [
    ("where is order 123-456?", "Sorry, we can not help with that over Twitter.", "1"),
    ("where is order 123-456?", "Here is a photo of a banana.", "1"),
    ("where is order 123-456?", "Your order shipped yesterday and arrives tomorrow. Track it here: https://t.co/x", "5"),
    ("where is order 123-456?", "Thank you for contacting Amazon.", "3"),
    ("where is order 123-456?", "We can help. Pls share phone number.", "2"),
    ("where is order 123-456?", "I am a banana and I like ships.", "1"),
]
for q, a, exp in exs:
    p = rub + f"Customer: {q}\nSupport: {a}\nRating:"
    inp = tok(p, return_tensors="pt", truncation=True, max_length=320)
    t0 = time.time()
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=3, num_beams=1)
    dt = time.time() - t0
    r = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
    print(f"exp={exp} got={r} ({dt:.2f}s)", flush=True)
