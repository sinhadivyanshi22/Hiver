import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
m = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
m.eval()

p = "Rate the support reply 1-5.\nCustomer: where is order 123-456?\nSupport: Your order is in transit, arriving tomorrow. Track it at https://t.co/x\nRating:"
inp = tok(p, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    o = m.generate(**inp, max_new_tokens=3, num_beams=1)
print("warmup:", tok.batch_decode(o, skip_special_tokens=True)[0].strip())

t0 = time.time()
with torch.no_grad():
    o = m.generate(**inp, max_new_tokens=4, num_beams=1, early_stopping=True)
dt = time.time() - t0
print("greedy:", tok.batch_decode(o, skip_special_tokens=True)[0].strip(), "(%5.2fs)" % dt)
