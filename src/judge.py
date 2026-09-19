"""LLM-as-judge for reply quality.

Uses google/flan-t5-base (a small instruction-tuned encoder-decoder LLM) on CPU
with greedy decoding: ~0.5s per rating, deterministic. It rates a drafted reply
1-5 on a rubric (relevance, helpfulness, grounding, tone).

A small-model (flan-t5-small) variant was tested but was non-discriminative
(it rated nearly everything as 1 or 5 regardless of content); flan-t5-base is
the smallest locally-runnable model that discriminates reliably.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

RUBRIC = (
    "You are a strict but fair support-quality evaluator. Rate the Amazon "
    "support reply on a 1-5 scale.\n"
    "1 = irrelevant or hallucinated; 2 = barely relevant, generic; 3 = relevant "
    "but generic; 4 = relevant and helpful; 5 = highly relevant, helpful, and "
    "grounded in real Amazon practice.\n"
    "Consider: does the reply address the customer's actual problem, does it "
    "offer a real resolution or direction, and does it sound like a real "
    "Amazon support reply? Output ONLY an integer 1-5 and nothing else.\n\n"
    "Example:\nCustomer: where is my order?\n"
    "Support: Here is a photo of a banana.\nRating: 1\n\n"
    "Example:\nCustomer: where is my order?\n"
    "Support: Thank you for contacting Amazon.\nRating: 3\n\n"
    "Example:\nCustomer: where is my order?\n"
    "Support: Your order is in transit and will arrive tomorrow. Track it at https://t.co/x\nRating: 5\n\n"
    "Example:\nCustomer: where is my order?\n"
    "Support: I am a banana.\nRating: 1\n\n"
)


class Judge:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self.tok: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSeq2SeqLM] = None

    def _load(self):
        if self.tok is None:
            self.tok = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.eval()

    def rate(self, query: str, reply: str, context: str = "") -> int:
        """Return an integer 1-5."""
        self._load()
        prompt = RUBRIC
        if context:
            prompt += f"Context (similar past issues): {context}\n\n"
        prompt += f"Customer: {query}\nSupport: {reply}\nRating:"
        inp = self.tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=3, num_beams=1)
        text = self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        for ch in text:
            if ch in "12345":
                return int(ch)
        return 3
