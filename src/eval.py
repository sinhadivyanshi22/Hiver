"""Evaluation harness for the AmazonHelp agent.

Metrics per system (agent / trivial / simple) on the held-out golden set:
  * intent : accuracy + macro-F1
  * escalate: accuracy, precision, recall, F1 (escalate=True is the positive class)
  * reply  : ROUGE-L (surface overlap vs the brand's real reply) + retrieval sim
  * reply quality: LLM-as-judge (flan-t5-base, 1-5) rating of each draft, plus the
    real reference reply as an upper-bound anchor.

All judge scores are cached on disk so reruns are fast.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from rouge_score import rouge_scorer

from .agent import Agent
from .baselines import SimpleBaseline, TrivialBaseline
from .data import load_cached_or_fetch
from .judge import Judge
from .pairs import to_pairs

DATA = Path(__file__).resolve().parent.parent / "data"
GOLDEN_CSV = DATA / "golden_labeled.csv"
TRAIN_CACHE = "trainpool_queries"
JUDGE_CACHE = DATA / "judge_cache.json"


def load_golden() -> list[dict]:
    rows = list(csv.DictReader(GOLDEN_CSV.open(encoding="utf-8")))
    for r in rows:
        r["idx"] = int(r["idx"])
        r["should_escalate"] = r["should_escalate"] == "True"
        r["escalate_reason"] = r.get("escalate_reason", "")
    return rows


def split_pairs(golden_rows: list[dict]):
    """Return (train_pairs, golden_rows_with_query+reference)."""
    convs = load_cached_or_fetch()
    pairs = to_pairs(convs)
    gidx = {r["idx"] for r in golden_rows}
    train_pairs = [p for i, p in enumerate(pairs) if i not in gidx]
    return train_pairs, pairs


class CachedJudge:
    def __init__(self, judge: Judge):
        self.judge = judge
        self.cache: dict = {}
        if JUDGE_CACHE.exists():
            self.cache = json.loads(JUDGE_CACHE.read_text())

    def _key(self, query: str, reply: str) -> str:
        return hashlib.md5(f"{query}|{reply}".encode()).hexdigest()

    def rate(self, query: str, reply: str, context: str = "") -> int:
        k = self._key(query, reply)
        if k in self.cache:
            return int(self.cache[k])
        r = self.judge.rate(query, reply, context)
        self.cache[k] = r
        # flush
        JUDGE_CACHE.write_text(json.dumps(self.cache))
        return r


def _responder(system, query: str, k: int = 5) -> dict:
    out = system.respond(query)
    return {
        "intent": out.intent,
        "reply": out.reply,
        "should_escalate": out.should_escalate,
        "reason": getattr(out, "reason", ""),
        "retrieval_score": getattr(out, "retrieval_score", None),
    }


def macro_f1_intent(y_true, y_pred, labels) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def run_eval(train_pairs, golden_rows, judge: CachedJudge, n_rated: int = 220) -> dict:
    agent = Agent.build(train_pairs)
    trivial = TrivialBaseline(train_pairs)
    simple = SimpleBaseline(train_pairs)

    queries = [r["query"] for r in golden_rows]
    refs = [r["reference_reply"] for r in golden_rows]
    intents_true = [r["intent"] for r in golden_rows]
    esc_true = [r["should_escalate"] for r in golden_rows]

    systems = {"agent": agent, "trivial": trivial, "simple": simple}
    results = {name: [] for name in systems}
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for i, q in enumerate(queries):
        for name, sys_ in systems.items():
            out = _responder(sys_, q)
            r = rouge.score(refs[i], out["reply"])["rougeL"].fmeasure
            out["rougeL"] = float(r)
            out["judge_rating"] = None
            results[name].append(out)

    # LLM judge ratings (cached) for drafts + reference upper bound
    ref_ratings = []
    for i in range(n_rated):
        for name in systems:
            results[name][i]["judge_rating"] = judge.rate(queries[i], results[name][i]["reply"])
        ref_ratings.append(judge.rate(queries[i], refs[i]))

    summary = {}
    labels = list(intent_labels_ordered())
    for name in systems:
        rs = results[name]
        int_pred = [r["intent"] for r in rs]
        esc_pred = [r["should_escalate"] for r in rs]
        judge_ratings = [r["judge_rating"] for r in rs]
        rouge_ls = [r["rougeL"] for r in rs]
        summary[name] = {
            "intent_acc": float(np.mean([p == t for p, t in zip(int_pred, intents_true)])),
            "intent_macro_f1": macro_f1_intent(intents_true, int_pred, labels),
            "esc_acc": float(np.mean([p == t for p, t in zip(esc_pred, esc_true)])),
            "esc_f1": _binary_f1(esc_true, esc_pred),
            "judge_mean": float(np.mean(judge_ratings)),
            "rougeL_mean": float(np.mean(rouge_ls)),
        }
    summary["__reference__"] = {"judge_mean": float(np.mean(ref_ratings))}
    return summary, results


def _binary_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, pos_label=True, zero_division=0))


def intent_labels_ordered():
    from .intents import INTENTS

    return list(INTENTS.keys())


def main():
    golden = load_golden()
    train_pairs, _ = split_pairs(golden)
    j = Judge()
    cj = CachedJudge(j)
    summary, results = run_eval(train_pairs, golden, cj)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
