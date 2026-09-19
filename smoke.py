import sys
sys.stdout.reconfigure(encoding="utf-8")
from src.data import load_cached_or_fetch
from src.pairs import to_pairs
from src.agent import Agent
from src.baselines import TrivialBaseline, SimpleBaseline

convs = load_cached_or_fetch()
pairs = to_pairs(convs)
agent = Agent.build(pairs)
trivial = TrivialBaseline(pairs)
simple = SimpleBaseline(pairs)

tests = [
    "Where is my order 123-456?",
    "I received a broken item.",
    "My card was declined at checkout.",
    "Can you ship this to Canada?",
    "Thanks for the help.",
    "I want a refund for this.",
    "Your service is the worst. Fix this now.",
    "is this product available in red?",
]
for q in tests:
    a = agent.respond(q)
    s = simple.respond(q)
    t = trivial.respond(q)
    print("=== ", q)
    print("  agent      :", a.intent, "conf %.2f" % a.confidence, "esc", a.should_escalate, "|", a.reply[:120])
    print("  simple     :", s.intent, "esc", s.should_escalate, "|", s.reply[:120])
    print("  trivial    :", t.intent, "esc", t.should_escalate, "|", t.reply[:120])
    print()
