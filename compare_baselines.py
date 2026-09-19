import csv, sys
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from src.data import load_cached_or_fetch
from src.pairs import to_pairs
from src.baselines import SimpleBaseline, TrivialBaseline
from src.eval import split_pairs, load_golden

golden = load_golden()
train_pairs, _ = split_pairs(golden)
simple = SimpleBaseline(train_pairs)
trivial = TrivialBaseline(train_pairs)

yt=[r["intent"] for r in golden]; ec=[r["should_escalate"] for r in golden]
sp=[]; tp=[]
for r in golden:
    s=simple.respond(r["query"]); t=trivial.respond(r["query"])
    sp.append(s); tp.append(t)

for name, preds in [("simple",[s.intent for s in sp]),("trivial",[t.intent for t in tp])]:
    print(f"{name} intent acc {round(accuracy_score(yt,preds),3)} macroF1 {round(f1_score(yt,preds,average='macro',zero_division=0),3)}")
print("simple esc acc", round(accuracy_score(ec,[s.should_escalate for s in sp]),3), "F1", round(f1_score(ec,[s.should_escalate for s in sp],zero_division=0),3))
print("trivial esc F1", round(f1_score(ec,[t.should_escalate for t in tp],zero_division=0),3))
print("simple intent dist", Counter([s.intent for s in sp]))
