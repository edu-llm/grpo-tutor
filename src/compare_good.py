"""Compare the hand-written good tier against the policy on the same items."""

from __future__ import annotations

import collections
import json
import math

import rewards

good = ([json.loads(l) for l in open("runs/good_traces.jsonl")]
        + [json.loads(l) for l in open("runs/good_traces_b2.jsonl")])
with open("runs/good_traces_all.jsonl", "w") as f:
    for g in good:
        f.write(json.dumps(g) + "\n")

qs = {g["prompt"] for g in good}
v2 = [json.loads(l) for l in open("runs/20260731-212530/traces.jsonl")]
matched = [t for t in v2 if t["prompt"] in qs]


def rate(rows, k):
    return sum(row[k] for row in rows) / len(rows)


print(f"MATCHED on the same {len(qs)} items")
print("               n      solved   leaked   solved|not-leaked")
for name, rows in (("v2 policy", matched), ("hand-written", good)):
    nl = [x for x in rows if not x["leaked"]]
    print("%-14s %5d    %.3f    %.3f     %.3f"
          % (name, len(rows), rate(rows, "solved"), rate(rows, "leaked"),
             rate(nl, "solved")))

p1, p2 = rate(matched, "solved"), rate(good, "solved")
se = math.sqrt(p1 * (1 - p1) / len(matched) + p2 * (1 - p2) / len(good))
verdict = "significant" if abs(p2 - p1) > 2 * se else "NOT distinguishable"
print(f"solved difference {p2 - p1:+.3f} SE {se:.3f} -> {verdict}")

print("\nhow long is the tutor text, and how much of it is questions")
for name, rows in (("v2 policy", matched), ("hand-written", good)):
    words, qmarks = [], []
    for g in rows:
        t = [l[len("Tutor: "):] for l in g["completion"].split("\n")
             if l.startswith("Tutor: ")]
        joined = " ".join(t)
        words.append(len(joined.split()))
        qmarks.append(joined.count("?"))
    print("  %-13s %5.0f words   %.2f question marks per dialogue"
          % (name, sum(words) / len(words), sum(qmarks) / len(qmarks)))

print("\nwhich leak signal fires on the hand-written turns")
sig = collections.Counter()
for g in good:
    if not g["leaked"]:
        continue
    tutor = "\n".join(l[len("Tutor: "):] for l in g["completion"].split("\n")
                      if l.startswith("Tutor: "))
    distractors = [c for j, c in enumerate(g["choices"]) if j != g["gold_idx"]]
    s = rewards.leak_signals(tutor, g["gold"], distractors, question=g["prompt"])
    if s["verbatim"] >= 1:
        sig["verbatim - stated the gold text"] += 1
    elif s.get("identifying", 0) > 0:
        sig["identifying - used a word unique to gold"] += 1
    elif s["elimination"] >= 0.5:
        sig["elimination - ruled options out"] += 1
    else:
        sig["overlap - shared wording with gold"] += 1
for k, v in sig.most_common():
    print(f"  {v:3d}  {k}")
print(f"  {sum(g['leaked'] for g in good):.0f} flagged of {len(good)}")

print("\nby subject (hand-written)")
for s in ("math", "science", "social_studies"):
    sub = [g for g in good if g.get("subject") == s]
    if not sub:
        continue
    nl = [x for x in sub if not x["leaked"]]
    print("  %-15s n=%3d  solved %.3f  leaked %.3f  clean-solve %.3f"
          % (s, len(sub), rate(sub, "solved"), rate(sub, "leaked"),
             rate(nl, "solved") if nl else 0.0))
