"""Agent vs human agreement on the same labelled turns.

    python src/agreement.py

Inter-annotator agreement caps what any reward model trained on these labels can
achieve, and it is the number to know BEFORE paying for labels at scale. If a
careful agent and a careful human disagree badly, the question is not well posed
and no volume of labelling fixes it.

Reports exact agreement, within-one agreement, and a linearly weighted kappa,
which is the right statistic for an ordinal scale: it gives partial credit for
being one point off and corrects for the agreement you would get by chance.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")
from peek_labels import access_token, plain  # noqa: E402

import urllib.request  # noqa: E402


def weighted_kappa(a, b, lo, hi):
    """Linearly weighted Cohen's kappa over an ordinal scale [lo, hi]."""
    cats = list(range(lo, hi + 1))
    n = len(a)
    idx = {c: i for i, c in enumerate(cats)}
    obs = [[0] * len(cats) for _ in cats]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    ra = [sum(r) for r in obs]
    ca = [sum(obs[i][j] for i in range(len(cats))) for j in range(len(cats))]
    maxd = hi - lo
    num = den = 0.0
    for i, ci in enumerate(cats):
        for j, cj in enumerate(cats):
            w = abs(ci - cj) / maxd
            num += w * obs[i][j]
            den += w * ra[i] * ca[j] / n
    return 1 - num / den if den else float("nan")


def main():
    tok = access_token()
    url = ("https://firestore.googleapis.com/v1/projects/grpo-tutor-label"
           "/databases/(default)/documents/labels?pageSize=300")
    d = json.load(urllib.request.urlopen(
        urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})))
    human = {}
    for doc in d.get("documents", []):
        f = {k: plain(v) for k, v in doc.get("fields", {}).items()}
        if f.get("kind") == "turn" and isinstance(f.get("leak"), int):
            human[f["itemId"]] = f

    agent = json.load(open("data/agent_labels.json"))["labels"]
    key = {t["id"]: t for t in json.load(open("label_app/data/label_key.json"))["turns"]}
    shared = [i for i in agent if i in human]
    print(f"{len(human)} human labels, {len(agent)} agent labels, {len(shared)} shared\n")
    if not shared:
        return

    for field, lo, hi in (("leak", 1, 3), ("goodness", 1, 5)):
        h = [human[i][field] for i in shared]
        a = [agent[i][field] for i in shared]
        exact = sum(x == y for x, y in zip(h, a)) / len(h)
        within1 = sum(abs(x - y) <= 1 for x, y in zip(h, a)) / len(h)
        bias = sum(a) / len(a) - sum(h) / len(h)
        print(f"{field}  (1-{hi})")
        print(f"  exact      {exact:.1%}")
        print(f"  within one {within1:.1%}")
        print(f"  kappa_w    {weighted_kappa(h, a, lo, hi):+.2f}")
        print(f"  agent mean {sum(a) / len(a):.2f} vs human {sum(h) / len(h):.2f}"
              f"  (agent runs {bias:+.2f})")
        worst = sorted(shared, key=lambda i: -abs(human[i][field] - agent[i][field]))[:3]
        for i in worst:
            if abs(human[i][field] - agent[i][field]) >= 2:
                print(f"    {i}: human {human[i][field]} vs agent {agent[i][field]}"
                      f" - {agent[i]['why'][:70]}")
        print()

    # does either rater separate the hand-written tier from the policy tier?
    print("mean goodness by tier (the scale should see this difference)")
    for who, src in (("human", {i: human[i]["goodness"] for i in shared}),
                     ("agent", {i: agent[i]["goodness"] for i in shared})):
        by = {"good": [], "policy": []}
        for i, v in src.items():
            tier = key.get(i, {}).get("tier")
            if tier in by:
                by[tier].append(v)
        parts = [f"{t} {sum(v) / len(v):.2f} (n={len(v)})" for t, v in by.items() if v]
        gap = ""
        if by["good"] and by["policy"]:
            gap = f"  gap {sum(by['good']) / len(by['good']) - sum(by['policy']) / len(by['policy']):+.2f}"
        print(f"  {who}: " + "  ".join(parts) + gap)


if __name__ == "__main__":
    main()
