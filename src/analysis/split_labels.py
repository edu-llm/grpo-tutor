"""Split the rated turns by who rated them, and score the head against humans.

    python src/split_labels.py

The reward model is fitted on a pooled mean over human and agent ratings, and
1,027 of the 1,104 items are agent-only. That makes one question worth asking
directly: does a head trained on agent judgement predict HUMAN judgement?

Writes three files:
  data/rm_human.json   items with at least one human rating
  data/rm_agent.json   items rated only by agents
  data/rm_split.json   both, plus the per-item rater breakdown

The human set is small, so treat it as a sanity check rather than a benchmark -
it is the only ground truth anchoring the other 1,027 items, and its size is the
main reason to keep labelling by hand.
"""

from __future__ import annotations

import collections
import json
import sys

sys.path.insert(0, "src")


def main():
    rows = json.load(open("data/rm_dataset.json"))["rows"]
    human, agent = [], []
    for r in rows:
        if any(x.startswith("human:") for x in r["raters"]):
            human.append(r)
        else:
            agent.append(r)

    print(f"{len(rows)} rated turns")
    print(f"  {len(human):4d} have a human rating")
    print(f"  {len(agent):4d} are agent-only")

    both = [r for r in human if any(x.startswith("agent:") for x in r["raters"])]
    print(f"  {len(both):4d} rated by BOTH a human and an agent")

    counts = collections.Counter(x.split(":")[0] for r in rows for x in r["raters"])
    print(f"\nratings by source: {dict(counts)}")
    for t in ("good", "policy"):
        h = [r for r in human if r["tier"] == t]
        a = [r for r in agent if r["tier"] == t]
        if h and a:
            print(f"  {t:7s} human n={len(h):3d} mean goodness "
                  f"{sum(x['goodness'] for x in h) / len(h):.2f}   "
                  f"agent n={len(a):4d} mean {sum(x['goodness'] for x in a) / len(a):.2f}")

    json.dump({"schema": "grpo-tutor-rm/v1", "rows": human},
              open("data/rm_human.json", "w"))
    json.dump({"schema": "grpo-tutor-rm/v1", "rows": agent},
              open("data/rm_agent.json", "w"))
    json.dump({"human": [r["id"] for r in human], "agent": [r["id"] for r in agent],
               "both": [r["id"] for r in both]}, open("data/rm_split.json", "w"), indent=1)
    print("\nwrote data/rm_human.json, data/rm_agent.json, data/rm_split.json")

    # --- does a head fitted on agent labels predict the human ones? ---
    try:
        import torch
        import torch.nn as nn
        from train_rm import spearman
    except ImportError:
        return
    import os
    if not os.path.exists("data/rm_embeddings.pt"):
        print("\n(no cached embeddings; skipping the head check)")
        return

    blob = torch.load("data/rm_embeddings.pt")
    X, ids = blob["X"], blob["ids"]
    pos = {i: k for k, i in enumerate(ids)}
    tr = [pos[r["id"]] for r in agent]
    te = [pos[r["id"]] for r in human]
    y = torch.tensor([[r["leak"], r["goodness"]] for r in rows], dtype=torch.float32)

    mu, sd = X[tr].mean(0, keepdim=True), X[tr].std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    head = nn.Linear(Xn.shape[1], 2)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.SmoothL1Loss()
    for _ in range(600):
        head.train()
        opt.zero_grad()
        lossf(head(Xn[tr]), y[tr]).backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        p = head(Xn[te])

    print("\nhead trained on AGENT labels only, evaluated on the HUMAN-rated items")
    for j, name in ((0, "leak"), (1, "goodness")):
        rho = spearman(p[:, j].tolist(), y[te][:, j].tolist())
        print(f"  {name:9s} spearman {rho:+.3f}  (n={len(te)})")
    tiers = [rows[i]["tier"] for i in te]
    g = [v for v, t in zip(p[:, 1].tolist(), tiers) if t == "good"]
    q = [v for v, t in zip(p[:, 1].tolist(), tiers) if t == "policy"]
    if g and q:
        wins = sum(1 for a in g for b in q if a > b) + 0.5 * sum(
            1 for a in g for b in q if a == b)
        print(f"  AUC on the human-rated items: {wins / (len(g) * len(q)):.3f} "
              f"(good n={len(g)}, policy n={len(q)})")


if __name__ == "__main__":
    main()
