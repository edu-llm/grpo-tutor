"""Rank accuracy WITHIN a question - the only comparison GRPO actually makes.

    python src/rm_within_group.py

Global AUC pools comparisons across different problems. GRPO never makes those:
the advantage is computed inside a group of completions for ONE problem, so the
reward only has to order turns that share a question. A head can look strong
globally by learning "maths questions score higher than history questions" and
still be useless inside a group, because that between-question variance cancels
in the centring.

So this scores pairwise accuracy on same-question pairs, which is also why
scaling barely matters: the advantage is z-scored per group, so only the ordering
and relative spacing inside a group survive.
"""

from __future__ import annotations

import itertools
import json
import random

import torch
import torch.nn as nn


def fit(X, y, tr, hidden, epochs=400, lr=1e-3):
    mu, sd = X[tr].mean(0, keepdim=True), X[tr].std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    d = Xn.shape[1]
    head = (nn.Linear(d, 2) if hidden == 0 else
            nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.1),
                          nn.Linear(hidden, 2)))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    lossf = nn.SmoothL1Loss()
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        lossf(head(Xn[tr]), y[tr]).backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        return head(Xn), mu, sd


def main():
    rows = json.load(open("data/rm_dataset.json"))["rows"]
    blob = torch.load("data/rm_embeddings.pt")
    X = blob["X"]
    assert blob["ids"] == [r["id"] for r in rows]
    y = torch.tensor([[r["leak"], r["goodness"]] for r in rows], dtype=torch.float32)

    by_q = {}
    for i, r in enumerate(rows):
        by_q.setdefault(r["text"].split("\n")[0], []).append(i)
    qs = list(by_q)
    sizes = sorted(len(v) for v in by_q.values())
    print(f"{len(qs)} questions, {len(rows)} turns, "
          f"median {sizes[len(sizes) // 2]} turns per question")

    results = {}
    for hidden in (0, 256):
        within, glob = [], []
        for seed in range(5):
            rng = random.Random(seed)
            order = qs[:]
            rng.shuffle(order)
            n_te = int(len(order) * 0.2)
            te_qs, tr_qs = order[:n_te], order[n_te:]
            tr = [i for q in tr_qs for i in by_q[q]]
            pred, _, _ = fit(X, y, tr, hidden)
            pg = pred[:, 1].tolist()
            truth = y[:, 1].tolist()

            # within-question pairs, held-out questions only
            right = total = 0
            for q in te_qs:
                idx = by_q[q]
                for a, b in itertools.combinations(idx, 2):
                    if truth[a] == truth[b]:
                        continue          # no ordering to get right
                    total += 1
                    hi, lo = (a, b) if truth[a] > truth[b] else (b, a)
                    right += (pg[hi] > pg[lo]) + 0.5 * (pg[hi] == pg[lo])
            within.append(right / total if total else float("nan"))

            # global pairs across held-out questions, for contrast
            te = [i for q in te_qs for i in by_q[q]]
            r2 = t2 = 0
            for a, b in itertools.combinations(te, 2):
                if truth[a] == truth[b]:
                    continue
                t2 += 1
                hi, lo = (a, b) if truth[a] > truth[b] else (b, a)
                r2 += (pg[hi] > pg[lo]) + 0.5 * (pg[hi] == pg[lo])
            glob.append(r2 / t2 if t2 else float("nan"))

        name = "linear" if hidden == 0 else f"mlp{hidden}"
        results[name] = (within, glob)
        print(f"\n{name}")
        print(f"  within-question  {sum(within) / len(within):.3f}  "
              f"(per seed: {' '.join(f'{v:.3f}' for v in within)})")
        print(f"  across questions {sum(glob) / len(glob):.3f}  "
              f"(per seed: {' '.join(f'{v:.3f}' for v in glob)})")
        print(f"  pairs compared: {total} within, {t2} across")

    print("\n0.5 = coin flip. Within-question is the number that matters for GRPO;")
    print("across-questions flatters any head that has learned subject difficulty.")


if __name__ == "__main__":
    main()
