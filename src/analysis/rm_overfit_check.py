"""Is the reward-model head overfitting, and does capacity in the head help?

    python src/rm_overfit_check.py

The head drives training loss to near zero, which looks alarming. What matters is
whether held-out performance holds up across splits, so this refits the head on
the cached 0.5B embeddings over several seeds and compares a linear probe against
the 256-unit MLP. Cheap: the backbone pass is already cached, so each fit is a
second.

Splits are by QUESTION. Several turns share a question, so a turn-level split
lets a head recognise the question rather than judge the tutoring.
"""

from __future__ import annotations

import json
import math
import random

import torch
import torch.nn as nn

from train_rm import spearman


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for a in pos for b in neg if a > b)
    ties = sum(1 for a in pos for b in neg if a == b)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def fit(X, y, tr, te, hidden, epochs=400, lr=1e-3):
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
        ptr, pte = head(Xn[tr]), head(Xn[te])
    return ptr, pte


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

    print(f"{'head':<8} {'seed':>4} {'train rho':>10} {'test rho':>9} "
          f"{'gap':>6} {'test AUC':>9}")
    summary = {}
    for hidden in (0, 64, 256):
        aucs, rhos = [], []
        for seed in range(5):
            rng = random.Random(seed)
            order = qs[:]
            rng.shuffle(order)
            n_te = int(len(order) * 0.2)
            te = [i for q in order[:n_te] for i in by_q[q]]
            tr = [i for q in order[n_te:] for i in by_q[q]]
            ptr, pte = fit(X, y, tr, te, hidden)
            rtr = spearman(ptr[:, 1].tolist(), y[tr][:, 1].tolist())
            rte = spearman(pte[:, 1].tolist(), y[te][:, 1].tolist())
            tiers = [rows[i]["tier"] for i in te]
            g = [p for p, t in zip(pte[:, 1].tolist(), tiers) if t == "good"]
            p_ = [p for p, t in zip(pte[:, 1].tolist(), tiers) if t == "policy"]
            a = auc(g, p_)
            aucs.append(a)
            rhos.append(rte)
            name = "linear" if hidden == 0 else f"mlp{hidden}"
            print(f"{name:<8} {seed:>4} {rtr:>10.3f} {rte:>9.3f} "
                  f"{rtr - rte:>6.3f} {a:>9.3f}")
        summary[hidden] = (sum(rhos) / len(rhos), sum(aucs) / len(aucs),
                           max(aucs) - min(aucs))
        print()

    print(f"{'head':<8} {'mean test rho':>14} {'mean AUC':>9} {'AUC spread':>11}")
    for h, (r, a, spread) in summary.items():
        name = "linear" if h == 0 else f"mlp{h}"
        print(f"{name:<8} {r:>14.3f} {a:>9.3f} {spread:>11.3f}")

    # how well would the LEAK head do compared with the rule it would replace?
    print("\nleak head, 0.5B embeddings, same protocol")
    rng = random.Random(0)
    order = qs[:]
    rng.shuffle(order)
    n_te = int(len(order) * 0.2)
    te = [i for q in order[:n_te] for i in by_q[q]]
    tr = [i for q in order[n_te:] for i in by_q[q]]
    ptr, pte = fit(X, y, tr, te, 256)
    rho = spearman(pte[:, 0].tolist(), y[te][:, 0].tolist())
    mae = (pte[:, 0] - y[te][:, 0]).abs().mean().item()
    base = (y[te][:, 0] - y[te][:, 0].mean()).abs().mean().item()
    print(f"  spearman {rho:+.3f}   MAE {mae:.3f} vs {base:.3f} for predicting the mean")
    # treat a predicted leak above a threshold as a flag, sweep it
    truth = [1 if v >= 2.5 else 0 for v in y[te][:, 0].tolist()]
    print("  as a binary flag against 'gives it away':")
    print(f"    {'thresh':>7} {'precision':>10} {'recall':>7}")
    for th in (1.8, 2.0, 2.2, 2.4):
        pred = [1 if v >= th else 0 for v in pte[:, 0].tolist()]
        tp = sum(1 for p, t in zip(pred, truth) if p and t)
        fp = sum(1 for p, t in zip(pred, truth) if p and not t)
        fn = sum(1 for p, t in zip(pred, truth) if not p and t)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"    {th:>7.1f} {prec:>10.3f} {rec:>7.3f}")
    print("  the rule it would replace: precision 0.397  recall 0.665")


if __name__ == "__main__":
    main()
