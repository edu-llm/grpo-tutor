"""Does a teaching-quality signal survive inside the policy tier at all?

`train_rm.py --tier policy` fits 896 parameters to ~450 turns with AdamW and runs
the train loss to 2e-4. That is memorisation, and its test spearman of +0.05 says
nothing about whether signal exists - only that gradient descent with no capacity
control found none it could keep.

Ridge answers the question properly: closed form, one penalty swept on a
validation split, evaluated on questions held out from both. Reported three ways
because they disagree about what matters:

  spearman        rank correlation over all held-out turns
  within-question the only comparison GRPO ever makes - turns are z-scored inside
                  one problem's group, so ranking across problems is discarded
  tier AUC        hand-written vs policy, for reference: the number that looked
                  like 0.93 and was measuring provenance

    python src/rm_ridge.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict

import torch

from train_rm import spearman


def ridge(Xtr, ytr, alpha):
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * torch.eye(d, dtype=Xtr.dtype)
    return torch.linalg.solve(A, Xtr.T @ ytr)


def within_question_acc(pred, y, qid):
    """Fraction of same-question pairs ordered correctly. Ties count as half."""
    by = defaultdict(list)
    for p, t, q in zip(pred, y, qid):
        by[q].append((p, t))
    wins = pairs = 0.0
    for group in by.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                (pi, ti), (pj, tj) = group[i], group[j]
                if ti == tj:
                    continue
                pairs += 1
                if (pi > pj) == (ti > tj):
                    wins += 1
                elif pi == pj:
                    wins += 0.5
    return wins / pairs if pairs else float("nan"), int(pairs)


def splits(qs, seed):
    # sorted() first: iterating a set follows string hashing, which is salted per
    # process, so shuffling one directly gives different folds every run. Two
    # identical invocations disagreed by 0.066 on within-question accuracy before
    # this line existed - larger than the effect being measured.
    qs = sorted(qs)
    random.Random(seed).shuffle(qs)
    a, b = int(len(qs) * 0.2), int(len(qs) * 0.4)
    return set(qs[:a]), set(qs[a:b]), set(qs[b:])


def evaluate(X, y, qid, seeds=20, alphas=(1e1, 1e2, 1e3, 1e4, 1e5, 1e6)):
    out = []
    for s in range(seeds):
        te_q, va_q, tr_q = splits(set(qid), s)
        idx = {name: [i for i, q in enumerate(qid) if q in qs]
               for name, qs in (("te", te_q), ("va", va_q), ("tr", tr_q))}
        Xtr = X[idx["tr"]]
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
        Xn = (X - mu) / sd
        ytr = y[idx["tr"]] - y[idx["tr"]].mean()
        best, best_w = -1e9, None
        for a in alphas:
            w = ridge(Xn[idx["tr"]], ytr, a)
            r = spearman((Xn[idx["va"]] @ w).tolist(), y[idx["va"]].tolist())
            if r > best:
                best, best_w = r, w
        pt = (Xn[idx["te"]] @ best_w).tolist()
        yt = y[idx["te"]].tolist()
        acc, npairs = within_question_acc(pt, yt, [qid[i] for i in idx["te"]])
        out.append((spearman(pt, yt), acc, npairs))
    rho = [a for a, _, _ in out]
    acc = [b for _, b, _ in out if b == b]
    return rho, acc, sum(c for _, _, c in out)


def save_head(X, y, keep, path, backbone, alphas=(1e1, 1e2, 1e3, 1e4, 1e5, 1e6)):
    """Refit on every turn in the tier, penalty picked by held-out question CV.

    Written in TeachingScorer's format: an nn.Linear(d, 2) whose column 1 is the
    goodness direction. Column 0 is the leak head's slot and stays zero - the
    reward reads only column 1, and the rule-based detector owns leak.
    """
    import torch.nn as nn

    Xs, ys = X[keep], y[keep]
    qs = [qid_all[i] for i in keep]
    scores = defaultdict(list)
    for s in range(8):
        te_q, va_q, tr_q = splits(set(qs), s)
        tr = [i for i, q in enumerate(qs) if q in tr_q]
        va = [i for i, q in enumerate(qs) if q in va_q]
        mu, sd = Xs[tr].mean(0, keepdim=True), Xs[tr].std(0, keepdim=True) + 1e-6
        Xn = (Xs - mu) / sd
        for a in alphas:
            w = ridge(Xn[tr], ys[tr] - ys[tr].mean(), a)
            scores[a].append(spearman((Xn[va] @ w).tolist(), ys[va].tolist()))
    alpha = max(scores, key=lambda a: sum(scores[a]) / len(scores[a]))
    mu, sd = Xs.mean(0, keepdim=True), Xs.std(0, keepdim=True) + 1e-6
    w = ridge((Xs - mu) / sd, ys - ys.mean(), alpha)

    head = nn.Linear(X.shape[1], 2)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.weight[1] = w.float()
    torch.save({"head": head.state_dict(), "mu": mu.float(), "sd": sd.float(),
                "backbone": backbone, "hidden": 0, "alpha": alpha,
                "tier": "policy", "n": len(keep)}, path)
    print(f"\nwrote {path}  (ridge alpha {alpha:g}, fitted on {len(keep)} "
          f"policy-tier turns)")


def main():
    rows = json.load(open("data/rm_dataset.json"))["rows"]
    blob = torch.load("data/rm_embeddings.pt")
    X = blob["X"].double()
    assert blob["ids"] == [r["id"] for r in rows], "embedding cache is stale"
    global qid_all
    qid = qid_all = [r["text"].split("\n")[0] for r in rows]
    y = torch.tensor([r["goodness"] for r in rows], dtype=torch.float64)
    tier = [r["tier"] for r in rows]

    def report(name, keep):
        Xs, ys = X[keep], y[keep]
        qs = [qid[i] for i in keep]
        rho, acc, npairs = evaluate(Xs, ys, qs)
        n = len(keep)
        print(f"{name:26s} n={n:4d}  test spearman {sum(rho) / len(rho):+.3f} "
              f"[{min(rho):+.3f}, {max(rho):+.3f}]   "
              f"within-question {sum(acc) / len(acc):.3f} over {npairs} pairs")

    print("goodness, ridge with the penalty swept on held-out questions\n")
    report("both tiers pooled", list(range(len(rows))))
    report("policy tier only", [i for i, t in enumerate(tier) if t == "policy"])
    report("hand-written tier only", [i for i, t in enumerate(tier) if t == "good"])
    print("\n0.500 within-question = coin flip. This is the comparison GRPO makes.")

    if "--save" in sys.argv:
        save_head(X, y, [i for i, t in enumerate(tier) if t == "policy"],
                  "checkpoints/rm_head_policy_ridge.pt",
                  json.load(open("data/rm_dataset.json")).get(
                      "backbone", "Qwen/Qwen2.5-0.5B-Instruct"))


if __name__ == "__main__":
    main()
