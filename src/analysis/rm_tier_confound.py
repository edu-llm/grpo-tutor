"""Is the teaching head measuring teaching, or measuring provenance?

The label set is two tiers - 447 hand-written `good` turns and 657 `policy`
turns - and they differ on more than quality:

    good    goodness 4.26   question-rate 62%
    policy  goodness 2.55   question-rate 95%

Tier therefore predicts the label almost by itself, and any surface feature that
tracks tier is a shortcut the head can take instead of reading the tutoring. At
RL time every turn scored is policy-generated, so a head that works by spotting
hand-written text is being run entirely outside the regime it learned.

This asks the question the pooled AUC cannot: fit and evaluate WITHIN the policy
tier, where provenance is constant and only teaching quality varies. If the
correlation survives, the head reads tutoring. If it collapses, the reward is a
tier classifier and optimising it pushes the policy towards the surface style of
the hand-written set rather than towards better teaching.

    python src/rm_tier_confound.py
"""

from __future__ import annotations

import json
import random
import re

import torch
import torch.nn as nn

from train_rm import spearman


def rated_text(row: dict) -> str:
    m = re.search(r"Tutor message to rate: (.*)$", row["text"], re.S)
    return m.group(1) if m else ""


def fit_head(X, y, tr, va, epochs=400, lr=1e-2):
    """Linear probe on the goodness column, selected on validation spearman."""
    mu, sd = X[tr].mean(0, keepdim=True), X[tr].std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    head = nn.Linear(Xn.shape[1], 1)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    lossf = nn.SmoothL1Loss()
    best, best_state = -1e9, None
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        lossf(head(Xn[tr]).squeeze(-1), y[tr]).backward()
        opt.step()
        if (ep + 1) % 20 == 0:
            head.eval()
            with torch.no_grad():
                rho = spearman(head(Xn[va]).squeeze(-1).tolist(), y[va].tolist())
            if rho > best:
                best, best_state = rho, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, mu, sd, best


def split_by_question(rows, idx, seed, val_frac=0.3):
    """Group by question so a head cannot memorise the item instead of the turn."""
    by_q = {}
    for i in idx:
        by_q.setdefault(rows[i]["text"].split("\n")[0], []).append(i)
    qs = list(by_q)
    random.Random(seed).shuffle(qs)
    cut = int(len(qs) * val_frac)
    va = [i for q in qs[:cut] for i in by_q[q]]
    tr = [i for q in qs[cut:] for i in by_q[q]]
    return tr, va


def main():
    rows = json.load(open("data/rm_dataset.json"))["rows"]
    blob = torch.load("data/rm_embeddings.pt")
    X, ids = blob["X"], blob["ids"]
    assert ids == [r["id"] for r in rows], "embedding cache is stale"
    y = torch.tensor([r["goodness"] for r in rows], dtype=torch.float32)
    tier = [r["tier"] for r in rows]
    policy = [i for i, t in enumerate(tier) if t == "policy"]
    print(f"{len(rows)} turns: {len(policy)} policy, {len(rows) - len(policy)} good\n")

    # 1. how much of the pooled score is just tier?
    ytier = torch.tensor([1.0 if t == "good" else 0.0 for t in tier])
    tr, va = split_by_question(rows, list(range(len(rows))), seed=0)
    _, _, _, rho_tier = fit_head(X, ytier, tr, va)
    print(f"predicting TIER from the embedding      val spearman {rho_tier:+.3f}")
    print("  (how easily provenance is readable at all)\n")

    # 2. the pooled head, as shipped
    pooled = [fit_head(X, y, *split_by_question(rows, list(range(len(rows))), s))[3]
              for s in range(5)]
    print(f"goodness, POOLED  (both tiers)          val spearman "
          f"{sum(pooled) / len(pooled):+.3f}  "
          f"[{min(pooled):+.3f}, {max(pooled):+.3f}] over 5 seeds")

    # 3. the same head fitted and judged inside the policy tier only
    within = [fit_head(X, y, *split_by_question(rows, policy, s))[3] for s in range(5)]
    print(f"goodness, WITHIN the policy tier        val spearman "
          f"{sum(within) / len(within):+.3f}  "
          f"[{min(within):+.3f}, {max(within):+.3f}] over 5 seeds")
    print("  (this is the regime RL actually runs in: every scored turn is policy)\n")

    # 4. does the shipped head simply rank question-asking down?
    ship = torch.load("checkpoints/rm_head_linear.pt", map_location="cpu")
    head = nn.Linear(ship["mu"].shape[1], 2)
    head.load_state_dict(ship["head"])
    with torch.no_grad():
        pred = head(((X - ship["mu"]) / ship["sd"]))[:, 1]
    q = torch.tensor([1.0 if "?" in rated_text(r) else 0.0 for r in rows])
    ln = torch.tensor([float(len(rated_text(r))) for r in rows])
    print("the SHIPPED head, scored on the label set")
    for name, mask in (("with a question", q == 1), ("without a question", q == 0)):
        print(f"  {name:20s} n={int(mask.sum()):4d}  predicted goodness "
              f"{pred[mask].mean():+.3f}   labelled {y[mask].mean():.2f}")
    print(f"  spearman(prediction, has-question) {spearman(pred.tolist(), q.tolist()):+.3f}")
    print(f"  spearman(prediction, length)       {spearman(pred.tolist(), ln.tolist()):+.3f}")
    pol = torch.tensor([t == "policy" for t in tier])
    print(f"  spearman(prediction, goodness) within policy tier only "
          f"{spearman(pred[pol].tolist(), y[pol].tolist()):+.3f}")


if __name__ == "__main__":
    main()
