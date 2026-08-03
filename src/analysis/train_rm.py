"""Train a reward model: frozen backbone, two heads on the final hidden state.

    python src/train_rm.py

Why the FINAL hidden state, and why frozen. During GRPO the teacher's LoRA moves
its intermediate representations, so a head reading a middle layer would be
scoring a moving target. Freezing the backbone entirely and reading only the last
position of the last layer keeps the reward function fixed while the policy
changes, which is the property a reward model has to have.

Last POSITION, specifically: attention is causal, so only the final token has
attended to the whole tutor message. Mean-pooling over a causal model biases
towards early tokens, each of which has seen less of the text than the one after
it.

Two heads on one trunk, because both labels come from the same read:
  leak      1-3, higher means more of the answer given away
  goodness  1-5, higher means better teaching

Trained as regression on the MEAN rating rather than classification, because the
targets are averages over raters and land between integers - 3.5 is a real value
meaning "two raters split", and rounding it away discards that.

Embeddings are extracted once and cached, so retraining the heads takes seconds
and the expensive part happens a single time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random


def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rm_dataset.json")
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--cache", default="data/rm_embeddings.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256, help="0 for a linear probe")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15,
                    help="held out until the very end; val is what early "
                         "stopping looks at, so val is no longer clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tier", default=None,
                    help="fit and evaluate inside ONE tier. The two tiers differ "
                         "by style as much as by quality (hand-written turns ask "
                         "questions 62%% of the time against the policy's 95%%), "
                         "and tier is readable off the embedding at spearman "
                         "0.833 - so a head fitted on both scores provenance. "
                         "Every turn GRPO scores is policy-generated, so --tier "
                         "policy is the regime that matches deployment")
    ap.add_argument("--out", default="checkpoints/rm_head.pt")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    rows = json.load(open(args.data))["rows"]
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")

    # ---- embed once, cache, reuse ----
    if os.path.exists(args.cache):
        blob = torch.load(args.cache)
        X, ids = blob["X"], blob["ids"]
        assert ids == [r["id"] for r in rows], "cache is stale; delete it"
        print(f"loaded cached embeddings {tuple(X.shape)}")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"embedding {len(rows)} items with {args.backbone} on {device}")
        tok = AutoTokenizer.from_pretrained(args.backbone)
        model = AutoModelForCausalLM.from_pretrained(
            args.backbone, output_hidden_states=True).to(device).eval()
        vecs = []
        with torch.no_grad():
            for i in range(0, len(rows), 8):
                batch = [r["text"] for r in rows[i:i + 8]]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                          max_length=1024, padding_side="left").to(device)
                hs = model(**enc).hidden_states[-1]      # last layer
                vecs.append(hs[:, -1, :].float().cpu())  # last position
                if i % 200 == 0:
                    print(f"  {i}/{len(rows)}", flush=True)
        X = torch.cat(vecs)
        ids = [r["id"] for r in rows]
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        torch.save({"X": X, "ids": ids}, args.cache)
        print(f"cached embeddings {tuple(X.shape)} -> {args.cache}")

    if args.tier:
        keep = [i for i, r in enumerate(rows) if r["tier"] == args.tier]
        if not keep:
            raise SystemExit(f"no rows with tier={args.tier!r}")
        X = X[keep]
        rows = [rows[i] for i in keep]
        print(f"tier={args.tier}: {len(rows)} turns")

    y = torch.tensor([[r["leak"], r["goodness"]] for r in rows], dtype=torch.float32)

    # split by ITEM so no problem appears in both halves: several turns share a
    # question, and a head could otherwise memorise the question rather than the
    # tutoring
    rng = random.Random(args.seed)
    by_q = {}
    for i, r in enumerate(rows):
        by_q.setdefault(r["text"].split("\n")[0], []).append(i)
    qs = list(by_q)
    rng.shuffle(qs)
    n_val_q = int(len(qs) * args.val_frac)
    n_test_q = int(len(qs) * args.test_frac)
    test_qs, val_qs, tr_qs = (qs[:n_test_q], qs[n_test_q:n_test_q + n_val_q],
                              qs[n_test_q + n_val_q:])
    test_idx = [i for q in test_qs for i in by_q[q]]
    val_idx = [i for q in val_qs for i in by_q[q]]
    tr_idx = [i for q in tr_qs for i in by_q[q]]
    print(f"train {len(tr_idx)} / val {len(val_idx)} / test {len(test_idx)} turns "
          f"({len(tr_qs)}/{len(val_qs)}/{len(test_qs)} distinct questions)")

    mu, sd = X[tr_idx].mean(0, keepdim=True), X[tr_idx].std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    d = Xn.shape[1]
    head = (nn.Linear(d, 2) if args.hidden == 0 else
            nn.Sequential(nn.Linear(d, args.hidden), nn.GELU(),
                          nn.Dropout(0.1), nn.Linear(args.hidden, 2)))
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-2)
    lossf = nn.SmoothL1Loss()

    Xtr, ytr, Xva, yva = Xn[tr_idx], y[tr_idx], Xn[val_idx], y[val_idx]
    best, best_state = -1e9, None
    for ep in range(args.epochs):
        head.train()
        opt.zero_grad()
        loss = lossf(head(Xtr), ytr)
        loss.backward()
        opt.step()
        if (ep + 1) % 25 == 0:
            head.eval()
            with torch.no_grad():
                pv = head(Xva)
            sg = spearman(pv[:, 1].tolist(), yva[:, 1].tolist())
            sl = spearman(pv[:, 0].tolist(), yva[:, 0].tolist())
            score = sg + sl
            if score > best:
                best, best_state = score, {k: v.clone() for k, v in head.state_dict().items()}
            print(f"  epoch {ep + 1:4d}  train {loss.item():.4f}  "
                  f"val spearman leak {sl:+.3f} goodness {sg:+.3f}")

    head.load_state_dict(best_state)
    head.eval()
    Xte, yte = Xn[test_idx], y[test_idx]
    with torch.no_grad():
        pv, pt = head(Xva), head(Xte)

    print("\nvalidation (used for model selection)")
    for j, (name, lo, hi) in enumerate((("leak", 1, 3), ("goodness", 1, 5))):
        mae = (pv[:, j] - yva[:, j]).abs().mean().item()
        rho = spearman(pv[:, j].tolist(), yva[:, j].tolist())
        base = (yva[:, j] - yva[:, j].mean()).abs().mean().item()
        print(f"  {name:9s} MAE {mae:.3f}  (predicting the mean gives {base:.3f})"
              f"   spearman {rho:+.3f}")

    print("\ntest (never seen during training or selection)")
    for j, (name, lo, hi) in enumerate((("leak", 1, 3), ("goodness", 1, 5))):
        mae = (pt[:, j] - yte[:, j]).abs().mean().item()
        rho = spearman(pt[:, j].tolist(), yte[:, j].tolist())
        base = (yte[:, j] - yte[:, j].mean()).abs().mean().item()
        print(f"  {name:9s} MAE {mae:.3f}  (predicting the mean gives {base:.3f})"
              f"   spearman {rho:+.3f}")

    # the test that matters: can it tell the tiers apart on unseen questions?
    tiers = [rows[i]["tier"] for i in test_idx]
    pg = pt[:, 1].tolist()
    good = [p for p, t in zip(pg, tiers) if t == "good"]
    pol = [p for p, t in zip(pg, tiers) if t == "policy"]
    if good and pol:
        pairs = sum(1 for a in good for b in pol)
        wins = sum(1 for a in good for b in pol if a > b) + 0.5 * sum(
            1 for a in good for b in pol if a == b)
        print(f"\n  predicted goodness: hand-written {sum(good) / len(good):.2f} "
              f"vs policy {sum(pol) / len(pol):.2f}")
        print(f"  AUC separating the tiers: {wins / pairs:.3f}  "
              f"(0.5 = blind, 1.0 = perfect)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"head": head.state_dict(), "mu": mu, "sd": sd,
                "backbone": args.backbone, "hidden": args.hidden}, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
