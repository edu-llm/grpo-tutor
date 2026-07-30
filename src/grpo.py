"""GRPO core: group-relative advantage + clipped PG loss with a KL leash.

Speed-oriented structure. The naive version re-tokenizes prompts and recomputes
the frozen reference forward pass on EVERY update epoch; both are pure waste.
Here:

  prepare_batch()   - tokenize + pad ONCE per step (length-sorted to cut padding)
  reference_logprobs() - frozen reference forward ONCE per step (it can't change
                      during the epochs, so caching it halves the forward passes)
  grpo_loss()       - per-epoch, micro-batched so long sequences don't OOM

Token log-probs use F.cross_entropy (fused log_softmax+gather) instead of
materializing a (B, L, V) log_softmax tensor - meaningfully less memory/time.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def group_normalized_advantages(rewards, group_size):
    """Within each consecutive group of `group_size`, subtract mean / divide std."""
    adv = []
    for i in range(0, len(rewards), group_size):
        grp = rewards[i : i + group_size]
        mean = statistics.fmean(grp)
        std = statistics.pstdev(grp) if len(grp) > 1 else 0.0
        adv.extend([(r - mean) / (std + 1e-8) for r in grp])
    return adv


@dataclass
class Batch:
    input_ids: torch.Tensor   # (B, L)
    attn: torch.Tensor        # (B, L)
    mask: torch.Tensor        # (B, L) 1 on policy-generated tokens
    old_lp: torch.Tensor      # (B, L) behavior logprobs, aligned to mask
    adv: torch.Tensor         # (B, 1)

    def __len__(self):
        return self.input_ids.shape[0]

    def slice(self, a, b):
        return Batch(self.input_ids[a:b], self.attn[a:b], self.mask[a:b],
                     self.old_lp[a:b], self.adv[a:b])


def prepare_batch(tokenizer, samples, device, max_len: int = 1024) -> Batch:
    """Tokenize + pad once per step. Length-sorted so padding waste is minimal."""
    pad_id = tokenizer.pad_token_id
    rows = []
    for s in samples:
        p_ids = tokenizer(s["prompt"], add_special_tokens=False).input_ids
        g_ids = list(s["gen_ids"])
        old = list(s["old_logprobs"])
        budget = max_len - len(g_ids)
        if budget < 1:
            g_ids, old, budget = g_ids[: max_len - 1], old[: max_len - 1], 1
        p_ids = p_ids[-budget:]          # keep the tail of the prompt if truncating
        rows.append((p_ids + g_ids,
                     [0] * len(p_ids) + [1] * len(g_ids),
                     [0.0] * len(p_ids) + old,
                     float(s["advantage"])))

    rows.sort(key=lambda r: len(r[0]))   # bucket by length -> less padding
    B, L = len(rows), max(len(r[0]) for r in rows)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.float)
    old_lp = torch.zeros((B, L), dtype=torch.float)
    adv = torch.zeros((B, 1), dtype=torch.float)
    for i, (seq, m, o, a) in enumerate(rows):
        n = len(seq)
        input_ids[i, :n] = torch.tensor(seq, dtype=torch.long)
        attn[i, :n] = 1
        mask[i, :n] = torch.tensor(m, dtype=torch.float)
        old_lp[i, :n] = torch.tensor(o, dtype=torch.float)
        adv[i, 0] = a
    return Batch(input_ids.to(device), attn.to(device), mask.to(device),
                 old_lp.to(device), adv.to(device))


def _token_logprobs(model, input_ids, attn, want_entropy: bool = False):
    """(B, L-1) logprobs of targets input_ids[:,1:] via fused cross_entropy.

    If `want_entropy`, also returns the TRUE per-token policy entropy
    H = -sum_v p(v) log p(v) over the full vocab (not just the sampled token).
    Entropy collapse -> 0 is a classic RL failure mode (the policy stops exploring
    and locks onto one phrasing), so it is worth watching directly.
    """
    logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits[:, :-1, :]
    B, T, V = logits.shape
    targets = input_ids[:, 1:]
    nll = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1), reduction="none")
    lp = -nll.view(B, T)
    if not want_entropy:
        return lp, None
    with torch.no_grad():   # diagnostic only - never backprop through it
        full_lp = torch.log_softmax(logits.float(), dim=-1)
        ent = -(full_lp.exp() * full_lp).sum(-1)
    return lp, ent


@torch.no_grad()
def reference_logprobs(model, batch: Batch, micro_batch: int = 8):
    """Frozen-reference logprobs, computed ONCE per step (LoRA disabled = base model)."""
    if not hasattr(model, "disable_adapter"):
        return None
    outs = []
    with model.disable_adapter():
        for a in range(0, len(batch), micro_batch):
            sub = batch.slice(a, a + micro_batch)
            outs.append(_token_logprobs(model, sub.input_ids, sub.attn)[0])
    return torch.cat(outs, dim=0)


def grpo_loss(model, batch: Batch, ref_lp, cfg, micro_batch: int = 8, accumulate: bool = True):
    """One update epoch. Micro-batched; if `accumulate`, calls backward per chunk
    so peak memory stays flat. Returns (total_loss_value, metrics)."""
    n = len(batch)
    tot = {"loss": 0.0, "kl": 0.0, "ratio": 0.0, "clip_frac": 0.0,
           "entropy_proxy": 0.0, "entropy": 0.0}
    total_tokens = 0.0

    for a in range(0, n, micro_batch):
        b = min(a + micro_batch, n)
        sub = batch.slice(a, b)
        m = sub.mask[:, 1:]
        denom_all = batch.mask[:, 1:].sum().clamp(min=1.0)   # normalize over the whole step
        new, ent = _token_logprobs(model, sub.input_ids, sub.attn, want_entropy=True)
        old = sub.old_lp[:, 1:]

        ratio = torch.exp((new - old).clamp(-20, 20))
        surrogate = torch.min(ratio * sub.adv,
                              ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * sub.adv)

        if ref_lp is not None:
            diff = (ref_lp[a:b] - new).clamp(-20, 20)
            kl = torch.exp(diff) - diff - 1.0     # k3: unbiased, >= 0
        else:
            kl = torch.zeros_like(new)

        chunk_loss = (((-surrogate + cfg.kl_coef * kl) * m).sum()) / denom_all
        if accumulate:
            chunk_loss.backward()

        with torch.no_grad():
            tk = m.sum()
            total_tokens += tk.item()
            tot["loss"] += chunk_loss.item()
            tot["kl"] += (kl * m).sum().item()
            tot["ratio"] += (ratio * m).sum().item()
            tot["clip_frac"] += ((((ratio < 1 - cfg.clip_eps) | (ratio > 1 + cfg.clip_eps)).float()) * m).sum().item()
            tot["entropy_proxy"] += (-new * m).sum().item()
            tot["entropy"] += (ent * m).sum().item()

    d = max(total_tokens, 1.0)
    metrics = {"loss": tot["loss"], "kl": tot["kl"] / d, "ratio": tot["ratio"] / d,
               "clip_frac": tot["clip_frac"] / d, "entropy_proxy": tot["entropy_proxy"] / d,
               "entropy": tot["entropy"] / d, "tokens": int(total_tokens)}
    return tot["loss"], metrics
