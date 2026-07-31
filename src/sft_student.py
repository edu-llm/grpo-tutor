"""LoRA SFT so the student TALKS like a stuck kid - without changing how it SCORES.

Critical constraint: the reward comes from HFStudent.choose(), a log-prob
comparison over the answer options. Every number in this project - the ZPD
curation, the QASC baselines, the leak rates - assumes that channel is fixed. So
the adapter is trained on conversational turns only, and choose() runs with the
adapter DISABLED. Same trick GRPO already uses to get a reference model.

Loss is masked to the assistant's tokens: the student should learn to produce
kid-like replies, not to reproduce the tutor prompts it is conditioned on.
"""

from __future__ import annotations

import argparse
import json
import math

import torch

import paths


def build_examples(rows, tok, max_len):
    """Tokenize (system, user) -> assistant with the prompt masked out of the loss."""
    out = []
    for r in rows:
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": r["system"]},
             {"role": "user", "content": r["user"]}],
            tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        a_ids = tok(r["assistant"] + tok.eos_token, add_special_tokens=False).input_ids
        ids = (p_ids + a_ids)[:max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:max_len]
        if sum(1 for x in labels if x != -100) == 0:
            continue                      # prompt filled the budget; nothing to learn
        out.append((ids, labels))
    return out


def collate(batch, pad_id):
    n = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for r, (i, l) in enumerate(batch):
        ids[r, : len(i)] = torch.tensor(i)
        lab[r, : len(l)] = torch.tensor(l)
        att[r, : len(i)] = 1
    return ids, lab, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default=str(paths.DATA / "student_sft.jsonl"))
    ap.add_argument("--out", default=str(paths.CHECKPOINTS / "student-persona"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)   # SFT LoRA tolerates far
    ap.add_argument("--lora-r", type=int, default=16)   # more than RL's 5e-6
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if not rows:
        raise SystemExit(f"no rows in {args.data}")
    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    data = build_examples(rows, tok, args.max_len)
    n_val = max(1, int(len(data) * args.val_frac))
    val, train = data[:n_val], data[n_val:]
    print(f"[sft] {len(train)} train / {len(val)} val examples", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16).to(device)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def evaluate():
        model.eval()
        tot = ntok = 0.0
        with torch.no_grad():
            for i in range(0, len(val), args.batch_size):
                ids, lab, att = collate(val[i:i + args.batch_size], tok.pad_token_id)
                o = model(input_ids=ids.to(device), attention_mask=att.to(device),
                          labels=lab.to(device))
                m = (lab != -100).sum().item()
                tot += o.loss.item() * m
                ntok += m
        model.train()
        return tot / max(1, ntok)

    print(f"[sft] val loss before: {evaluate():.4f}", flush=True)
    for ep in range(args.epochs):
        run = 0.0
        for i in range(0, len(train), args.batch_size):
            ids, lab, att = collate(train[i:i + args.batch_size], tok.pad_token_id)
            loss = model(input_ids=ids.to(device), attention_mask=att.to(device),
                         labels=lab.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            run += loss.item()
        v = evaluate()
        print(f"[sft] epoch {ep + 1}/{args.epochs}  train {run / max(1, math.ceil(len(train) / args.batch_size)):.4f}"
              f"  val {v:.4f}", flush=True)

    model.save_pretrained(args.out)
    print(f"[sft] wrote adapter -> {args.out}")
    print("[sft] NOTE: load it for reply() only; choose() must run with the "
          "adapter disabled or every ZPD/QASC number shifts.")


if __name__ == "__main__":
    main()
