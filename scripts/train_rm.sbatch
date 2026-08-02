#!/bin/bash
#SBATCH --job-name=train_rm
#SBATCH --output=logs/rm_%j.out
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#
# Reward model: frozen backbone, two heads on the FINAL hidden state.
#
# Frozen and final-layer on purpose. GRPO moves the teacher's LoRA, so a head
# reading intermediate activations would score a moving target; the last layer of
# a frozen trunk is fixed for the whole run. Last POSITION too - attention is
# causal, so only the final token has read the entire tutor message.
#
# Three backbones, same heads and same split, to see what capacity buys. The
# 0.5B run on a laptop already reached AUC 0.918 separating hand-written tutoring
# from the policy on unseen questions; this is whether 3B or 7B does better.
#
# Split is by QUESTION, not by turn: several turns share a question, and a
# turn-level split lets the head memorise the question instead of judging the
# tutoring. Test is held out from model selection entirely.
set -euo pipefail
source scripts/env.sh
export HF_HUB_OFFLINE=1

for M in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  TAG=$(basename "$M")
  echo "=============== $M ==============="
  python -u src/train_rm.py \
    --data data/rm_dataset.json \
    --backbone "$M" \
    --cache "data/rm_emb_${TAG}.pt" \
    --hidden 256 --epochs 400 --lr 1e-3 \
    --val-frac 0.15 --test-frac 0.15 --seed 0 \
    --out "checkpoints/rm_head_${TAG}.pt"
done
