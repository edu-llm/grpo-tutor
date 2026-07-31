#!/bin/bash
#SBATCH --job-name=grpo_v1c
#SBATCH --output=logs/v1c_%j.out
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#
# v1-controlled: run v0 with EXACTLY ONE change, the specificity reward.
#
# v0 established that leaking halves while honest teaching stays flat at 0.32,
# and that specificity sits at ~0 - a hint written for a different problem helps
# the student as much as the correct one. Nothing in the reward could tell
# teaching from generic help, so no amount of training moved it.
#
#   --specificity difference
#       reward = solved(own problem) - solved(other problem | same hint)
#       A hint that works anywhere now earns nothing.
#
# Everything else is v0's configuration on purpose: same 549 dual-screened
# OpenBookQA problems, same kl_coef 0.05, no persona adapter. If specificity
# moves, the reward is why - there is nothing else it could be.
#
# The one deviation is eval size: 200 items rather than 40. At n=40 the smallest
# trustworthy difference is 0.217, larger than every effect here, which is why
# v0's per-eval numbers were noise.
#
# 250 steps: v0 plateaued by ~150 and the last 350 taught us nothing.
set -euo pipefail
source scripts/env.sh
export WANDB_PROJECT=grpo_tutor
python -u src/train_h100.py \
  --backend vllm --reward real \
  --teacher Qwen/Qwen2.5-3B-Instruct \
  --student Qwen/Qwen2.5-0.5B-Instruct \
  --problems data/zpd_problems.jsonl \
  --specificity difference \
  --steps 250 --turns 3 --group-size 8 --micro-batch 4 \
  --no-sleep --gpu-mem-util 0.40 \
  --sync-every 1 --save-every 10 \
  --hint-probe --eval-every 25 --eval-n 200 --eval-benchmark qasc \
  --seed 0 --wandb
