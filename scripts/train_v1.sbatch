#!/bin/bash
#SBATCH --job-name=grpo_v1
#SBATCH --output=logs/v1_%j.out
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#
# v1. Run v0 established that the leak penalty works and that NOTHING in the
# reward improves teaching: leaking halved while the solve rate on non-leaked
# dialogues sat flat at 0.32, and specificity stayed at ~0 - a hint written for
# a different problem helped as much as the right one.
#
# What changed, and why:
#   --specificity difference   the actual experiment. Pays solved(own problem)
#                              minus solved(other problem | same hint), so a hint
#                              that works anywhere earns nothing.
#   real state assessments     CA + TX released items. v0 traced specificity ~0
#                              to answer shape: single-word answers make
#                              "explain" and "reveal" the same act.
#   --eval-benchmark pa        Pennsylvania, a DIFFERENT state - held out means
#                              different authors and a different year, not a
#                              random slice. 235 items, so the minimum detectable
#                              difference is 0.089 rather than v0's 0.217.
#   --persona-adapter          the student talks like a child; applies to reply()
#                              only, choose() runs with it disabled.
#   --kl-coef 0.01             0.05 anchors the policy to a base model whose
#                              instinct is to answer the question.
#   250 steps                  v0 plateaued by ~150. 500 taught us nothing extra.
set -euo pipefail
source scripts/env.sh
export WANDB_PROJECT=grpo_tutor
python -u src/train_h100.py \
  --backend vllm --reward real \
  --teacher Qwen/Qwen2.5-3B-Instruct \
  --student Qwen/Qwen2.5-0.5B-Instruct \
  --problems data/state_tests/train_items.jsonl \
  --persona-adapter checkpoints/student-persona \
  --specificity difference \
  --kl-coef 0.01 \
  --steps 250 --turns 3 --group-size 8 --micro-batch 4 \
  --no-sleep --gpu-mem-util 0.40 \
  --sync-every 1 --save-every 10 \
  --hint-probe --eval-every 25 --eval-n 200 --eval-benchmark pa \
  --seed 0 --wandb
