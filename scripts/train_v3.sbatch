#!/bin/bash
#SBATCH --job-name=grpo_v3
#SBATCH --output=logs/v3_%j.out
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#
# v3. v2's setup with ONE addition: a learned teaching score in the reward.
#
# Why. v2 established that the outcome channel can barely see teaching quality -
# hand-written expert tutoring beat the policy by +0.07 on the student's answer,
# while human-style raters put the same gap at +1.75 on a 5-point scale (t=8.3,
# 1,104 rated turns). The objective has been blind to the thing we care about.
#
#   --teach-head    frozen 0.5B backbone, LINEAR head on the final hidden state.
#                   Linear, not MLP: the MLP memorised (train rho 0.976 vs test
#                   0.55) and its AUC swung 0.87-0.94 across splits, where the
#                   linear probe holds 0.93 with a third of the variance and
#                   extrapolates predictably when RL pushes off-distribution.
#                   Test AUC 0.918 separating hand-written from policy turns on
#                   questions it never saw.
#   --teach-coef    0.5. `solved` has a within-group sd near 0.4, so this makes
#                   the two terms comparable rather than letting either dominate.
#
# The score is z-scored WITHIN the group before it is added: GRPO centres
# advantages inside a group, so only the spread across one problem's completions
# survives, and raw values would let an easy problem dominate the batch.
#
# A leaked turn keeps its flat -1 and never receives the teaching bonus. Paying
# for quality on top of a leak would reward a well-phrased give-away.
#
# `solved` stays in the reward on purpose. It is the one term the head cannot be
# gamed against, so if the teaching score climbs while eval clean_solved stays
# flat, the head is being hacked and we will see it within one eval cycle.
#
# Everything else is v2 exactly: same 307 screened CA+TX+MA+NJ items, all 235
# Pennsylvania items held out, specificity reward, persona adapter, kl 0.03,
# 250 steps, seed 0.
set -euo pipefail
source scripts/env.sh
export HF_HUB_OFFLINE=1
export WANDB_PROJECT=grpo_tutor
python -u src/train_h100.py \
  --backend vllm --reward real \
  --teacher Qwen/Qwen2.5-3B-Instruct \
  --student Qwen/Qwen2.5-0.5B-Instruct \
  --problems data/state_tests/train_items.jsonl \
  --persona-adapter checkpoints/student-persona \
  --specificity difference \
  --teach-head checkpoints/rm_head_linear.pt \
  --teach-coef 0.5 \
  --kl-coef 0.03 \
  --steps 250 --turns 3 --group-size 8 --micro-batch 4 \
  --no-sleep --gpu-mem-util 0.35 \
  --sync-every 1 --save-every 10 \
  --hint-probe --eval-every 25 --eval-n 235 \
  --eval-benchmark data/state_tests/eval_items.jsonl \
  --seed 0 --wandb
