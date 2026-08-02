#!/bin/bash
#SBATCH --job-name=grpo_v3lf
#SBATCH --output=logs/v3lf_%j.out
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#
# v3-leakfix. v2 exactly, with ONE change: the corrected leak rule.
#
# The learned teaching score is deliberately ABSENT. v3's first attempt showed
# the head is largely a hand-written-vs-policy detector (spearman +0.559 pooled,
# +0.211 within the policy tier, which is the only tier that exists at RL time),
# and optimising it drove the question rate from 79% to 32% while held-out
# clean_solved sat flat. That term goes back in once the head is refitted within
# tier - see docs/run_v3.md.
#
# So this run isolates a change we have independent evidence for. The reward is
# now `verbatim OR identifying_hits >= 1`; `overlap` and `elimination` no longer
# fire it. Both were measured worthless on 1,104 rated turns - elimination sits
# at AUC 0.493, chance - and both collapse on numeric golds, where "8 hours" and
# every distractor reduce to {hour}. On 576 live turns the old rule flagged 53.3%
# and this one flags 37.7%: 15.6% of turns were being pinned to -1 for saying
# "words per minute".
#
# That matters more than a detector metric usually would, because the penalty
# REPLACES the reward rather than adjusting it - a false positive discards the
# outcome and specificity terms for that completion outright.
#
# The question this run answers: how much of v2's flat clean_solved was the
# policy being punished at random?
#
# Everything else is v2: 307 screened CA+TX+MA+NJ items, all 235 Pennsylvania
# items held out, specificity reward, persona adapter, kl 0.03, 250 steps, seed 0.
#
# NB --save-dir. Sharing a checkpoint directory with a finished run makes the new
# job resume that one and exit immediately; each experiment gets its own.
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
  --save-dir checkpoints/v3lf \
  --kl-coef 0.03 \
  --steps 250 --turns 3 --group-size 8 --micro-batch 4 \
  --no-sleep --gpu-mem-util 0.40 \
  --sync-every 1 --save-every 10 \
  --hint-probe --eval-every 25 --eval-n 235 \
  --eval-benchmark data/state_tests/eval_items.jsonl \
  --seed 0 --wandb
