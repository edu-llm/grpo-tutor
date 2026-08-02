#!/bin/bash
#SBATCH --job-name=gen_nopersona
#SBATCH --output=logs/gennp_%j.out
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#
# Generate labelling traces from the v2 policy, with two fixes to the thing that
# made v2's traces useless to a human reader: the student had nothing to say, and
# nothing it could be moved off.
#
#   student state   grade + the option choose() actually picks. Every training
#                   item was screened on that pick being wrong, so it is a real
#                   named misconception, not an invented one.
#   student stops   the STUDENT ends the dialogue with [READY] when something
#                   clicks. The old stop consulted gold, which meant stopping
#                   measured reachability rather than teaching.
#
# Each dialogue is scored twice - plain and anchored to the stated belief. The
# anchored channel starts at 0 by construction, so unlike the plain one (flat
# across three runs) it has headroom for a tutor to move.
#
# No training happens here. 307 problems x 4 = 1,228 dialogues.
set -euo pipefail
source scripts/env.sh
export HF_HUB_OFFLINE=1
python -u src/gen_traces.py \
  --problems data/state_tests/train_items.jsonl \
  --teacher Qwen/Qwen2.5-3B-Instruct \
  --teacher-adapter checkpoints-v2/teacher-final \
  --student Qwen/Qwen2.5-0.5B-Instruct \
  --persona-adapter none \
  --k 2 --turns 3 --batch 64 \
  --gpu-mem-util 0.40 --seed 0 \
  --out runs/gen_traces_nopersona.jsonl
