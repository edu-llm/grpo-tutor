#!/bin/bash
#SBATCH --job-name=rescore
#SBATCH --output=logs/rescore_%j.out
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#
# Is the STUDENT the bottleneck, or is the answer channel?
#
# No dialogues are generated. The same transcripts - 149 hand-written and 600
# from the v2 policy, on the same questions - are re-read by 0.5B, 3B and 7B
# students, through both the log-prob channel used for the reward and a
# reasoning channel that gets a chat template and room to think.
#
# The number to read is the GAP between hand-written and policy inside each cell.
# It is +0.00 today with 0.5B on log-prob. If it opens up as the student grows,
# the student was the limit. If it stays flat everywhere, the task cannot
# register teaching quality and no reward design fixes that.
set -euo pipefail
source scripts/env.sh
export HF_HUB_OFFLINE=1
python -u src/rescore_students.py \
  --good runs/good_traces_all.jsonl \
  --policy runs/gen_traces.jsonl \
  --students Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct \
  --max-policy 600 \
  --out runs/rescore.json
