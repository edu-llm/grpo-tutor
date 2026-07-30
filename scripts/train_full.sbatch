#!/bin/bash
#SBATCH --job-name=grpo_full
#SBATCH --output=logs/full_%j.out
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#
# Long run: enough steps to see whether reward climbs, and whether it climbs by
# teaching or by hacking. The eval is the point - every 25 steps it reports
# teaching gain against a real baseline (QASC, unfiltered), the leak probe with
# its choices-only floor, and the swapped-hint specificity check.
#
# --requeue: on preemption Slurm resubmits and train_h100.py resumes from
# checkpoints/ckpt.pt, so progress survives. --open-mode=append keeps the log.
set -euo pipefail
source scripts/env.sh
export WANDB_PROJECT=grpo_tutor
python -u src/train_h100.py \
  --backend vllm --reward real \
  --teacher Qwen/Qwen2.5-3B-Instruct \
  --student Qwen/Qwen2.5-0.5B-Instruct \
  --steps 500 --group-size 8 --micro-batch 4 \
  --sync-every 1 --save-every 10 \
  --hint-probe --eval-every 25 --eval-n 40 --eval-benchmark qasc \
  --wandb
