#!/bin/bash
#SBATCH --job-name=grpo_real
#SBATCH --output=logs/train_%j.out
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#
# --requeue: on preemption Slurm requeues this job automatically; train_h100.py
# resumes from checkpoints/ckpt.pt (LoRA + optimizer + step), so progress is kept
# with NO manual resubmission. --open-mode=append keeps the log across requeues.
set -euo pipefail
source scripts/env.sh
export WANDB_PROJECT=grpo_tutor
export WANDB_ENTITY=eduLLM
python -u src/train_h100.py \
  --backend vllm --reward real \
  --teacher Qwen/Qwen2.5-3B-Instruct \
  --student Qwen/Qwen2.5-0.5B-Instruct \
  --steps 300 --group-size 8 --micro-batch 4 --sync-every 1 --save-every 10 --wandb
