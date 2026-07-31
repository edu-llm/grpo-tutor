#!/bin/bash
#SBATCH --job-name=grpo_mt
#SBATCH --output=logs/mt_%j.out
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#
# Multi-turn long run. A 4-step smoke measured leak 0.62-0.91 per step against
# 0.03-0.38 single-turn, so nearly every dialogue starts out earning the -1
# leak penalty. The question this run answers is whether GRPO trains that back
# down, and whether teaching gain rises with it or stays flat.
#
# --no-sleep: the engine stays resident instead of being offloaded between
# phases. Sleep/wake died at step 5 with an OOM inside wake_up() - vLLM could not
# reclaim its arena once the trainer's allocator had grown, and multi-turn makes
# that 3x worse. Keeping it resident removes the failure mode entirely and saves
# the cycle; gpu_mem_util drops to 0.40 since vLLM never hands the memory back.
#
# ~19s/step (3x the teacher turns and 3x the training samples of single-turn),
# so ~3h for 500 steps plus evals. Checkpoints every 10 steps; preemption is
# expected and handled by --requeue.
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
  --steps 500 --turns 3 --group-size 8 --micro-batch 4 \
  --no-sleep --gpu-mem-util 0.40 \
  --sync-every 1 --save-every 10 \
  --hint-probe --eval-every 25 --eval-n 40 --eval-benchmark qasc \
  --wandb
