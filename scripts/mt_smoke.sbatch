#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 00:50:00
#SBATCH -J mt_smoke
#SBATCH -o /home/zsophia/orcd/scratch/grpo_tutor/logs/mt_%j.out
source /home/zsophia/orcd/scratch/grpo_tutor/scripts/env.sh
rm -rf checkpoints runs
python -u src/train_h100.py --backend vllm --reward real --turns 3 \
  --teacher Qwen/Qwen2.5-3B-Instruct --student Qwen/Qwen2.5-0.5B-Instruct \
  --steps 3 --group-size 4 --micro-batch 4 --sync-every 1 --save-every 2
