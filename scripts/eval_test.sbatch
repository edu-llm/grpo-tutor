#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 00:50:00
#SBATCH -J eval_test
#SBATCH -o /home/zsophia/orcd/scratch/grpo_tutor/logs/evaltest_%j.out
source /home/zsophia/orcd/scratch/grpo_tutor/scripts/env.sh
rm -rf checkpoints runs
python -u src/train_h100.py --backend vllm --reward real \
  --teacher Qwen/Qwen2.5-3B-Instruct --student Qwen/Qwen2.5-0.5B-Instruct \
  --steps 5 --group-size 8 --micro-batch 4 --sync-every 1 --save-every 99 \
  --hint-probe --eval-every 2 --eval-n 20
