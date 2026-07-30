#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J eval_base
#SBATCH -o logs/eval_base_%j.out
source scripts/env.sh
export HF_HUB_OFFLINE=1
# BASELINE: untrained 3B teacher on the held-out ZPD split.
python -u src/evals.py --problems data/zpd_problems.jsonl \
  --teacher hf --teacher-model Qwen/Qwen2.5-3B-Instruct \
  --student-model Qwen/Qwen2.5-0.5B-Instruct \
  --out runs/eval_base_3b.json
