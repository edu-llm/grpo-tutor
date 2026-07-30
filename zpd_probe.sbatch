#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J zpd_probe
#SBATCH -o /home/zsophia/orcd/scratch/grpo_tutor/logs/zpd_%j.out
source /home/zsophia/orcd/scratch/grpo_tutor/env.sh
export HF_HUB_OFFLINE=1      # use the pre-downloaded scratch cache
python -u zpd_filter.py --limit 300 --student Qwen/Qwen2.5-0.5B-Instruct --out data/zpd_problems.jsonl
