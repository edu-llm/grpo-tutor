#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -J persona
#SBATCH -o logs/persona_%j.out
# 1) generate kid-like replies for REAL dialogue contexts with a 7B model
# 2) LoRA-SFT the 0.5B student on them
set -euo pipefail
source scripts/env.sh
python -u src/persona_data.py --n 1500
python -u src/sft_student.py --epochs 3
