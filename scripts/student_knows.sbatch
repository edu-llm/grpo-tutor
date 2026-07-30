#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:25:00
#SBATCH -J stu_knows
#SBATCH -o logs/stuknows_%j.out
source scripts/env.sh
python -u src/check_student_knows.py --n 120
