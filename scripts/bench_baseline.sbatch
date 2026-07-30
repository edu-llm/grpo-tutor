#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -J bench_base
#SBATCH -o logs/benchbase_%j.out
source scripts/env.sh
python -u src/bench_baseline.py --n 150
