#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J screen
#SBATCH -o logs/screen_%j.out
#
# Drop state-test items the student can already answer unaided. v1 step 0 ran
# without this and paid for it: solved=0.97, zero_adv_frac=0.75, so three
# quarters of the groups had identical rewards and produced no gradient.
#
# PENNSYLVANIA IS DELIBERATELY ABSENT. train_v1.sbatch evaluates on `pa`, and a
# held-out set is only held out if it is not also trained on. CA+TX+MA+NJ leave
# 495 items in the grade 3-8 band before screening.
#
# The earlier attempt (19392547) died in 6s because the state item files had
# never been rsynced to the cluster. Check they are present before blaming code.
set -euo pipefail
source scripts/env.sh
export HF_HUB_OFFLINE=1      # use the pre-downloaded scratch cache
python -u src/baseline_screen.py \
  --states ca tx ma nj \
  --student Qwen/Qwen2.5-0.5B-Instruct \
  --min-grade 3 --max-grade 8 \
  --out data/state_tests/train_items.jsonl
