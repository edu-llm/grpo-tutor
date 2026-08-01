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
# PENNSYLVANIA IS DELIBERATELY ABSENT. train_v1.sbatch evaluates on the 235 PA
# items in data/state_tests/eval_items.jsonl, and a held-out set is only held out
# if it is not also trained on. CA+TX+MA+NJ give 687 items across grades 3-11.
#
# All grades are kept by choice. A 0.5B student is near chance on the 192
# high-school items, but the baseline screen below decides that empirically -
# anything it can already answer is dropped whatever grade it carries.
#
# The first run of this screen (19403509) kept only 246/687 on a single argmax
# plus one free-text sample, and called items "known" as often at grade 11 as at
# grade 3 - which is what luck looks like, not knowledge, on 4-way items with a
# 25% floor. Hence 3 free-text trials (all must land) and a recorded margin.
#
# The earlier attempt (19392547) died in 6s because the state item files had
# never been rsynced to the cluster. Check they are present before blaming code.
set -euo pipefail
source scripts/env.sh
export HF_HUB_OFFLINE=1      # use the pre-downloaded scratch cache
python -u src/baseline_screen.py \
  --states ca tx ma nj \
  --student Qwen/Qwen2.5-0.5B-Instruct \
  --min-grade 3 --max-grade 12 \
  --free-trials 3 --seed 0 --margin 0.0 \
  --report data/state_tests/screen_report.jsonl \
  --out data/state_tests/train_items.jsonl
