# grpo_tutor

A hand-rolled GRPO stack that trains a **teacher** LLM to tutor a frozen **student**
LLM. The teacher sees a question the student cannot solve and writes a short hint
(never the answer); the student retries; the reward is whether it now gets it right.
Runs on 1x H100 with vLLM.

---

## Biggest things this accomplishes

**1. It proved the reward signal exists before spending any training compute.**
The whole idea only works on problems the student fails alone but solves with help.
`zpd_filter.py` measured that empirically across 4,957 items (baseline 37.95% ->
47.53% with an oracle hint) and curated the **731 problems** where the gap is real.
Without this the reward would have had no gradient and no amount of RL tuning
would have shown it.

**2. A working GRPO implementation, written from scratch.**
Group-relative advantage, clipped importance ratio, KL-to-reference via LoRA
disable, teacher-token masking - batched, micro-batched, with the frozen reference
cached once per step (halves the forward passes) and fused `cross_entropy` instead
of materializing a `(B,L,V)` log-softmax.

**3. Production training infra on a single GPU.**
vLLM colocated with the trainer via sleep/wake (**0.3s**, frees 30GB), LoRA weight
sync through a RAM disk (**0.3s**, so syncing *every step* is affordable and
training stays fully on-policy), preemption-safe checkpoint/resume, and a wandb run
that stays continuous across Slurm requeues.

**4. Reward hacking is measured and actively suppressed - not assumed away.**
`LeakGuard` makes leaking the answer (-1.0) strictly worse than honestly failing
(0.0). Five live detectors (leakage, repetition, group collapse, reward-length
correlation, length growth) plus a transfer check. This paid off concretely:
giving the teacher a proper system prompt in its chat format cut leaking
**~4x (0.70 -> 0.16)** and flipped mean reward from negative to positive.

**5. Clean seams for other people.**
`interfaces.py` is pure types - a teammate can implement `Student` or
`RewardModel` and drop it in without touching the training stack. Any reward model
wrapped in `LeakGuard` inherits leak protection for free.

---

## Measured results

| what | number |
|---|---|
| ZPD headroom (4,957 items) | 37.95% -> 47.53% (**+9.58%**) |
| Curated training set | **731** problems (0% -> 100% by construction) |
| Untrained 3B teacher | captures **42.7%** of ceiling, 2.7% leak (greedy) |
| Base vs instruct student | +14.0% vs +13.0% - a wash, so instruct kept |
| Throughput | ~4.6s/step (2.2 gen + 2.2 update + 0.3 sync) |

## Run

Everything below assumes **one** H100.

```bash
python src/zpd_filter.py --limit 5000    # 1. build the ZPD set (do this first)
sbatch scripts/train_real.sbatch         # 2. train (H100, preemptable, auto-resumes)
python src/evals.py --teacher-adapter checkpoints/adapter-latest   # 3. evaluate
python src/train_h100.py --backend stub --steps 6                  # no-GPU smoke test
```

Useful flags: `--turns 3` (multi-turn dialogue), `--hint-probe` (leak probe),
`--eval-every 25 --eval-n 30` (held-out benchmark during training),
`--eval-benchmark qasc` (evaluate on an unfiltered external set).

### Choosing an eval set

`python src/benchmarks.py --list` shows the available sets;
`python src/bench_baseline.py` measures what the student scores on each. Measured
for Qwen2.5-0.5B-Instruct over 150 items:

| set | chance | baseline | oracle hint | headroom |
|---|---|---|---|---|
| qasc (8-way) | 0.125 | 0.253 | 0.893 | **+0.64** |
| sciq | 0.250 | 0.687 | 0.970 | +0.28 |
| obqa_test | 0.250 | 0.313 | 0.413 | +0.10 |
| arc_easy | 0.249 | 0.533 | - | - |
| geography | 0.250 | 0.427 | - | - |
| commonsense | 0.200 | 0.393 | - | - |
| arc_challenge | 0.251 | 0.353 | - | - |

**qasc** is the default recommendation: the largest teachable gap by far, 8-way so
guessing adds less noise, and its answers require composing two facts - which is
work a tutor can actually do across turns rather than leak in one line.

## Layout

```
src/      the code            scripts/  slurm jobs + env.sh
data/     problems + personas runs/     traces, metrics, samples (gitignored)
```

`config.py` knobs · `engine.py` vLLM/HF/stub inference + sleep/wake/sync ·
`grpo.py` the GRPO loss · `tasks.py` problems + prompts · `rewards.py`
SolveReward + LeakGuard · `zpd_filter.py` ZPD curation + student ·
`evals.py` held-out eval · `monitor.py` traces + metrics + hack detectors ·
`train_h100.py` training loop · `interfaces.py` seams · `paths.py` repo-root anchors

Data and output paths are anchored to the repo root, so scripts work from any
working directory - a requeued job cannot miss `checkpoints/` and silently
restart from step 0.

## Watching a run

- **wandb** - loss/reward/KL/`zero_adv_frac`/hack metrics, plus a `samples` table of
  actual hints. Live, no port-forwarding.
- **`tail -f runs/*/samples.md`** - best/worst hints with the student's answer.
- **`traces.jsonl`** - every sample; `grep "HACK?" logs/train_*.out` for alerts.

Key metrics: `leak_rate` (should stay low), `solved_rate` (should rise),
`zero_adv_frac` (groups with no reward variance contribute **zero** gradient - if
this nears 1.0 the data has stopped providing signal), `clip_frac` (near 0 early).

The one that actually separates learning from hacking is `eval/teaching_gain` on
held-out problems the teacher never trains on. If training reward climbs while
that stays flat, the reward is being gamed.

## Known open issues

- **Generic-filler hack is unsolved.** At eval, ~70% of the teacher's gain survived
  swapping hints between problems, i.e. much of it is not question-specific. No
  detector catches this; the check is `teaching_gain` vs `transfer_gain` in
  `evals.py`. A specificity-adjusted reward (`solved(own) - solved(swapped)`) is the
  proposed fix and is **not implemented**.
- **No full training run has been done yet** - only smoke tests up to 6 steps. The
  encouraging early numbers are far too short to call a trend.
- **`eval/teaching_gain` equals `eval/teacher_acc` on the default eval set.** The
  ZPD filter keeps only problems the student fails alone, so held-out
  `baseline_acc` is 0.0 by construction and the baseline term subtracts nothing.
  Pass `--eval-benchmark qasc` (or any set above) to get a real baseline.
- **The leak probe is missing its control and is currently uninterpretable.**
  `hint_only_leak` (0.33-0.40 on QASC) is a partial-input baseline, and we never
  measured what the student scores on *choices alone*. LLMs beat majority
  baselines on choices-only prompts routinely, so an unknown share of that number
  is the student exploiting the distractors, not the teacher leaking. It also
  cannot distinguish leaking from legitimately nudging a student who already
  half-knew the answer. See `docs/eval_leakage.md` for the controls needed.
- **Multi-turn is verified in stub mode only.** `--turns > 1` runs the dialogue loop
  (student asks -> teacher guides -> student retries), masks loss to teacher tokens,
  and shares the terminal reward across turns, but it has not yet run on a GPU.
- Multi-GPU support was **removed**, not just disabled: 2-GPU queue times made it
  untestable, and untested concurrency code is worse than none. Single GPU is the
  only configuration.
