# Run v1-controlled — v0 plus the specificity reward

Job `19392772` · 250 steps · 17:07-19:55, 2h11m of compute across three
segments · 8,064 dialogues ·
[wandb](https://wandb.ai/eduLLM/grpo_tutor/runs/19392772) · code at `229d9bf`

## Setup


|          |                                                                                                            |
| -------- | ---------------------------------------------------------------------------------------------------------- |
| teacher  | Qwen2.5-3B-Instruct + LoRA r=32, KL coef 0.05, clip 0.2 — v0's exactly                                     |
| student  | Qwen2.5-0.5B-Instruct, frozen after SFT on more realistic student responses                                |
| task     | the same 549 dual-screened OpenBookQA problems, 467 train / 82 held out; 465 distinct problems seen        |
| reward   | `--specificity difference`: `-1` if the tutor leaked, else solved(own) − solved(other problem | same hint) |
| dialogue | up to 3 teacher turns, terminal reward, **dialogue stops early once the student can answer**               |
| eval     | every 25 steps on 200 unfiltered QASC items (v0 used 40)                                                   |
| infra    | H100 on `mit_preemptable`, preempted and requeued twice                                                    |




## Headline

**The specificity reward did not move specificity, and the run is not the clean
A/B it was designed to be.**

Every held-out eval, n=200 QASC items each:

```
step   baseline   teacher_acc   swapped_acc   specificity   teaching_gain
  25     0.260       0.485         0.365         0.120           0.225
  50     0.260       0.435         0.365         0.070           0.175
  75     0.260       0.515         0.365         0.150           0.255
 100     0.260       0.480         0.365         0.115           0.220
 125     0.260       0.465         0.410         0.055           0.205
 150     0.260       0.440         0.365         0.075           0.180
 175     0.260       0.465         0.400         0.065           0.205
 200     0.260       0.460         0.360         0.100           0.200
 225     0.260       0.455         0.375         0.080           0.195
```

Specificity: mean **+0.092**, sd 0.031, all nine points positive, fitted slope
**−0.037 ± 0.031 per 200 steps** — flat, or very slightly down. Against v0's
nineteen evals at n=40 (mean +0.013, sd 0.060, 9 of 19 positive) the *level* is
different; the *trajectory* is the same flat line v0 had.

Everything the reward gained, it gained by leaking less: reward rose
−0.271 → −0.048 while training leak fell 0.419 → 0.213, and the specificity term
the reward was built around improved on neither side of the split.

## The experiment, and why its result is confounded

The sbatch header says "v0 with EXACTLY ONE change ... if specificity moves, the
reward is why — there is nothing else it could be." That is not true of the run
that executed. Between v0 (`69154a2`, 11:24) and this one (`229d9bf`, 17:07) the
dialogue loop gained an **oracle early stop**: `stop_when_solved` ends the
conversation as soon as the student can answer. v0's 16,000 dialogues are all
exactly 3 turns; here:

```
turns      n     share   solved   specificity   reward
  1     2,801    34.7%   1.000       +0.739     +0.380
  2     1,007    12.5%   1.000       +0.726     −0.062
  3     4,256    52.8%   0.132       −0.151     −0.433
```

`solved` is **1.000 by construction** for the 47% of dialogues that stopped
early, because the stop condition *is* the solve check. It no longer means
"correct at the end", it means "correct at any checkpoint", so the reward's
own-problem term is a max over up to three chances. The swapped condition gets
no such max: it scores one fixed transcript written for another problem.
**Specificity = teacher_acc − swapped_acc is therefore biased upward by the
stop**, on the held-out eval as much as in training, since `heldout_eval` calls
the same loop. That is the most plausible explanation for +0.09 here against ~0
in v0, and it is not the reward. Nothing in this run separates the two: the
first eval is at step 25, already +0.120, and it never rises.

Training-side specificity, per trace from `solved − off_problem_solved`, says
the same thing more directly: +0.319 over steps 0-49, +0.231 at 100-149, +0.233
at 200-249, slope −0.094 ± 0.039. `off_problem_solved` — the term the reward
punishes — is flat at 0.27-0.29 start to finish. The policy never learned to
write hints that stop working on other problems.

## Leaking, and the v0 probe divergence

```
step   eval/leak_rate   hint_only_leak   above floor
  25       0.260            0.360          0.270
 125       0.140            0.320          0.230
 225       0.135            0.265          0.175
```

Both fell and both fits are real: −0.096 ± 0.025 for the rule and −0.071 ± 0.015
for the probe per 200 steps, against a constant 0.090 `choices_only` floor.

**The 5x divergence v0 reported did not persist, but it did not close either.**
In v0 the rule-based rate fell 77% while the probe sat at 0.400 all run. Here
the rule fell 48% and the probe fell 26% with it — the probe is finally moving,
but at step 225 it still reads 0.265 against the rule's 0.135, and
`leak_above_floor` (0.175) still exceeds the rule rate outright.

Read the level, not the change, with care: the leak rules themselves changed
between the runs (`7128710`, `a11b057`). Re-scored on v0's own 16,000 traces
that commit moved the leak label 0.223 → 0.266, so the two runs' leak numbers
differ by ~4% before any policy difference.

## Held-out teaching

From the table above: `baseline_acc` is a deterministic 0.260 at every eval, the
sanity check that neither the eval set nor the student moved. `teacher_acc`
(0.485 → 0.455) and `teaching_gain` (0.225 → 0.195) are flat within noise for
250 steps, both at slope −0.025 ± 0.025. The gain averages +0.207 against v0's
+0.100, but the oracle stop inflates it too, so it is not a clean improvement.

## Training-side behaviour

```
steps      reward   leak   solved   zero_adv    kl   entropy   tokens
  0-24     -0.271   0.419   0.604     0.090   0.002   1.093    3103
 50-99     -0.126   0.304   0.549     0.070   0.013   1.053    2713
150-199    -0.041   0.226   0.550     0.091   0.023   1.077    2612
200-249    -0.048   0.213   0.519     0.080   0.029   1.105    2637
```

- **Reward and leak are the only robust training trends** (t = +4.9 and −8.0).
KL climbed 0.002 → 0.029, the shape v0 had, still below its 0.040.
- `zero_adv_frac` **did not creep.** v0 went 0.105 → 0.162 and flagged it; here
it is flat at 0.081 overall (slope +0.017 ± 0.030). A three-valued reward
gives a group more ways to differ than a binary solve does — the obvious
candidate explanation, untested.
- Where the reward comes from, over all 8,064 dialogues: 27.3% leaked (−1),
27.3% scored +1, 10.3% scored −1 (the hint solved the *other* problem and not
its own), 35.1% scored 0. First half to second, the leak share fell
0.331 → 0.217 while the +1 share fell 0.393 → 0.361 and −1 held flat.
- Teacher turns stayed the same length (127 → 120 tokens); dialogue tokens fell
3103 → 2637, which is the early stop, not brevity. `hack/leakage` stayed 0.000.



## Solved-but-not-leaked, v0's honest teaching signal

v0's 0.32 is the *conditional* rate; recomputed from v0's 16,000 traces it is
0.328 overall. Here it is **0.512** overall — 0.548 over steps 0-49, 0.472 at
100-149, 0.496 at 200-249, no trend. The joint rate (solved *and* not leaked)
runs 0.337 → 0.391. The gap to v0 is again mostly the oracle stop: a dialogue
that halts the moment the student is right cannot then be talked out of the
answer, which is what v0's extra turns were doing.

## Preemption: is the series trustworthy?

Mostly yes, with one seam. Preemptions at 18:30:25 and 18:35:53; wandb segments
at 17:07, 18:32 and 19:11; both resumes loaded the same `ckpt.pt` at step 150.

- **No gaps and no rollback.** Steps 0-249 each appear once, except step 150,
logged three times — once per segment. The middle segment ran that one step
and died, which is why it holds no sample tables, and the duplicate is why
there are 8,064 dialogues rather than 8,000.
- **A one-step discontinuity at the seam.** Pre-preemption step 150 logged
kl 0.025 / clip 0.01; the two replays logged kl 0.065 / clip 0.17; step 151 is
back to 0.022 / 0.01. Optimiser state does not survive the restore cleanly,
but the effect is confined to that step.
- Eval ran at steps 25-225 only — **no step-0 and no step-249 eval** — so every
held-out claim here is about that interval.



## Readings that are not signal

Each has a story attached and none survives a fit:

- *"Specificity is decaying under its own reward."* −0.037 ± 0.031, t = −1.19
on nine points. Flat.
- *"The honest solve rate is finally rising."* Joint clean rate 0.337 → 0.391,
t = +2.2 — but 32 dialogues per step come from only 4 problems, so the
effective n is a quarter of what the fit assumes.
- *"Training specificity is falling, the reward is anti-optimising."* t = −2.4,
same clustering problem, and `off_problem_solved` is flat throughout.
- *"Entropy is climbing"* (t = +2.9, whole move 1.09 → 1.11) and *"the tutor is
getting terse"* (eval `hint_words` 64 → 58, t = −1.0).



## What this says about next

1. **Fix the eval before trusting another specificity number.** The own-problem
  condition gets an oracle max over three checkpoints and the swapped condition
   does not, so `eval/specificity` measures the stop as much as the hint. Either
   disable `stop_when_solved` inside `heldout_eval` or give the swapped
   condition the same checkpointing. Until then this run's +0.09 and v0's ~0 are
   not comparable quantities.
2. **A one-variable claim needs the code pinned, not just the flags.** Two
  behavioural changes — the oracle stop and the rewritten leak rules — landed
   between v0 and this run while the sbatch header asserted a single change.
3. **The corpus move is still the live hypothesis and it is ready.** 308
  screened CA+TX+MA+NJ items with 235 Pennsylvania items held out are staged in
   `data/state_tests/`, and the motivation survives all of the above: those golds
   run 4 words median and 19% single-word against OpenBookQA's 2 and 31%, so
   "explain the misconception" and "reveal the answer" stop being one act.
4. **250 steps was right.** Leak and KL had settled by ~150 as in v0, and the
  last 100 steps changed no conclusion.

