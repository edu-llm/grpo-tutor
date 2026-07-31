# Run v0 — first clean multi-turn GRPO run

Job `19371176` · 500 steps · 2h57m · 16,000 dialogues ·
[wandb](https://wandb.ai/eduLLM/grpo_tutor/runs/19371176) · code at `69154a2`

## Setup

| | |
|---|---|
| teacher | Qwen2.5-3B-Instruct + LoRA r=32, lr 5e-6, KL coef 0.05, clip 0.2 |
| student | Qwen2.5-0.5B-Instruct, frozen, scores MC by length-normalised log-prob |
| task | 549 OpenBookQA problems, dual-channel ZPD screen (fails `choose()` **and** cannot free-text answer) |
| dialogue | 3 teacher turns, student replies in between, terminal reward |
| reward | `-1` if the tutor leaked, else `1` if the student then solved it, else `0` |
| eval | every 25 steps on 40 held-out QASC items (unfiltered, real baseline) |

## Headline

**The leak penalty works. Nothing in the reward improves teaching.**

```
step    leak   solved   clean-solved
   0   0.389   0.425      0.318
 166   0.220   0.407      0.342
 249   0.169   0.387      0.341
 416   0.187   0.368      0.320
```

Leaking halved (0.39 → 0.19; held-out 0.325 → 0.075). The solve rate among
dialogues that did **not** leak — the honest teaching signal — never moved:
0.32 ± 0.02 across 16,000 dialogues.

So the drop in overall `solved` is the leak-driven solves disappearing, not a
regression. The tutor stopped cheating and its genuine teaching stayed exactly
where it started. Held-out `teaching_gain` opened at +0.100 and closed at +0.100.

## The finding that matters most

**Specificity is approximately zero, and sometimes negative.**

```
step   teacher_acc   swapped_acc   specificity
  25      0.350        0.450         -0.100
 250      0.450        0.350         +0.100
 475      0.350        0.375         -0.025
```

`swapped_acc` is the student's accuracy when handed a hint written for a
*different problem*. It is statistically indistinguishable from the accuracy
with the correct hint. **The teacher's hint is worth no more than a random
other hint from the same policy.**

That reframes the +0.100 teaching gain: it is not weak teaching, it is *generic
help*. Something about receiving any tutor-shaped text lifts the student ~10
points, and question-specific content contributes nothing measurable. A reward
that pays only for "the student got it right" cannot tell these apart, which is
why 500 steps of optimisation left it untouched.

## Secondary observations

- **`hint_only_leak` never moved** — flat at 0.40 for the whole run while the
  rule-based leak rate fell to 0.075, a 5x divergence. With the choices-only
  floor at 0.125, `leak_above_floor` sat at a constant 0.275. Either the rules
  miss most leakage or the probe measures something else. Unresolved; needs the
  judge calibration.
- **`zero_adv_frac` crept up** 0.105 → 0.162. Groups where all 8 completions
  score identically contribute no gradient. Not yet a problem, worth watching.
- **Everything happened in the first ~150 steps.** Steps 150-500 confirmed a
  plateau. Future runs of this configuration can be much shorter.

## Two mid-run readings that were noise

Recorded because both looked like signal at the time and neither survived:

- *"Token count is falling, the tutor is going vacuous."* Over the full run
  tokens went 3561 → 3543. The apparent decline was a dip to 3293 around step
  200 that recovered.
- *"KL is decaying, the reference anchor is dragging the policy back."* KL was
  0.040 at the start and 0.038 at the end. Flat.

Both were read off ~100-step windows. At this batch size that is not enough to
distinguish a trend from oscillation.

## Bugs this run existed to flush out

Found and fixed before or during it; each was silently corrupting the numbers:

- `LeakGuard` scored the whole transcript, so the teacher was charged for the
  **student's** words — 17% of all leak flags.
- Student turns were truncated at 48 tokens mid-sentence, and the tutor then
  *completed the student's sentence*, once with the gold answer. 20% of turns.
- The student could see the answer options while speaking and read them aloud;
  since scoring used the transcript, it graded itself correct. Pure
  contamination fell 11% → 4% when the options were hidden.
- A quarter of the ZPD set was solvable by the student in free text. "Cannot
  solve" was a property of the log-prob channel, not of knowledge.
- vLLM `wake_up()` OOM'd at step 5 under multi-turn memory pressure; fixed by
  keeping the engine resident.

## What this says about v1

1. **A reward term for question-specific help is now the priority**, not an
   optimisation. Specificity ≈ 0 says the current objective is blind to the
   thing we care about. `solved(own) − solved(other problem)` — with the hint
   held fixed and the problem varied, so the term survives group normalisation.
2. **The task may be the ceiling.** OpenBookQA answers are single words, so
   "correct the misconception" and "reveal the answer" are the same action. But
   note QASC's `combinedfact` states the gold answer verbatim in 88.8% of items,
   so its apparent +0.64 headroom is mostly copying — `fact1` is the honest
   candidate and its real headroom is unmeasured.

   > **Corrected later — see `docs/dataset_choice.md`.** Two halves of this are
   > wrong. OpenBookQA answers are *not* mostly single words (median 2 words, 31%
   > single-word); QASC's are (1 word, 60%). And the 549 items this run trained
   > on were not representative of OpenBookQA: 28.2% of their hints contain the
   > answer against 10.1% in the pool, because the ZPD screen selects for exactly
   > that. The corpus was less at fault than the curation, and this run never
   > tested the hypothesis it was blamed on.
3. **Leak measurement needs calibrating** before the leak rate is quotable.
4. Shorter runs. 150 steps would have produced the same conclusions in an hour.
