# Run v3-leakfix — the corrected leak rule, on its own

Job `19468005`, COMPLETED, 250 steps in 2h17m · `checkpoints/v3lf` ·
`runs/20260801-194043`

v2 exactly, with one change: `overlap` and `elimination` no longer fire the leak
penalty. No learned teaching score — v3's first attempt showed that head is
largely a provenance detector (`docs/run_v3.md`), so this run isolates the one
change with independent evidence behind it.

The question: v2's leak rule paid `-1` on 17.9% of turns to punish a 6.4%
behaviour, and the penalty **replaces** the reward rather than adjusting it. On
576 live turns the old rule flagged 53.3% against this one's 37.7%. How much of
four runs' worth of flat teaching was the policy being punished at random?

## Answer: none of it

```
step             25    50    75   100   125   150   175   200   225
teacher_acc    .549  .536  .502  .562  .532  .566  .566  .532  .532
clean_solved   .391  .357  .409  .451  .413  .443  .434  .413  .434
leak_rate      .234  .281  .187  .200  .209  .204  .217  .221  .174
hint_only_leak .494  .532  .438  .477  .443  .523  .477  .438  .472
```

Nothing is significant over 225 steps:

```
                 change    t
teacher_acc      +0.008  +0.30
clean_solved     +0.057  +1.98
leak_rate        -0.053  -1.64
hint_only_leak   -0.037  -0.90
```

`teacher_acc` is the one that matters — how often the student answers correctly
after tutoring, independent of any leak rule — and it is flat to three decimal
places. **That is four runs in a row.** Removing a false penalty from 15.6% of
turns did not move it, so the noise in the leak rule was not what was holding
teaching back.

## clean_solved, again

It is the only metric that comes close (t = +1.98, p ≈ 0.09), and it is the same
artefact v2 documented. Split into its two factors:

```
step           25    50    75   100   125   150   175   200   225
clean share  0.77  0.72  0.81  0.80  0.79  0.80  0.78  0.78  0.83
solved|clean 0.51  0.50  0.50  0.56  0.52  0.56  0.55  0.53  0.53

clean share                        +0.053  (t=+1.64)
solve rate inside clean dialogues  +0.038  (t=+1.45)
```

Neither factor is significant on its own; the product looks better than either
because both drift the same way. This is weaker evidence than v2's, where the
clean share carried the whole rise at t = +2.31.

Note `clean_solved` starts at .391 here against v2's .298. That is not
improvement — a narrower leak rule disqualifies fewer dialogues, so the metric
is mechanically higher. **v2 and v3-leakfix `clean_solved` and `leak_rate` are
not comparable.** `teacher_acc` is, and it did not move in either.

## What the fix did buy

The measured leak rate now means something closer to what it says. Precision on
math went 0.191 → 0.476, and the population-weighted flag rate went 17.9% → 10.9%
against an estimated 6.4% true give-away rate. The reward stopped pinning turns
like "how many words should he type each minute?" to the floor. That is a real
improvement in the instrument. It is not an improvement in the tutor.

## The policy stayed healthy

Worth recording, because v3's learned score wrecked exactly this:

```
                question-rate   mean chars   solved   leak
steps 0-24           81%           210        0.340   0.331
steps 225-249        83%           198        0.269   0.230
```

Stable and still Socratic. The collapse to 32% questions in v3 was caused by the
teaching term, not by anything else in the setup.

## Where this leaves the project

Four runs — v0 OpenBookQA, v1 controlled, v2 state assessments, v3-leakfix — have
now failed to move `teacher_acc`, under three different reward shapes and two
corpora. The remaining explanations are not about the reward:

- **The student cannot use hints.** The leading hypothesis. Hand-written expert
  tutoring beat the policy by only +0.07 on the student's answer while raters put
  the same gap at +1.75 on a 5-point scale — if expert teaching barely moves a
  0.5B student, no reward shaped from that student's accuracy can either. The
  rescore across student sizes speaks to this directly.
- **The outcome channel is too coarse.** One binary per dialogue, 32 dialogues a
  step. A learned dense signal was the intended fix and needs labels whose tiers
  are not separable by style before it can work.

Fixing the leak rule was necessary and is done. It was not sufficient, and it
was never going to be.
