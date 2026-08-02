# Run v3 — a learned teaching score in the reward

Job `19459983` · 250 steps · state assessments, Pennsylvania held out ·
**running, results below are empty until it lands**

v2's setup with one addition. Everything else — 307 screened CA+TX+MA+NJ items,
all 235 Pennsylvania items as eval, specificity reward, persona adapter,
kl_coef 0.03, 250 steps, seed 0 — is unchanged.

## Setup

| | |
|---|---|
| teacher | Qwen2.5-3B-Instruct + LoRA r=32, KL coef 0.03 |
| student | Qwen2.5-0.5B-Instruct, frozen, persona adapter on `reply()` only |
| task | 307 screened state-assessment items, all trained on |
| reward | `-1` if the rule says leaked, else solved − solved(other problem) **+ 0.5 · z(teaching score)** |
| eval | every 25 steps on all 235 Pennsylvania items |
| new | `--teach-head checkpoints/rm_head_linear.pt --teach-coef 0.5` |

## Why

v2 measured the same thing two ways and got answers two orders of magnitude
apart. Hand-written expert tutoring beat the trained policy by:

```
            student's answer channel   human-style raters
gap              +0.07                    +1.75  (of 5, t=8.3)
```

The objective has been optimising the left column. The right column is what
anyone means by teaching. This run puts a model of the right column into the
reward.

## The reward model

Trained on **1,104 rated tutor turns** — every turn in the labelling bundle.
Ratings came from the labelling app (human), one lead agent, and six independent
agents working from a shared rubric with worked calibration examples
(`data/label_slices/RUBRIC.md`). Where raters overlapped, the target is their
mean, not a vote: these are ordinal judgements, so 3 and 5 genuinely averages to
4, and a majority would discard the disagreement.

The six agents never saw each other's work and landed within 0.11 of each other
on the policy tier (2.48–2.59 of 5), which is the evidence that the rubric
transfers rather than each rater inventing a private standard.

Architecture, and why each part:

| choice | reason |
|---|---|
| frozen backbone | GRPO moves the teacher's LoRA every step; a head reading a moving trunk scores a moving target |
| final hidden state | attention is causal, so only the last position has read the whole tutor turn; mean-pooling biases towards early tokens |
| **linear** head, not MLP | the MLP memorised — train ρ 0.976 against test 0.55 — and its AUC swung 0.87–0.94 across splits. The linear probe holds AUC 0.918–0.933 with a third of the variance |
| 0.5B backbone | 3B and 7B were no better: test AUC 0.930 / 0.930 / 0.917 for 0.5B / 3B / 7B. Capacity buys nothing here, and 0.5B is cheap enough to run inside every step |

Held-out performance, on questions never seen in training or model selection:

```
test goodness spearman   +0.65
test AUC, hand-written vs policy   0.918
same-question pairwise ranking     0.72
```

That last number is the one that matters and the one that is easy to get wrong.
Global AUC pools comparisons across different problems, which GRPO never makes —
its advantage is centred inside a group of completions for ONE problem. A head
could score well globally by learning that maths items rate higher than history
items and be useless inside a group. 0.72 on same-question pairs is the honest
figure.

## How it enters the reward

```
reward = -1                              if the leak rule fires
       = solved − solved(other) + 0.5·z  otherwise
```

- **z-scored within the group.** Only the spread across one problem's eight
  completions survives GRPO's centring. Raw scores would let a problem the head
  rates high everywhere dominate the batch with variance the algorithm discards.
- **Scored one tutor turn at a time**, each against the conversation as it stood
  before that turn, and averaged over the dialogue's turns. The head was fitted on
  exactly that pair — a single turn rated in the context it answered — and the
  second v3 attempt fed it the finished transcript with all three tutor turns
  concatenated as the "message to rate": 1,910 characters against a trained-on 616,
  with the rated text also duplicated inside the transcript it was supposedly
  being rated against. A linear probe on a frozen backbone has no way to be right
  about input of a shape it never saw, so that run's teaching term was noise
  wearing a reward's clothes.
- **A leaked turn keeps its flat −1** and never receives the bonus. Paying for
  quality on top of a leak rewards a well-phrased give-away.
- **`solved` stays in.** It is the one term the head cannot be gamed against. If
  the teaching score climbs while held-out `clean_solved` stays flat, the head is
  being hacked, and that shows up within one eval cycle.
- **coefficient 0.5**, because `solved` has a within-group standard deviation
  near 0.4 — this makes the two terms comparable rather than letting either
  dominate.
- **Floored at −0.95**, strictly above the leak penalty. The first attempt at
  this run (job `19459983`, killed at step 14) had no floor, and 8.2% of
  non-leaking turns scored below −1: the specificity term gives −1 when the hint
  solved the OTHER problem and not its own, and a negative teaching score stacked
  on top. On those turns the policy was being told that giving the answer away
  would have scored better, which inverts the one invariant `LeakGuard` exists to
  enforce. Any non-leaking turn must beat any leaking one.

## What we know about the leak rule going in

Calibrated against the same 1,104 ratings (`docs/leak_calibration.md`), and it is
worse than it looked:

```
                 precision  recall
current rule       0.397     0.665
```

When it fires it is wrong about 60% of the time, and re-scoring v2's traces under
a corrected rule puts the true leak rate at 0.255 rather than 0.384. About a
quarter of v2's dialogues were penalised for a leak that did not happen.

Two things it got right, though. The **trend survives** — under both the old and
corrected rules v2's leak rate falls by the same proportion across the run
(0.47 → 0.35 old, 0.33 → 0.21 corrected), so the policy really did learn to leak
less rather than merely learning the detector's vocabulary. And `identifying` is
carrying the whole detector: `elimination` sits at AUC 0.493, which is chance.

**The rule is changed for this run**, reversing an earlier decision to leave it
alone for comparability with v2. Two things forced it.

The comparability argument was weaker than it looked: v2's traces are on disk, so
re-scoring them under the new rule recovers the comparison exactly, whereas a
reward term cannot be un-applied after the fact. Paying for comparability with a
known-broken training signal is the wrong side of that trade.

And the cost is not noise, it is bias. The penalty **overrides** the whole
reward — a flagged completion loses its outcome and teaching terms entirely — so
a false positive does not blur the gradient, it deletes it. Measured on 576 live
turns from the first v3 attempt, the four-signal rule flagged 53.3% and the new
rule flags 37.7%: **15.6% of all turns were being pinned to the floor for
nothing.** They look like this:

```
gold "22 words per minute"   distractors "18/25/30 words per minute"
turn "If he needs to type 550 words and has 25 minutes, how many words
      should he type each minute?"          overlap 1.0, elim 1.0 → -1
```

Textbook Socratic questioning, scored as a give-away, because `_content()` drops
tokens of two characters or fewer and every option collapses to
`{word, per, minut}`.

So the reward now uses `verbatim OR identifying_hits ≥ 1`: the rule five-fold CV
picked in 5 of 5 folds, worth +0.066 F1 held out and more than doubling precision
on math (0.191 → 0.476). `overlap` and `elimination` are still computed and
logged, just not wired to the penalty; `LeakGuard(use_overlap=True,
use_elimination=True)` restores the old behaviour. Still queued: `_content()`
should stop stripping digits.

**v2 and v3 leak rates are therefore not directly comparable** — re-score v2's
traces under the new rule before putting the two numbers side by side.

## Results — stopped at step 105. The teaching score is mostly a tier detector.

Job `19461561`, killed deliberately. Four held-out evals, 25 steps apart:

```
step   teacher_acc   clean_solved   gain     leak
  25      0.536         0.413      +0.047   0.230
  50      0.540         0.404      +0.051   0.238
  75      0.583         0.404      +0.094   0.251
 100      0.553         0.413      +0.064   0.217
```

`clean_solved` does not move — every value is inside ±0.032, one standard error
on 235 items. Meanwhile the training-side teaching score climbs hard, −0.190 →
+0.358 over 100 steps, about 1.3 within-group standard deviations. That is the
failure this document said to watch for, so the run was stopped rather than
finished.

**What the policy actually learned.** Comparing tutor turns at steps 0-19 against
steps 80+:

```
                        early   late
turns asking a question  79%    32%
mean turn length        209ch  171ch
```

It stopped asking questions. The highest-scoring recent dialogue (+2.15) works
the arithmetic through to "making it 14" and is flagged as a leak.

**Why.** `src/rm_tier_confound.py`. The label set is two tiers that differ on far
more than quality:

```
          n     goodness   question-rate
good     447      4.26         62%
policy   657      2.55         95%
```

Tier is readable straight off the embedding at spearman +0.833, and it predicts
the label nearly by itself, so the head can score provenance instead of reading
the tutoring. It does:

```
goodness, pooled over both tiers      val spearman  +0.559  [+0.526, +0.600]
goodness, within the policy tier      val spearman  +0.211  [+0.155, +0.262]
```

Two thirds of the apparent signal is provenance. And the policy tier is the only
regime that exists at RL time — every turn GRPO scores is policy-generated — so
the head is applied exactly where its main learned direction is constant. What
remains is style. The shipped head correlates −0.428 with containing a question
and −0.500 with length: it is, in the regime it is used, close to "be short and
do not ask anything", which is what the policy dutifully became.

The AUC 0.932 reported in *The reward model* above is not wrong, it is the wrong
measurement — it separates hand-written from policy text, which is not a task
anyone needs done.

This does not clear the reward model of being hackable; it never got the chance
to be hacked, because the direction it points is bad from step 0.

### Refitting within tier — how much signal is actually there

`src/rm_ridge.py`. Ridge, penalty swept on held-out questions, 20 seeds, scored
on questions absent from both fits. `within-question` is the only comparison GRPO
makes: the score is z-scored inside one problem's group, so ranking across
problems is discarded before it reaches the advantage.

```
                        n     test spearman            within-question
both tiers pooled     1104   +0.649 [+0.534, +0.725]        0.760
policy tier only       657   +0.222 [+0.042, +0.444]        0.581
hand-written only      447   +0.490 [+0.381, +0.639]        0.748
```

So there is real signal inside the policy tier, and it is weak: **0.581 against
0.500 for a coin flip**, over 3,216 pairs. The head reads quality well among
hand-written turns (0.748) and poorly among the policy turns it will actually be
asked about — unsurprising once you notice the policy tier's ratings are
compressed (mean 2.55) where the hand-written ones spread up to 4.26.

Not to be confused with `train_rm.py --tier policy`, which reports test spearman
+0.05. That fits 896 parameters to ~450 turns with AdamW and drives the train
loss to 2e-4; it is pure memorisation, and says nothing about whether signal
exists. Regularisation is doing the work here, and the penalty ridge selects is
large.

`checkpoints/rm_head_policy_ridge.pt` holds this head, written in
`TeachingScorer`'s format.

**Sizing `teach_coef` for v4.** 0.5 was chosen for a head believed to rank at
0.93. Scaling by how far each sits above chance, (0.581 − 0.5) / (0.93 − 0.5)
≈ 0.19, puts the honest coefficient near **0.1**. A term that is right 58% of the
time should not be allowed to move the reward as far as one that is right 93% of
the time, and it is the same z-scored quantity either way.

The alternative worth more than the refit: labels whose tiers are not separable
by style — one generator, matched length, quality varying — so provenance carries
no information and the head has nothing to shortcut to.

Either way, report within-tier ranking. Never pooled AUC.

### To fill in on the next attempt

```
step   teacher_acc   clean_solved   leak_rate   specificity   teach_score
```

plus training-side reward, `zero_adv_frac`, KL, and the check that decides
whether this worked:

- does `eval/clean_solved` move, for the first time in four runs?
- does the teaching score climb while `clean_solved` stays flat? That is the
  reward model being hacked, and it is the failure mode to watch for.

## What to watch

- **`zero_adv_frac` should fall.** A dense continuous term gives groups more ways
  to differ than a near-binary solve does. v2 sat at 0.27.
- **Traces, not just curves.** The head was fitted on the policy's current
  distribution. As the policy moves, the head is extrapolating, and the samples
  are the only place that becomes visible early.
