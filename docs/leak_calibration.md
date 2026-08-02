# Leak detector calibration

1,104 rated tutor turns · `data/rm_dataset.json` + `label_app/data/label_items.json`
· detector `rewards.leaked_answer` at its shipped thresholds · numbers in
`data/leak_calibration.json` · reproduce with
`PYTHONPATH=src:scripts python scripts/calibrate_leak_eval.py` and
`scripts/calibrate_leak_errors.py`. Measurement only; `src/rewards.py` untouched.

## Setup

|          |                                                                                                                            |
| -------- | -------------------------------------------------------------------------------------------------------------------------- |
| detector | `verbatim ≥ 1` OR `overlap ≥ 0.6` OR `elimination ≥ 0.5` OR `identifying_hits ≥ 1`, evaluated on the tutor turn alone       |
| inputs   | `tutor_turn`, base64 `gold`, the other three `choices` as distractors, `question` — taken from `label_items.json`, not parsed out of `text` |
| labels   | leak 1-3 per `data/label_slices/RUBRIC.md`: 1 never points at an option, 2 hints, 3 states / restates / eliminates down to one |
| raters   | 1,027 turns rated by one LLM rater, 77 touched by a person, 54 double-rated                                                 |
| sample   | stratified on the rule's own decision boundary, plus 447 hand-written `good` turns; **not** a random draw from a run        |

## Headline

**The detector reproduces exactly, and it is one signal in a trenchcoat.**
`identifying_hits` carries it, `verbatim` adds four turns, `overlap` at 0.6 is
net negative, and `elimination` is anti-correlated with leaking. Deleting the
latter two is worth +0.066 F1 held-out on the give-away label, buying 11 points
of precision for 3 of recall. Reweighted to the run's real mix, `-1` is paid on
17.9% of turns while 6.4% actually give the answer away.

## The labels reproduce the code

Recomputing all four signals from the raw fields and re-running
`rewards.leaked_answer`:

```
turns                     1104
mismatches vs rule_flagged   0
```

`rule_flagged` was built against this code. Nothing downstream is confounded by
a version skew.

Rating distribution, and how "2" is handled:

```
leak  1.0   451      class 1 (no leak)        451
      1.5     9      class 2 (hints)          423
      2.0   409      class 3 (gives it away)  230
      2.5     5
      3.0   230
```

The 14 fractional means are split two-rater rows; they are binned to `hints`.
Dropping them instead moves the current detector's F1 by ≤0.004 under either
treatment, so nothing below rests on that choice. Everything is reported three
ways: **A** hints count as leaks, **B** hints count as clean, **C** hints
dropped and only the unambiguous 1-vs-3 turns scored.

## The current detector

```
treatment                 n   pos    P      R     F1    acc    MCC
A hints are leaks      1104   653  0.743  0.438  0.551  0.578  0.225
B hints are clean      1104   230  0.397  0.665  0.498  0.720  0.341
C hints dropped         681   230  0.607  0.665  0.635  0.742  0.437
```

Read A and B as the two questions they are. **A**: of the turns it flags, 74%
say something the raters thought a student could work from — but it misses 56%
of them. **B**: it catches two thirds of outright give-aways and 60% of its
flags are not give-aways.

Do not compare F1 across the two rows. Under A the positive class is 59% of the
sample, so flagging *everything* scores F1 0.743 — better than the detector and
better than anything below. F1 is a degenerate objective at that base rate; MCC
is the honest column for A.

Under A the current thresholds are near-optimal within this rule family: the
best MCC any OR-combination reaches is 0.242 against 0.225, and the bootstrap
CI on that difference includes zero. **The interesting question is B.**

## Which signal is doing the work

At the shipped thresholds, alone:

```
treatment B          fires    P      R     F1    MCC    AUC
verbatim ≥ 1            99  0.778  0.335  0.468  0.440  0.655
overlap ≥ 0.6          206  0.485  0.435  0.459  0.327  0.683
elimination ≥ 0.5       85  0.118  0.043  0.063 -0.064  0.493
identifying_hits ≥ 1   268  0.519  0.604  0.558  0.433  0.733
```

`elimination` is not a weak signal, it is not a signal. AUC 0.493 is chance;
its precision of 0.118 is *below* the 0.208 base rate, and its MCC is negative.
It fires 85 times, and the mean rating when it fires is 1.66 against a corpus
mean of 1.80 — it fires slightly more often on turns that do not leak.

`verbatim` is precise and nearly redundant: of its 99 fires, 4 are ones no other
signal caught. A tutor who quotes the gold string almost always trips
`identifying` on the way.

Dropping one member of the OR at a time, treatment B:

```
rule                      P      R     F1    MCC
current (all four)      0.397  0.665  0.498  0.341
without verbatim        0.396  0.657  0.494  0.336
without overlap         0.420  0.648  0.509  0.358
without elimination     0.409  0.657  0.504  0.350
without identifying     0.463  0.465  0.464  0.323
```

Removing `overlap` or `elimination` **improves** the detector. Removing
`identifying_hits` costs 20 points of recall.

## Sweeping

Best cutoff per signal, treatment B — barely anything moves:

```
signal              current   best   F1 at best
verbatim               1.0     1.0      0.468
overlap                0.6    0.55      0.467
elimination            0.5     1/3      0.139
identifying_hits         1       1      0.558
```

`verbatim` and `identifying_hits` are already at their F1-optimal cutoff;
`overlap` is one grid step away and gains 0.008 by moving. Only `elimination`
prefers a different cutoff, and at its best it still scores 0.139. **The
thresholds were never the problem; the membership of the OR is.** Exhaustive
search over all 15 subsets × their cutoffs:

```
treatment B                              P      R     F1     MCC
current                                0.397  0.665  0.498  0.341
verbatim OR identifying_hits ≥ 1       0.512  0.635  0.567  0.441
verbatim OR identifying_hits ≥ 2       0.744  0.404  0.524  0.471
```

`verbatim OR identifying_hits ≥ 1` is the best rule by F1 and the honest
recommendation. Paired bootstrap, 2,000 resamples:

```
                 ΔF1                  ΔMCC
B   +0.070  [+0.048, +0.092]   +0.101  [+0.071, +0.130]
C   +0.037  [+0.014, +0.060]   +0.084  [+0.049, +0.121]
A   −0.069  [−0.093, −0.047]   +0.017  [−0.019, +0.053]
```

Not an artefact of fitting on the evaluation data: five-fold cross-validation
selected the same `{verbatim, identifying_hits ≥ 1}` rule in **5 of 5 folds**,
for a held-out mean F1 of 0.557 against the current rule's 0.491, gain +0.066.

`identifying_hits ≥ 2` buys precision 0.744 at recall 0.404 and the best MCC of
anything searched. Which of the two to ship is a policy call about whether a
missed leak or an unearned `-1` costs more; the measurement does not settle it.

One non-OR family was checked and loses. Requiring k of 4 signals at the current
thresholds, treatment B: k≥2 gives F1 0.482, k≥3 gives 0.433, k≥4 gives 0.050.
Conjunctions with swept cutoffs were not searched.

## What the errors look like

99 false positives (flagged, rated 1) and 77 false negatives (clean, rated 3).
All of them, with text, are in `data/leak_calibration.json`.

**False positives are one word of topical echo, and arithmetic that has no
words.** 50 of 99 have exactly one `identifying` hit, and 38 involve
`elimination` — 28 of those on a gold containing a digit.

```
gold "Increased levels of voter participation"
turn "...what kind of voter supported him?"          ident 1/4  → flagged, rated 1

gold "8 hours"    distractors "12 hours" "27 hours" "72 hours"
turn "...3 workers working together can finish in less time..."
                                        overlap 1.000, elim 1.000 → flagged, rated 1
```

The second is a mechanism, not a coincidence. `_content()` drops tokens of two
characters or fewer, so `"8 hours"` reduces to `{hour}` and every distractor
reduces to `{hour}` as well. `overlap` becomes "did the tutor say the unit" and
`elimination` becomes "did the tutor say the unit, three times". 234 of 437 math
golds collapse to one content word or none; on math, turns where `overlap` fires
average a rating of 1.81 against the corpus mean of 1.80, i.e. zero
information. `identifying` has a numeric path (`_identifying_numbers`);
`overlap` and `elimination` never got one.

**False negatives are correct answers described instead of named.** 46 of 77
share no content word with gold at all — every signal reads exactly zero — and
they are 37 social studies, 33 science, 7 math.

```
gold "Roman Republic."
turn "Which ancient government split power between consuls, a senate and assemblies?"

gold "religious idea of eternal life."
turn "Almost all of it served the dead and the gods. It was made to last forever..."

gold "applying waterproof paint to the wood"
turn "Which option happens last and treats the surface rather than cutting or testing it?"
```

The third is elimination in substance — the three distractors are sawing,
drilling and testing, and "rather than cutting or testing" dismisses them as a
class — and the `elimination` signal misses it because
it demands that *every* content word of a distractor appear in the turn. The
rubric counts this as a 3. String rules cannot reach any of these; that is the
gap `hint_only_leak` exists to cover.

## By subject, and where the gain lives

```
subject          n     current B         recommended B
                      P      R     F1     P      R     F1
math           437  0.191  0.759  0.306  0.476  0.690  0.563
science        408  0.493  0.692  0.576  0.556  0.654  0.601
social_studies 259  0.475  0.606  0.533  0.479  0.596  0.531
```

The entire improvement is math. On math the current detector is worse than
useless as a precision instrument — 93 false positives against 22 true ones —
and dropping `overlap` and `elimination` more than doubles precision for 7
points of recall.
Science and social studies do not move at all.

Run v2 read math as the subject that leaked least (0.334 against science's
0.460) and best on every other axis. If math precision is 0.191, that 0.334 was
substantially false positives, and the gap between the subjects is wider than
v2 reported. Different run, so this is a prediction to check, not a correction.

## What this means for the reward

The label set oversamples the boundary, so the numbers above describe the
sample. Reweighting the 657 policy-tier rows by `label_key.population` — the
stratum counts of the run they were drawn from, hand-written turns excluded:

```
population-weighted     flag rate    P      R     F1
current, B (gives away)    0.179   0.262  0.732  0.386
current, A (hints+)        0.179   0.624  0.349  0.448
recommended, B             0.109   0.414  0.700  0.520
recommended, A             0.109   0.714  0.242  0.362
```

with estimated true rates of **6.4% gives-it-away** and **32.1% hints or worse**.

So the `-1` fires on 17.9% of turns to punish a 6.4% behaviour. About 62% of
penalties land on a turn that at least hints, and 26% on an outright give-away.
The recommended rule cuts the flag rate to 10.9% and raises give-away precision
from 0.262 to 0.414 while keeping 70% recall — under GRPO's within-group
normalisation, that is fewer members of each group pinned to the floor for
saying "voter".

## What this does not settle

- **The `hint_only_leak` disagreement is still open.** Zero of the 1,104 rated
  turns overlap the 310 dialogues in `review_app/data/analysis_key.json` that
  carry a probe reading, and those are v0 OpenBookQA dialogues besides. Nothing
  here scores the probe against a rating. The one thing this adds: the rule
  already over-flags give-aways by 2.8×, so the probe reading *above* the rule
  cannot be the rule missing give-aways — the probe has to be counting hints, or
  the choices-only floor. That is an inference across two different runs and two
  different populations, and it should be checked directly by rating turns that
  have a probe value.
- **These are mostly not human labels.** 1,027 of 1,104 turns were rated by one
  LLM rater working from the rubric, and 77 were touched by a person. On those
  77 the current detector reads P 0.36 / R 0.69 under B against 0.40 / 0.67 on
  the full set — consistent, but n=13 positives is far too small to call it
  validated. The calibration is against the rubric as an LLM applies it.
- **Rater noise caps everything.** On the 54 double-rated turns the two raters
  agreed exactly 37 times (68.5%), differed by 1 fourteen times and by 2 three
  times. A detector cannot be measured cleaner than the labels.
- **No held-out corpus.** Cross-validation was over random folds of one label
  set drawn from one run on state-assessment items. The recommendation to drop
  `overlap` and `elimination` rests on a mechanism that generalises — neither
  handles numbers — but the exact F1 figures do not transfer to another corpus.
- **`identifying_hits ≥ 1` vs `≥ 2` is unresolved.** They optimise different
  metrics and the choice depends on the cost ratio, which is not in the data.

## Readings that are not signal

- *"Sweeping the thresholds fixes the detector."* It does not. Two of the four
  hand-picked cutoffs are already F1-optimal and a third is one grid step off.
  The gain comes from deleting two signals.
- *"F1 0.645 under treatment A is the best rule found."* Flagging every turn
  scores 0.743 under treatment A. That row means nothing.
- *"Recall 0.665 is decent."* On the give-away class, yes; but 46 of the 77
  misses read exactly zero on all four signals, so no threshold on this feature
  set reaches them.
- *"Precision 0.397 means the detector is 40% right."* It means 40% of flags are
  rated 3; 74% are rated 2 or 3. Whether a rating-2 flag is an error depends on
  what the `-1` is for, and that is a design question, not a measurement.
