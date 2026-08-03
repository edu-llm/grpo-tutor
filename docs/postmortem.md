# Postmortem: why the frozen-student setup was scrapped

Four training runs (v0–v4) produced a flat `teacher_acc`. `why_flat.md` traced
that to the reward channel: rated goodness correlated −0.012 with the student
solving, while leakage correlated +0.291. This document records the two
experiments run afterwards, which answered a question the training runs could
not — whether there was anything in the environment for a reward to track.

The answer is no, and the reason is sharper than "no signal". **The tutoring
dialogues make the student worse.**

## What was measured

Both experiments score a student's `P(gold)` — the softmax over the offered
options, restricted to those options — rather than binary correctness. The
binary indicator only moves when a hint flips the argmax, so it reads an effect
that shifts belief without crossing the boundary as exactly zero.

Neither experiment trains anything. Both re-use the 1,232 dialogues in
`runs/gen_traces.jsonl` and cost about a GPU-minute.

### 1. Transfer probe — does teaching about A help on B?

125 pairs sharing a knowledge unit, matched by embedding similarity ≥ 0.82,
answered by the frozen 0.5B.

| condition | P(gold) |
| --- | --- |
| baseline (B alone) | 0.259 |
| treated (real dialogue about A) | 0.263 |
| swapped (dialogue from a *different* unit) | 0.263 |

Gain +0.0039 ± 0.0049. Specificity, which is `treated − swapped` and the only
term that required A and B to share anything, **+0.0001 ± 0.0050**. Zero to
three decimal places.

The positive control is what makes that null mean something. Told the answer
outright, the same student moves 0.259 → 0.517, about 13 SE. The instrument
was awake; there was nothing to detect.

### 2. Hint ladder — does teaching about A help on **A**?

The transfer probe asked a harder question than training does. Training only
asks that a dialogue about A help on A. Five rungs, ordered by how much of the
answer they contain, all on the same 272 items so every comparison is paired.

| rung | 0.5B | 3B |
| --- | --- | --- |
| nothing | 0.238 | 0.354 |
| the unit, named | **+0.0382** (+6.2 SE) | **+0.0404** (+5.4 SE) |
| the misconception, named | −0.0395 (−5.5 SE) | −0.0719 (−6.5 SE) |
| a real non-leaking dialogue | **−0.0216** (−3.6 SE) | **−0.0498** (−5.3 SE) |
| the answer, stated | +0.3627 (+28.2 SE) | +0.3974 (+26.6 SE) |

## What this says

**The channel is open.** A one-sentence topic pointer — "This problem is about
solving direct variation problems." — reliably helps, on both students. So the
setup can register a hint that is not the answer. That possibility was the thing
most in doubt, and it is not the problem.

**The dialogues are harmful, not merely useless.** A real, non-leaking dialogue
leaves the student *worse* than silence, on both models, at 3.6 and 5.3 SE.

**A better student makes it worse, not better.** The 3B has a higher floor
(0.354 vs 0.238) and a higher ceiling (0.752 vs 0.601), so it is the better
instrument by every measure — and every effect above is *larger* on it. This
rules out the most attractive explanation, that the 0.5B was simply too weak to
be taught.

**The likely mechanism is misconception rehearsal.** Naming a wrong idea hurts
by roughly the same magnitude as a whole dialogue does, and tutoring dialogues
rehearse wrong ideas by design. This also retro-explains `why_flat.md`: a
dialogue that *looks* pedagogical is one that engages the misconception, which
is exactly the kind that hurts — hence r ≈ 0 between rated goodness and solving.

**So the objective was sign-flipped, not weak.** A reward built on the student's
correctness pays for leaking (+0.291) and penalises tutoring (−0.02 to −0.05).
The leak-fix removed the only thing that paid, leaving a landscape where every
legitimate teaching action was neutral or negative. `teacher_acc` was flat
because there was nothing to climb, not because GRPO or the judge failed.

No reward model repairs this. It is a property of the environment.

## Caveats

- The misconception rung names the wrong idea **without correcting it**, so it
  shows that *mentioning* a wrong idea is costly, not that *addressing*
  misconceptions is. Separating those needs a sixth rung. The dialogue rung
  carries no such caveat — it is the teacher's real output, unedited.
- Everything here is four-way multiple choice, where the floor is a coin flip.
  Free response would have more resolution and no guessing floor.
- A frozen LLM has no weights to update, so "teaching" can only ever mean
  in-context conditioning. Whether that is a reasonable model of learning at all
  is the assumption underneath the whole design, and it was never tested.
- Simulator-to-human transfer was never validated, and the transfer probe is
  weak evidence against it.

## What would have to be different

Not a better judge, and not a bigger reward model. In rough order of how much
they are worth:

1. **A student that can be taught rather than only told.** Something with
   persistent state that updates, or at minimum an item set chosen inside the
   measured gap between "fails alone" and "succeeds when hinted".
2. **Free-response items**, so the outcome is not floored at 0.25 and the
   student cannot guess.
3. **A reward that is the measured gain**, not a judge's opinion of the
   dialogue. If a dialogue moves `P(gold)`, that number is verifiable and needs
   no rubric — which is what `open-instruct`'s RLVR path is built for.
4. **Positive and negative controls in the training loop**, not only in
   analysis. The topic-pointer rung is a natural upper baseline: a tutor that
   cannot beat one sentence is not teaching.

## Reproducing

The probe and ladder lived in `projects/tutor/` in the `edu-llm/open-instruct`
fork and were removed when this was scrapped; recover them from the history of
the `scored-rewards` branch. The generic reward layer they were built on,
`open_instruct/scored_rewards/`, is unaffected by any of this and remains in
that branch — nothing above reflects on it.

Raw outputs and job logs are in [`probe_results/`](probe_results/): the ladder as
`hint_ladder_*_19584426.{json,out}`, the probe as
`transfer_probe_1957{8035,8315}.json` with the second job's full log. They live
under `docs/` rather than `runs/` because `runs/` is gitignored, and a writeup
whose evidence is untracked is not checkable.
