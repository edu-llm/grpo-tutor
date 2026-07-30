# Measuring answer leakage without fooling ourselves

Notes for `hint_only_leak`. Short version: **our current number is uninterpretable
as it stands, and the objection that "maybe the student just barely knows it" is
the exact failure mode the literature warns about.**

---

## What we currently measure

- `hint_only_leak` = student sees **hint + choices**, not the question. Counts as
  "leaked" if it picks gold.
- First GPU eval on QASC: **0.325-0.40**, vs `teaching_gain` 0.25-0.275.
- I read that as "leakage could explain the whole gain". That reading is not
  supported yet.

## Why it's not trustworthy yet

- This is a **partial-input baseline** (hypothesis-only / choices-only family).
  Known to be a blunt instrument.
- **We never measured the floor.** No control for what the student scores on
  *choices alone*, with no hint at all. Without it, 0.40 has no reference point.
- Chance on QASC is 0.125 (8-way) - but chance is the wrong floor. The right
  floor is choices-only, which is typically **far above chance**.
- Balepur et al. (ACL 2024): LLMs beat the majority baseline on choices-only
  prompts in **11/12** dataset-model pairs. Distractors leak via style, length,
  specificity, topical coherence.
- So a chunk of our 0.40 is probably the student exploiting *the choices*, not
  the hint. That share is attributable to the teacher only by mistake.
- **Sophia's objection, formalized:** a legitimate conceptual hint given to a
  student with partial knowledge can tip it over without containing the answer.
  The probe scores that as leakage. Teaching and leaking are not separable by
  "did the student get it right without the question".
- Asymmetry to remember (Feng et al., ACL 2019): a partial-input baseline
  scoring **high** proves the task is cheatable; scoring **low** does *not* prove
  it is clean. Our metric is only valid evidence in one direction.

## The controls to add

Run the same student over matched conditions on the same items. Each isolates
one confound.

| condition | student sees | isolates |
|---|---|---|
| `choices_only` | choices | **the missing floor** - artifact exploitation |
| `random_hint` | unrelated hint + choices | does *any* plausible text help |
| `swapped_hint` | another item's teacher hint + choices | teacher style vs item content |
| `oracle_hint` | dataset gold fact + choices | what a legitimate hint scores |
| `teacher_hint` | our teacher's hint + choices | what we report today |
| `full` | question + choices (+hint) | the real task |

- Report **`teacher_hint − choices_only`**, never raw `teacher_hint`.
- Calibrate against `oracle_hint`: QASC's `combinedfact` is a *legitimate*
  teaching fact. If the teacher scores at or below it, the teacher is not
  leaking more than a good hint inherently does.
- If `teacher_hint ≈ choices_only`, the hint adds nothing and our leak signal is
  an artifact of the choices.

## Stronger test: wrong hints, not missing hints

- From TRACE (arXiv:2510.01367): for in-context hint loopholes, test whether the
  model **fails when given the WRONG hint** - explicitly "a stricter test than
  simply removing the hint".
- Removing the question asks "could the student have guessed?". Injecting a
  *misleading* hint asks "is the student actually following the hint?" - which is
  the thing we care about.
- Design (RWRR-Bench): four matched variants per item - **clean /
  helpful-hint / misleading-hint / counterfactual** - and report a *shortcut
  reliance gap* rather than one number.
- Concretely for us: hand the student a hint pointing at a **distractor**. If it
  follows into the wrong answer, the hint is doing the work (leak-like). If it
  resists, the student had real knowledge and the hint was a nudge.

## Specificity (separate problem, same machinery)

- `swapped_hint` doubles as the generic-filler test we already owe: reward
  should be `solved(own hint) − solved(swapped hint)`.
- Earlier finding: ~70% of the gain survived swapping. That's the same signal,
  measured once and never wired into the reward.

## Tutoring-specific practice

- **MRBench / unified AI-tutor taxonomy** (NAACL 2025) scores 8 pedagogical
  dimensions, two of which are exactly ours: *Revealing of the Answer* and
  *Providing Guidance*. Worth adopting the label definitions rather than
  inventing our own.
- **Adversarial-student leakage benchmark** (ACL 2026, Zhao et al.): measures
  **leakage rate** *and* **turns-until-leakage**, with a student agent fine-tuned
  to jailbreak the tutor. Directly relevant once multi-turn is on: a tutor that
  holds out for 10 turns differs from one that folds in 2, and our current
  single number can't see that.
- Their defenses that transfer cheaply: explicit pedagogical system prompt,
  training on educational dialogue, refusing on direct answer requests.

## Recommended metric set

- `leak_rate` (rules) - cheap, high precision, low recall. Keep.
- `hint_only_leak − choices_only` - the corrected probe. Replace the raw one.
- `misleading_hint_follow_rate` - the strict causal test. Add.
- `solved(own) − solved(swapped)` - specificity. Add, then put in the reward.
- `turns_until_leak` - once multi-turn runs on GPU.

## Decision rule

- Only claim leakage when `teacher_hint` clears **both** `choices_only` **and**
  `oracle_hint`.
- Report the floor next to the number, always.
- Treat a low probe as "not caught", never as "clean" (Feng et al.).

## References

- [Misleading Failures of Partial-input Baselines](https://aclanthology.org/P19-1554/) - Feng, Wallace, Boyd-Graber, ACL 2019
- [Artifacts or Abduction: How Do LLMs Answer MCQs Without the Question?](https://aclanthology.org/2024.acl-long.555/) - Balepur et al., ACL 2024
- [Is Your LLM Knowledgeable or a Choices-Only Cheater?](https://aclanthology.org/2024.knowllm-1.2) - Balepur & Rudinger, KnowLLM 2024
- [Is It Thinking or Cheating? (TRACE)](https://arxiv.org/pdf/2510.01367) - arXiv:2510.01367
- [RWRR-Bench: right-answer, wrong-reason](https://openreview.net/pdf?id=6696c7a045158fa9ccfea87bd90eaeaadc980e2d)
- [Evaluating Answer Leakage Robustness of LLM Tutors](https://aclanthology.org/2026.acl-long.1412/) - Zhao et al., ACL 2026
- [Unifying AI Tutor Evaluation / MRBench](https://aclanthology.org/2025.naacl-long.57.pdf) - NAACL 2025
