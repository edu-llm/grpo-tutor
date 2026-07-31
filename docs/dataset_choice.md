# Choosing a training corpus after run v0

Run v0 ended with specificity ~0: a hint written for a *different* problem helped
the student as much as the right one. The diagnosis on the table was structural —
"OpenBookQA answers are single words, so correcting the misconception and
revealing the answer are the same act" — and the implied fix was a new corpus.

This is the survey that was supposed to find one. It found something else first.

---

## Headline

**The corpus was not the main problem. The ZPD screen was.**

`zpd_filter.py` keeps items the student fails alone but solves *with the oracle
hint*. That criterion has an easy degenerate solution — a hint that contains the
answer — so the screen preferentially keeps exactly the items whose ceiling an
honest tutor can never reach. Measured with `rewards.leak_signals` on the live
549-item set against the OpenBookQA pool it was drawn from:

| | hints tripping the leak rule | gold answers that are one word |
|---|---|---|
| `obqa_train` pool (4,957) | **10.1%** | 31% |
| the curated 549 (`data/zpd_problems.jsonl`) | **28.2%** | 47% |

A 2.8x enrichment in leaky hints, and a 1.5x enrichment in exactly the one-word
answers run v0 blamed on the corpus. The screen manufactured a worse task than
the one it was handed.

**Two things also turned out to be false in the framing:**

1. OpenBookQA answers are mostly **not** single words — median 2 words, 31%
   single-word. **QASC's are**: median 1 word, **60%** single-word. On the
   metric the diagnosis names, the corpus we moved *to* is the worse one.
2. SciQ's `support` passage leaks the gold answer in **96.4%** of items, so its
   quoted +0.283 oracle headroom is contaminated in the same way QASC's +0.64
   was. That was not previously known.

**Recommendation:** primary `openbookqa_honest` (OpenBookQA `fact1`,
leak-screened, 4,454 items) — now the `zpd_filter.py` default. Fallback
`race_middle` (24,587 items, 4-word median answers). Neither is a new corpus in
the exciting sense; the evidence says the win available here is in the screen and
the hint field, not in a corpus nobody has tried.

---

## What the oracle hint audit found

Every number below is `rewards.leak_signals` over the **full** corpus — pure
string matching, no model, seconds per thousand items. Reproduce with:

```bash
python src/hint_audit.py --candidates
```

`trips` is the fraction of oracle hints the training leak rule fires on. It is
the number that killed QASC, and it kills most of the field.

| corpus / hint field | n | chance | gold words | 1-word | verbatim | trips |
|---|---|---|---|---|---|---|
| **race_middle / masked locator** | 25,416 | 0.250 | 4 | 0.19 | 0.003 | **0.033** |
| race_high / masked locator | 8,000* | 0.250 | 6 | 0.09 | 0.001 | 0.014 |
| dream / masked locator | 6,115 | 0.333 | 4 | 0.13 | 0.003 | 0.054 |
| qasc / combinedfact, gold masked | 8,134 | 0.125 | 1 | 0.60 | 0.000 | 0.008 |
| **obqa_train / fact1** | 4,957 | 0.250 | 2 | 0.31 | 0.058 | **0.101** |
| obqa_test / fact1 | 500 | 0.250 | 3 | 0.31 | 0.096 | 0.150 |
| qasc_train / fact1 | 8,134 | 0.125 | 1 | 0.60 | 0.306 | 0.372 |
| scienceqa / lecture (text-only) | 6,274 | 0.443 | 3 | 0.28 | 0.239 | 0.426 |
| quartz / para | 2,696 | 0.500 | 1 | 0.77 | 0.222 | 0.482 |
| dream / whole dialogue | 6,116 | 0.333 | 4 | 0.13 | 0.166 | 0.556 |
| qasc_train / fact2 | 8,134 | 0.125 | 1 | 0.60 | 0.496 | 0.615 |
| race_middle / whole article | 25,421 | 0.250 | 4 | 0.19 | 0.271 | 0.779 |
| scienceqa / solution | 5,837 | — | — | — | 0.776 | 0.877 |
| ecqa / taskA_pos | 7,598 | 0.200 | 1 | 0.56 | 0.856 | 0.936 |
| sciq_train / support | 10,481 | 0.250 | 1 | 0.61 | 0.915 | 0.964 |
| **qasc_train / combinedfact** | 8,134 | 0.125 | 1 | 0.60 | 0.885 | **0.967** |
| ecqa / taskB | 7,598 | 0.200 | 1 | 0.56 | 0.862 | 0.980 |
| *live* `zpd_problems.jsonl` | 549 | 0.250 | 2 | **0.47** | 0.189 | **0.282** |

\* the first 8,000 train rows, not the full RACE-high split; every other row is
the complete corpus.

Corpora with **no oracle hint field at all**, so they cannot be ZPD-screened and
have no ceiling to quote (shape measured anyway, since it is what makes a corpus
tutorable):

| corpus | train pool | chance | gold words | 1-word |
|---|---|---|---|---|
| arc_challenge | 1,119 | 0.251 | 5 | 0.15 |
| arc_easy | 2,251 | 0.250 | 3 | 0.29 |
| commonsense_qa | 9,741 | 0.200 | 1 | 0.54 |
| winogrande | 40,398 | 0.500 | — | — |
| mmlu us_history | 204 | 0.250 | 7 | 0.04 |

### Why each candidate was rejected

- **SciQ** — `support` leaks in 96.4% of items (91.5% verbatim). The +0.283
  headroom in the README is the student copying out of the passage. A new result;
  SciQ should be dropped as an oracle-hint reference wherever it is quoted.
- **ECQA** (CommonsenseQA with human explanations) — the explanation exists to
  justify the gold option, so it names it: 93.6% trips for `taskA_pos`, 98.0%
  for the free-flow `taskB`. The best-looking hint field on the hub is
  structurally the worst. CoS-E is the same annotation idea on the same
  questions and was not measured separately.
- **ScienceQA** — the most promising design on paper: `lecture` is *general
  background shared across every question of a skill*, and `solution` is the
  answer-specific step, so the two are cleanly separated. It fails on shape, not
  on leakage. Of 6,274 text-only items only **598 are 4-way** (4,439 are 2-way,
  chance 0.5), and after leak-screening only **303** 4-way items remain — an
  order of magnitude short of the ~2,000 needed. Much of its `lecture` leakage is
  also an artifact: a lecture that defines solid/liquid/gas names all three
  options and trips the *elimination* rule without telling you anything.
- **QuaRTz** — 2-way (chance 0.50) and 77% single-word answers.
- **LogiQA** — formal logic, not grade 3-9, and the English release is a
  translation. Wrong difficulty band for a 0.5B.
- **WinoGrande / PIQA-shaped 2-way sets** — chance 0.50 puts half the reward
  signal in the coin flip.
- **ARC-Challenge / ARC-Easy / CommonsenseQA** — no hint field, so there is
  nothing to screen the ZPD band with and no ceiling to compare a tutor against.
  ARC-Challenge has the best answer shape of any science set (median 5 words,
  15% single-word) and would be the corpus to revisit *if* someone derived a
  hint for it, e.g. by retrieving the nearest fact from OpenBookQA's 1,326-fact
  open book. Not attempted here.
- **QASC** — see below. Kept and registered, but not recommended.
- **DREAM** — a viable smaller sibling of RACE (6,115 items, masked locator
  trips 5.4%), but 3-way and 4x smaller. No reason to prefer it over RACE.

### The QASC trap, in full

QASC is the current default and the audit is unkind to it in three separate ways.

1. `combinedfact` states the gold option verbatim in 88.5% of items. Its
   0.253 → 0.893 ceiling is copying.
2. Every honest repair shrinks the ceiling roughly in proportion to how much
   leakage it removes (measured — see the next section).
3. Its answers are the shortest in the survey: **60% single-word**, against
   OpenBookQA's 31%. If the run v0 structural diagnosis is right at all, it
   indicts QASC harder than the corpus it replaced.

The 8-way option set — QASC's real advantage, chance 0.125 — is also weaker than
it looks. The distractors are recycled from other questions, so a single item can
carry `'h2o'` and `'H20'` as two separate options, and sets like
`['Necklaces.', 'Steam.', 'Glass beads .', 'a wave', 'tiny', 'a solute', 'rain',
'Bracelets.']` are not eight plausible answers.

`qasc_train_honest` (fact1, leak-screened, 5,112 items) is registered because it
has the largest *measured* honest headroom of anything here, and it is the right
fallback if the problem turns out to be headroom rather than structure. It is not
the recommendation because it makes the diagnosed problem worse.

---

## What was measured on the real student

**Provenance, stated plainly.** These five rows were produced by running
Qwen2.5-0.5B-Instruct locally on MPS (fp32, `HFStudent.choose`, 150 items sampled
per row, seed 0) before the instruction to stop local model inference arrived.
They are real measurements on the real reward channel, not estimates. Nothing was
run after that point, and the OpenBookQA and RACE equivalents were **not**
reached — they are in the "needs GPU" section below.

Four matched conditions per item, the control set `docs/eval_leakage.md` asks for:

| pool | n | baseline | +hint | gain | hint-only | choices-only | above floor | ZPD keep |
|---|---|---|---|---|---|---|---|---|
| qasc / combinedfact | 150 | 0.160 | 0.893 | **+0.733** | 0.753 | 0.107 | **+0.647** | 0.733 |
| qasc / fact1, all | 150 | 0.160 | 0.487 | +0.327 | 0.360 | 0.107 | +0.253 | 0.333 |
| qasc / fact1, leak-screened | 150 | 0.180 | 0.387 | +0.207 | 0.200 | 0.053 | +0.147 | 0.227 |
| qasc / fact1-or-fact2, screened | 150 | 0.193 | 0.353 | +0.160 | 0.233 | 0.093 | +0.140 | 0.187 |
| qasc / combinedfact, gold masked | 150 | 0.160 | 0.293 | +0.133 | 0.233 | 0.107 | +0.127 | 0.173 |

Read the last two columns together. "Above floor" is how much of the hint's value
survives **hiding the question** — i.e. how much of it is answer-pointing rather
than teaching:

- `combinedfact`: 0.647 of a 0.733 gain, **88%**. Confirmed: the hint is the answer.
- `fact1` leak-screened: 0.147 of a 0.207 gain, **71%**. The best ratio measured,
  and still most of the hint's value is available without the question.
- Masking the gold out of `combinedfact` gets the string leak rate to 0.008 but
  leaves a cloze whose value is **95%** answer-pointing. **String-honest is not
  semantically honest**, and this is the concrete counterexample. Any future
  "mask the answer out of the supporting fact" idea has to clear this bar, not
  the leak rule.

`choices_only` came out at 0.053–0.107 against a 0.125 chance level, so on QASC
the choices-only artifact the leakage doc worries about is small — the floor is
at or below chance. That is a genuinely reassuring control and it is new.

---

## Recommendation

### Primary: `openbookqa_honest` — OpenBookQA `fact1`, leak-screened, 4,454 items

```bash
python src/zpd_filter.py --limit 5000 --out data/zpd_obqa_honest.jsonl
```

Now the `zpd_filter.py` default. The evidence:

- **The cleanest oracle hint of any corpus surveyed**: 10.1% trips, 5.8%
  verbatim, mean gold-overlap 0.148. Nothing else with a human-written hint field
  is within 3x of it.
- **The hint states a principle the student must apply, and the answer is the
  application.** This is requirement 1 met by a hint field that already exists:

  > *Q:* "When standing miles away from Mount Rushmore"
  > *hint:* "as distance to an object increases, that object will appear smaller"
  > *gold:* "the mountains seem smaller than in photographs"

  A tutor can say the whole principle out loud and still not have named the
  option. That move does not exist on QASC, where gold is `'h2o'`.
- **Answer shape is second-best among sets that have a hint at all** — median 2
  words, 31% single-word, versus QASC's 1 and 60%.
- **OpenBookQA has never actually been tried honestly.** Run v0 trained on a set
  whose hints leak at 28.2%. The negative result is evidence about that set, not
  about this pool.

The known risk is volume: 4,454 items at the ~11% keep rate leaves ~490 problems,
and screening out the leaky items removes disproportionately many of the ones
that pass "solves with help", so the real number will be lower. If it lands under
~300, take the fallback.

### Fallback: `race_middle` — RACE-middle with a derived locator hint, 24,587 items

```bash
python src/zpd_filter.py --limit 25000 --source race_middle --out data/zpd_race.jsonl
```

The structural upgrade, if OpenBookQA's honest headroom is too thin:

- **Best answer shape in the survey**: median 4 words, only 19% single-word.
  Naming the concept and naming the option stay distinct acts by construction.
- **~10x the volume** of any science pool, so a strict screen still leaves
  thousands of items.
- **The hint is honest twice over.** `benchmarks._locator` picks the article
  sentence with the highest content-word overlap with the *question* — the
  options play no part in selection — and then masks any gold content word out of
  it. 3.3% trips, against 77.9% for handing over the whole article.
- **It is the one corpus where scaffolding is definitionally not telling.** The
  student already has the passage, so "reread the line where the writer describes
  the weather" transfers no information it did not already hold; it directs
  attention. That is the cleanest possible separation of teaching from leaking,
  and it is the failure mode run v0 could not move.
- Grade 6-9 English exam reading, so it is also not maths.

Its risks are real and unmeasured: 198-word median prompts through a bare
`Fact: …\nQuestion: …\nAnswer:` completion with no chat template, and a locator
that is only as good as question/sentence lexical overlap — spot-checking found
it picking the wrong sentence when the question is short ("How old is she?").

### Not recommended, but registered

`qasc_train_honest` (5,112) keeps the largest measured honest headroom (+0.207)
available if the blocker turns out to be headroom rather than structure.

---

## Needs GPU

Nothing below was run. Each is a couple of minutes on the idle H100. All of them
write to an explicit `--out`, so **none of them overwrites
`data/zpd_problems.jsonl`**, which is still the set run v0 trained on.

### 1. The decisive one — headroom and reachability for every new pool

`--probe` adds `hint_only` and its `choices_only` floor, so the answer is not
just "is there headroom" but "is the headroom reachable by an honest tutor".

```bash
srun --partition=mit_preemptable --gres=gpu:h100:1 --mem=64G --time=0:30:00 \
  --pty bash -lc 'source scripts/env.sh && python -u src/bench_baseline.py \
    --n 200 --probe --out runs/bench_honest.json \
    --names obqa_train_honest obqa_honest obqa_train qasc_train_honest \
            race_middle_train race_middle'
```

Decision rule, applied to the `obqa_train_honest` row:

- `oracle_headroom` >= ~0.10 **and** `leak_above_floor` materially below the
  headroom → OpenBookQA confirmed, rebuild and train.
- headroom near zero → the 0.5B cannot use an honest OpenBookQA hint at all;
  switch to `race_middle`.
- `leak_above_floor` ≈ `oracle_headroom` → even the screened hint only works by
  pointing at the answer, and the leak *rule* is too weak. That would be an
  argument for the semantic screen in "still unmeasured" below, not for another
  corpus.

### 2. Rebuild the curated set on the winner

```bash
sbatch --wrap='source scripts/env.sh && python -u src/zpd_filter.py \
  --limit 5000 --source openbookqa_honest --out data/zpd_obqa_honest.jsonl' \
  --partition=mit_preemptable --gres=gpu:h100:1 --mem=64G --time=1:00:00 \
  --output=logs/zpd_%j.out
```

Then re-audit the *output*, which is the check that would have caught this whole
problem in the first place, and costs no GPU:

```bash
python src/hint_audit.py --name obqa_train_honest    # pool, for comparison
python -c "import json,sys; sys.path.insert(0,'src'); import hint_audit; \
  print(hint_audit.audit('curated', [json.loads(l) for l in open('data/zpd_obqa_honest.jsonl')]))"
```

`trips_leak_rule` on the curated output must be 0.000. If a future change makes
it non-zero, the screen has started selecting for leakage again.

### 3. The fallback, if step 1 says OpenBookQA is too thin

```bash
sbatch --wrap='source scripts/env.sh && python -u src/zpd_filter.py \
  --limit 25000 --source race_middle --out data/zpd_race.jsonl' \
  --partition=mit_preemptable --gres=gpu:h100:1 --mem=64G --time=2:00:00 \
  --output=logs/zpd_race_%j.out
```

RACE prompts are ~10x longer than OpenBookQA's, so budget accordingly and check
the baseline row from step 1 first — if `race_middle` baseline is at chance
(0.250), the 0.5B cannot read the passage through `choose()` and the corpus is
unusable regardless of how honest its hints are.

---

## Still unmeasured

- **Honest headroom for OpenBookQA and RACE.** The single most important number
  in this document is the one the local run did not reach. Everything in the
  recommendation rests on structure and leak rates; step 1 above settles it.
- **Whether an honest OpenBookQA hint leaves enough items.** The ~490 estimate
  assumes the old 11% keep rate still holds after leak-screening. It will not
  hold exactly, and it could be much worse, because leaky hints were
  overrepresented among the items that pass "solves with help".
- **A semantic honesty screen.** The masked-cloze result proves the string rule
  under-detects: 0.008 trips, 95% of the hint's value answer-pointing. Screening
  the pool by `hint_only_leak − choices_only` per item instead of by string rules
  would be strictly better and needs one student forward pass per item — cheap on
  a GPU, and it composes with any corpus choice.
- **Whether fixing the screen alone fixes specificity.** The honest reading of
  this survey is that run v0 never tested its own hypothesis: it trained on a set
  28.2% of which had the answer in the hint. Rerunning v0 unchanged on
  `zpd_obqa_honest.jsonl` is a cheaper and better-controlled experiment than any
  corpus swap, and should probably come first.
- **RACE locator quality.** 3.3% trips says it is honest; nothing says it is
  *useful*. A sample of 50 hand-read locator hints would cost nothing and would
  catch the "wrong sentence for short questions" failure before a GPU run does.
- **ARC-Challenge with retrieved OpenBookQA facts.** The best answer shape in
  science (median 5 words, 15% single-word) and 1,119 + 2,251 items, blocked only
  by the absence of a hint field. Deriving one by lexical retrieval against the
  1,326-fact open book is a few hours and needs no GPU.
