# review_app — human labels for tutor dialogues

A static site (no backend, no build step) where reviewers label real dialogues from
training run v0. Deployed to GitHub Pages at
**https://edu-llm.github.io/tutor-review/**

## What it collects and why

| output | answers the open question |
|---|---|
| `leak` ∈ `no` / `hints_at_it` / `names_it` | **`LeakGuard` has never been checked against a human.** These labels give it a precision and a recall. |
| `student_reply` (free text, gold answer hidden) | **Persona seeds.** 91 hand-written turns exist; prompting the student into a child's voice provably failed, so more voices from more people is the only lever. |
| `helpful` ∈ `helps` / `too_vague` / `just_tells` | Run v0 measured **specificity ≈ 0** — a hint written for a *different* question helped as much as the right one. This asks humans whether the hints are useful at all. |
| the 5-item shared set | **Inter-rater agreement.** If humans disagree with each other about leakage, no rule or judge can be expected to match them. |

The gold answer is hidden during stage 1 and revealed only in stage 2. That ordering
is the point: a reviewer who already knows the answer cannot write an uncontaminated
student reply.

## Files

```
index.html   markup
styles.css   dark theme, WCAG AA contrast, 48px tap targets
app.js       assignment, staging, localStorage, export.  SUBMIT_URL lives at the top.
data/items.json         built artefact the browser fetches
data/analysis_key.json  LeakGuard's verdict per item - NOT fetched by the app
```

## Rebuilding the item set

```bash
python src/build_review_set.py        # writes both files under review_app/data/
```

Sampling is stratified on LeakGuard's own decision (flagged verbatim / by word
overlap / by elimination, plus near-threshold "borderline" and clearly-clean
dialogues) so both false positives and false negatives are measurable. Population
counts for reweighting are in `analysis_key.json`.

**Licensing:** OpenBookQA only. The build refuses to run if any state-assessment
string (STAAR, PSSA, MCAS, CAASPP, NJSLA) appears in the output — that content is
state-copyright and must never be republished.

## The queue — no quotas

Nobody is assigned a number of items. A quota makes a busy person do nothing rather
than something, and it makes stopping early feel like failing.

1. The **5 shared items come first**, in the same order for everybody. Somebody who
   does five and stops has still produced a complete agreement datapoint — which is
   the measurement that cannot be recovered any other way.
2. After that the **whole remaining pool**, shuffled by a PRNG seeded on the
   reviewer's normalised name, so ten people start at ten different points and
   coverage spreads on its own. They keep going until they choose to stop.

The UI never shows a denominator: the top bar reads "12 reviewed — thank you!" and
a "Done for now — download my reviews" button is visible from the first item.
Everything is written to `localStorage` after each item, so the same name resumes
exactly where it left off — on the same device, days later.

Shared items are not visibly marked; flagging them would change how carefully they
are judged, which is precisely what the agreement measurement must not do.

## Collecting results

There is no server. Answers live in `localStorage` and reviewers download a JSON
file. To add a backend later, set `SUBMIT_URL` at the top of `app.js` to a
CORS-enabled endpoint that accepts `POST` — the app then posts exactly the payload
the download produces. Nothing else changes.

## Deploying

`review_app/` **is** the site root; there is no build step.

```bash
python src/build_review_set.py
bash review_app/deploy.sh        # pushes to edu-llm/tutor-review and enables Pages
```
