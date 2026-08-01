"""Keep only problems the student cannot already answer.

State-assessment items have no oracle hint, so zpd_filter's usual screen
("fails alone but solves WITH the hint") cannot run on them. v1 was launched
without any screen and step 0 said what that costs:

    solved=0.97   zero_adv=0.75

The student answered 97% of the training problems, and three quarters of the
groups produced identical rewards across all K completions - which is zero
gradient, because advantages are mean-centred within the group. Most of the
compute was going to problems with nothing to teach.

This applies the half of the screen that does not need a hint: drop anything the
student answers unaided. It also screens the FREE-TEXT channel, because 26% of
the old single-channel set turned out to be answerable when simply asked, and
optionally drops grades outside a band (a 0.5B on grade-11 content is at chance,
which is noise rather than difficulty).

Both channels are multiple-choice, so both floor at 25% on a 4-way item and a
single correct answer is weak evidence of knowledge. Measured on the 687-item
CA+TX+MA+NJ pool, the union of the two called 64.2% "known", and did so as often
on grade 11 as on grade 3 - a profile that knowledge does not have. So neither
channel is trusted on one observation:

  free text   sampled at temperature 0.8, so --free-trials repeats disagree.
              An item counts as known only if EVERY trial names gold.
  choose()    deterministic log-prob argmax; repeating it returns the same
              answer, and the options never enter the prompt, so permuting them
              changes nothing either. Its reliability signal is the MARGIN -
              gold's length-normalised log-prob minus the best distractor's. A
              lucky argmax wins by a hair, knowledge wins by a gap. --margin sets
              the gap required before the item counts as known.

Every item's raw measurements are written to --report regardless of the rule, so
a different threshold can be re-derived offline without touching a GPU:

    python src/baseline_screen.py --replay data/state_tests/screen_report.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json

import torch

import benchmarks
import paths
from zpd_filter import HFStudent, _names_gold, FREE_TEXT_ASK, _format_choices


def knows(rec: dict, margin: float, free_trials: int) -> bool:
    """Is the student's success on this item reliable enough to call it known?

    Both conditions are deliberately hard to satisfy by luck: the choose()
    channel must win by `margin`, and every free-text trial must land.
    """
    by_choice = rec["choose_correct"] and rec["margin"] >= margin
    by_free = free_trials > 0 and rec["free_hits"] >= free_trials
    return bool(by_choice or by_free)


def report_summary(recs, margin: float, free_trials: int):
    kept = [r for r in recs if not knows(r, margin, free_trials)]
    n = max(1, len(recs))
    print(f"choose() correct : {sum(r['choose_correct'] for r in recs)}/{n} "
          f"({sum(r['choose_correct'] for r in recs) / n:.1%}) "
          f"- of those, {sum(r['choose_correct'] and r['margin'] >= margin for r in recs)} "
          f"clear the margin {margin:g}")
    if free_trials:
        print(f"free-text {free_trials}/{free_trials}   : "
              f"{sum(r['free_hits'] >= free_trials for r in recs)}/{n} "
              f"(any trial: {sum(r['free_hits'] > 0 for r in recs)})")
    print(f"KEPT             : {len(kept)} ({len(kept) / n:.1%})")
    return kept


def margin_table(recs, free_trials: int):
    """Keep count against margin, so the threshold is chosen from data."""
    print("\n  margin   kept   (choose() drops surviving the gap)")
    for m in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        k = sum(1 for r in recs if not knows(r, m, free_trials))
        drops = sum(1 for r in recs if r["choose_correct"] and r["margin"] >= m)
        print(f"  {m:5.2f}   {k:4d}   {drops}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", default=["ca", "tx"])
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-grade", type=int, default=8,
                    help="drop items above this grade; a 0.5B is at chance on high school")
    ap.add_argument("--min-grade", type=int, default=3)
    ap.add_argument("--free-trials", type=int, default=3,
                    help="sampled free-text attempts; the item counts as known "
                         "only if ALL of them name gold (0 disables the channel)")
    ap.add_argument("--seed", type=int, default=0,
                    help="trial t is seeded with seed+t: the trials stay different "
                         "from each other, and the screen repeats exactly")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="log-prob gap gold must beat the runner-up by before "
                         "choose() counts as knowing it (0 = plain argmax)")
    ap.add_argument("--replay", default=None,
                    help="re-derive the kept set from a saved report, no GPU")
    ap.add_argument("--report", default=str(paths.DATA / "state_tests" / "screen_report.jsonl"))
    ap.add_argument("--out", default=str(paths.DATA / "state_tests" / "train_items.jsonl"))
    args = ap.parse_args()

    if args.replay:
        recs = [json.loads(l) for l in open(args.replay)]
        print(f"replaying {len(recs)} measured items from {args.replay}\n")
        margin_table(recs, args.free_trials)
        print()
        kept = report_summary(recs, args.margin, args.free_trials)
        write_kept(kept, args.out)
        return

    rows = []
    for s in args.states:
        got = benchmarks.load_benchmark(s)
        rows.extend(got)
        print(f"  loaded {s}: {len(got)}", flush=True)

    in_band = [r for r in rows
               if isinstance(r.get("grade"), int)
               and args.min_grade <= r["grade"] <= args.max_grade]
    print(f"\ngrade {args.min_grade}-{args.max_grade}: {len(in_band)}/{len(rows)} kept",
          flush=True)

    student = HFStudent(args.student)

    # free-text pass first: batched generation is far cheaper than per-item scoring
    free_hits = [0] * len(in_band)
    for t in range(args.free_trials):
        torch.manual_seed(args.seed + t)
        views = [FREE_TEXT_ASK.format(q=r["question"], choices=_format_choices(r["choices"]))
                 for r in in_band]
        replies = []
        for i in range(0, len(views), 32):
            replies.extend(student.reply(views[i:i + 32], max_new_tokens=60))
        for i, (r, rep) in enumerate(zip(in_band, replies)):
            gold = r["choices"][r["gold_idx"]]
            distr = [c for j, c in enumerate(r["choices"]) if j != r["gold_idx"]]
            free_hits[i] += int(_names_gold(rep, gold, distr))
        print(f"  free-text trial {t + 1}/{args.free_trials} (seed {args.seed + t}): "
              f"{sum(1 for h in free_hits if h > t)} still perfect", flush=True)

    recs = []
    for r, hits in zip(in_band, free_hits):
        scores = student.score_choices(r["question"], r["choices"])
        gold_idx = r["gold_idx"]
        best_other = max(s for j, s in enumerate(scores) if j != gold_idx)
        recs.append({**r,
                     "choose_correct": bool(max(range(len(scores)),
                                                key=lambda i: scores[i]) == gold_idx),
                     "margin": scores[gold_idx] - best_other,
                     "free_hits": hits,
                     "free_trials": args.free_trials})

    with open(args.report, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote per-item measurements -> {args.report}")

    margin_table(recs, args.free_trials)
    print()
    kept = report_summary(recs, args.margin, args.free_trials)
    write_kept(kept, args.out)


def write_kept(kept, out):
    with open(out, "w") as f:
        for r in kept:
            f.write(json.dumps({**r, "baseline_correct": False}) + "\n")
    print(f"\nwrote {out}")
    print("subject:", dict(collections.Counter(r.get("subject") for r in kept)))
    print("grade  :", dict(sorted(collections.Counter(r.get("grade") for r in kept).items())))
    if len(kept) < 200:
        print(f"\n[WARNING] only {len(kept)} problems - below the ~200 floor where the "
              f"teacher starts memorising per-problem hints. Add another state.")


if __name__ == "__main__":
    main()
