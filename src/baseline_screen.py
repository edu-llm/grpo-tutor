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
"""

from __future__ import annotations

import argparse
import collections
import json

import benchmarks
import paths
from zpd_filter import HFStudent, _names_gold, FREE_TEXT_ASK, _format_choices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", default=["ca", "tx"])
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-grade", type=int, default=8,
                    help="drop items above this grade; a 0.5B is at chance on high school")
    ap.add_argument("--min-grade", type=int, default=3)
    ap.add_argument("--free-text-screen", action="store_true", default=True,
                    help="also drop items the student can answer in free text")
    ap.add_argument("--out", default=str(paths.DATA / "state_tests" / "train_items.jsonl"))
    args = ap.parse_args()

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
    knows_free = [False] * len(in_band)
    if args.free_text_screen:
        views = [FREE_TEXT_ASK.format(q=r["question"], choices=_format_choices(r["choices"]))
                 for r in in_band]
        replies = []
        for i in range(0, len(views), 32):
            replies.extend(student.reply(views[i:i + 32], max_new_tokens=60))
        for i, (r, rep) in enumerate(zip(in_band, replies)):
            gold = r["choices"][r["gold_idx"]]
            distr = [c for j, c in enumerate(r["choices"]) if j != r["gold_idx"]]
            knows_free[i] = _names_gold(rep, gold, distr)
        print(f"free-text correct: {sum(knows_free)}/{len(in_band)}", flush=True)

    kept, solved_alone = [], 0
    for r, kf in zip(in_band, knows_free):
        alone = student.choose(r["question"], r["choices"]) == r["gold_idx"]
        solved_alone += int(alone)
        if not alone and not kf:
            kept.append({**r, "baseline_correct": False, "free_text_correct": bool(kf)})

    n = max(1, len(in_band))
    print(f"choose() correct : {solved_alone}/{n} ({solved_alone / n:.1%})")
    print(f"KEPT             : {len(kept)} ({len(kept) / n:.1%})")

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {args.out}")
    print("subject:", dict(collections.Counter(r.get("subject") for r in kept)))
    print("grade  :", dict(sorted(collections.Counter(r.get("grade") for r in kept).items())))
    if len(kept) < 200:
        print(f"\n[WARNING] only {len(kept)} problems - below the ~200 floor where the "
              f"teacher starts memorising per-problem hints. Add another state.")


if __name__ == "__main__":
    main()
