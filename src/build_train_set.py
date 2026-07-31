"""Combine state-assessment items into a training set.

State items carry no oracle hint, so zpd_filter's screen ("fails alone but
solves WITH the hint") cannot run on them. That screen was also found to enrich
leaky hints 2.8x, because "solves with help" has a degenerate solution, so
losing it is not purely a cost.

What we lose is the guarantee that the student cannot already answer. Problems
it solves unaided produce identical rewards across the whole group, contribute
no gradient, and show up as zero_adv_frac. Watch that metric; if it climbs, add
a baseline screen (one GPU pass) rather than more steps.
"""

from __future__ import annotations

import argparse
import json
import random

import benchmarks
import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", default=["ca", "tx"])
    ap.add_argument("--out", default=str(paths.DATA / "state_tests" / "train_items.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-grade", type=int, default=3)
    ap.add_argument("--max-grade", type=int, default=8,
                    help="a 0.5B student is at chance on high-school items, which is "
                         "noise rather than difficulty")
    args = ap.parse_args()

    rows = []
    for s in args.states:
        got = benchmarks.load_benchmark(s)
        for r in got:
            r.setdefault("hint", None)
        in_band = [r for r in got if isinstance(r.get("grade"), int)
                   and args.min_grade <= r["grade"] <= args.max_grade]
        rows.extend(in_band)
        print(f"  {s}: {len(in_band)} kept of {len(got)} "
              f"(grades {args.min_grade}-{args.max_grade})")

    # interleave states so a resumed or truncated run still sees both
    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    import collections
    print(f"\nwrote {len(rows)} -> {args.out}")
    print("subject:", dict(collections.Counter(r.get("subject") for r in rows)))
    print("grade  :", dict(sorted(collections.Counter(r.get("grade") for r in rows).items(),
                                  key=lambda kv: str(kv[0]))))
    w = [len(r["choices"][r["gold_idx"]].split()) for r in rows]
    import statistics as st
    print(f"gold answer: median {st.median(w):.0f} words, "
          f"{sum(1 for x in w if x == 1) / len(w):.0%} single-word "
          f"(OpenBookQA: 2 words, 31%)")


if __name__ == "__main__":
    main()
