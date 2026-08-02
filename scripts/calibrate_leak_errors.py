"""Print the detector's false positives and false negatives against the ratings.

FP: rule flags a leak, raters said 1 (no leak).
FN: rule says clean, raters said 3 (gives it away).
Run with PYTHONPATH=src:scripts.
"""

import json
import os
import random

from calibrate_leak import load, ROOT

CUR = {"verbatim": 1.0, "overlap": 0.6, "elimination": 0.5, "identifying_hits": 1.0}


def fired(r):
    return [s for s in CUR if r[s] >= CUR[s]]


def show(r, i):
    print(f"\n--- [{i}] {r['id']}  leak={r['leak']}  n_raters={r['n_raters']}  "
          f"subject={r['subject']} grade={r['grade']}")
    print(f"  Q:        {r['question'][:220]}")
    print(f"  gold:     {r['gold']!r}")
    print(f"  distr:    {r['distractors']}")
    print(f"  tutor:    {r['tutor_turn'][:500]}")
    print(f"  signals:  verbatim={r['verbatim']:.0f} overlap={r['overlap']:.3f} "
          f"elim={r['elimination']:.3f} ident={r['identifying_hits']:.0f}/{r['identifying_n']:.0f}"
          f"  fired={fired(r)}")


def main():
    rows = load()
    fp = [r for r in rows if r["rule_flagged"] and r["leak"] <= 1.0]
    fn = [r for r in rows if not r["rule_flagged"] and r["leak"] >= 3.0]
    print(f"total FP (flagged, rated 1): {len(fp)}")
    print(f"total FN (clean, rated 3):   {len(fn)}")

    import collections
    print("\nFP by which signals fired:",
          collections.Counter(tuple(sorted(fired(r))) for r in fp).most_common())
    print("FN identifying_n == 0 (no identifying words at all):",
          sum(1 for r in fn if r["identifying_n"] == 0 and r["identifying"] == 0), "of", len(fn))
    print("FN overlap distribution:", collections.Counter(
        round(r["overlap"], 1) for r in fn).most_common())
    print("FP subjects:", collections.Counter(r["subject"] for r in fp).most_common())
    print("FN subjects:", collections.Counter(r["subject"] for r in fn).most_common())

    rng = random.Random(0)
    print("\n" + "=" * 72 + "\n8 FALSE POSITIVES (rule=leak, raters=1)\n" + "=" * 72)
    for i, r in enumerate(rng.sample(fp, 8), 1):
        show(r, i)
    print("\n" + "=" * 72 + "\n8 FALSE NEGATIVES (rule=clean, raters=3)\n" + "=" * 72)
    for i, r in enumerate(rng.sample(fn, 8), 1):
        show(r, i)


if __name__ == "__main__":
    main()
