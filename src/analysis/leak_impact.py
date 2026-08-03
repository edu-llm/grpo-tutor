"""What does fixing the leak rule do to the runs it has already scored?

    python src/leak_impact.py

Calibration (docs/leak_calibration.md) found the detector fires wrongly about 60%
of the time, that `elimination` is at chance, and that for numeric golds
`_content()` strips the digits so "8 hours" and every distractor collapse to
{hour} - the tutor merely saying "hours" trips two signals at once.

That matters beyond precision. The -1 leak penalty is the largest gradient in
every run so far, so if it fires on the wrong turns, the runs learned to avoid
the detector rather than to avoid leaking. This re-scores saved traces under the
proposed rule and reports how much of each run's headline leak rate survives.

Nothing is retrained and rewards.py is not modified: this reads traces off disk.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rewards import leak_signals, leaked_answer  # noqa: E402


def tutor_text(row):
    if row.get("teacher_text"):
        return row["teacher_text"]
    return "\n".join(ln.split(":", 1)[1].strip()
                     for ln in row.get("completion", "").split("\n")
                     if ln.startswith("Tutor:"))


def proposed(text, gold, distractors, question):
    """verbatim OR identifying_hits >= 1 - the rule calibration selected."""
    s = leak_signals(text, gold, distractors, question=question)
    return float(s["verbatim"] >= 1.0 or s.get("identifying", 0) > 0)


def main():
    runs = [
        ("v2 policy (19405111)", "runs/20260731-212530/traces.jsonl"),
        ("state-test regen", "runs/gen_traces.jsonl"),
        ("hand-written tier", "runs/good_traces_all.jsonl"),
    ]
    print(f"{'run':<24} {'n':>6} {'current':>9} {'proposed':>9} {'changed':>9}")
    for name, path in runs:
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path)]
        cur = new = flip = 0
        for r in rows:
            t = tutor_text(r)
            gold = r["gold"]
            distractors = [c for j, c in enumerate(r["choices"]) if j != r["gold_idx"]]
            a = float(leaked_answer(t, gold, distractors, question=r["prompt"]))
            b = proposed(t, gold, distractors, r["prompt"])
            cur += a
            new += b
            flip += (a != b)
        n = len(rows)
        print(f"{name:<24} {n:>6} {cur / n:>9.3f} {new / n:>9.3f} {flip / n:>8.1%}")

    # the trend is what three runs concluded from; does it survive the fix?
    path = "runs/20260731-212530/traces.jsonl"
    if os.path.exists(path):
        rows = [json.loads(l) for l in open(path)]
        print("\nv2 leak rate by 50-step window, current rule vs proposed")
        print(f"{'steps':<10} {'current':>9} {'proposed':>9}")
        for lo, hi in ((0, 49), (50, 99), (100, 149), (150, 199), (200, 249)):
            w = [r for r in rows if lo <= r["step"] <= hi]
            if not w:
                continue
            c = n2 = 0
            for r in w:
                t = tutor_text(r)
                d = [x for j, x in enumerate(r["choices"]) if j != r["gold_idx"]]
                c += float(leaked_answer(t, r["gold"], d, question=r["prompt"]))
                n2 += proposed(t, r["gold"], d, r["prompt"])
            print(f"{lo:>3}-{hi:<6} {c / len(w):>9.3f} {n2 / len(w):>9.3f}")


if __name__ == "__main__":
    main()
