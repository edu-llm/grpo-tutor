"""Prove the persona adapter does not touch the reward channel.

The student is fine-tuned to TALK like a kid. The reward comes from choose(),
which runs with the adapter disabled. If that wiring is wrong, every number in
the project moves silently - the ZPD set was curated with choose(), and so were
the QASC baselines.

So: run choose() on the same items with and without the adapter loaded, and
require the predictions to be IDENTICAL, not merely similar. Also report the
reply() difference, which should be large - that is the change we paid for.
"""

from __future__ import annotations

import argparse

import benchmarks
import tasks
from zpd_filter import HFStudent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="checkpoints/student-persona")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--benchmark", default="qasc")
    args = ap.parse_args()

    items = benchmarks.load_benchmark(args.benchmark, limit=args.n)

    base = HFStudent(args.student)
    tuned = HFStudent(args.student, persona_adapter=args.adapter)

    same = correct_b = correct_t = 0
    for p in items:
        b = base.choose(p["question"], p["choices"])
        t = tuned.choose(p["question"], p["choices"])
        same += int(b == t)
        correct_b += int(b == p["gold_idx"])
        correct_t += int(t == p["gold_idx"])

    n = len(items)
    print(f"items                    : {n}")
    print(f"choose() identical       : {same}/{n} ({same / n:.1%})")
    print(f"choose() accuracy base   : {correct_b / n:.3f}")
    print(f"choose() accuracy tuned  : {correct_t / n:.3f}")
    verdict = "PASS - reward channel untouched" if same == n else \
              "FAIL - the adapter is leaking into scoring"
    print(f"\n{verdict}")

    # the part that SHOULD differ
    views = [tasks.student_opening_view(p) for p in items[:6]]
    print("\nreply() base vs tuned (should look different):")
    for v, b, t in zip(views, base.reply(views), tuned.reply(views)):
        q = v.split("\n")[1][:60]
        print(f"  Q {q}")
        print(f"    base  {b[:90]}")
        print(f"    tuned {t[:90]}")

    raise SystemExit(0 if same == n else 1)


if __name__ == "__main__":
    main()
