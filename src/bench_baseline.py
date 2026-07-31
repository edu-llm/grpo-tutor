"""Measure the student's UNAIDED accuracy on each external benchmark.

A benchmark is only useful for measuring teaching if the student neither aces it
(no headroom) nor floors it (nothing to build on). This reports baseline accuracy
plus the oracle-hint ceiling where the dataset provides one, so we can pick eval
sets on evidence rather than vibes.

`--probe` adds the two conditions that say whether the ceiling is REACHABLE:

    hint_only     hint + choices, question hidden - can the hint alone do it
    choices_only  choices alone - the floor hint_only has to clear

Headroom without those is not interpretable. QASC's famous +0.64 is the case in
point: the student scores 0.753 from the hint alone against a 0.107 floor, so
almost all of the "teaching gain" is copying, and no tutor bound by LeakGuard can
reach it. Report `hint_only - choices_only` next to the headroom, always.
"""

from __future__ import annotations

import argparse
import json

import benchmarks
import paths
import rewards
from zpd_filter import HFStudent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--names", nargs="*", default=sorted(benchmarks.REGISTRY))
    ap.add_argument("--probe", action="store_true",
                    help="also measure hint_only and its choices_only floor")
    ap.add_argument("--out", default=str(paths.RUNS / "bench_baseline.json"))
    args = ap.parse_args()

    student = HFStudent(args.student)
    rows = []
    for name in args.names:
        items = benchmarks.load_benchmark(name, limit=args.n)
        base = oracle = n_or = 0
        hint_only = choices_only = zpd = 0
        for it in items:
            alone = student.choose(it["question"], it["choices"]) == it["gold_idx"]
            base += int(alone)
            if it.get("hint"):
                n_or += 1
                helped = student.choose(it["question"], it["choices"],
                                        hint=it["hint"]) == it["gold_idx"]
                oracle += int(helped)
                zpd += int(helped and not alone)
                if args.probe:
                    hint_only += rewards.hint_only_leak(
                        student, it["hint"], it["choices"], it["gold_idx"])
                    choices_only += rewards.choices_only_baseline(
                        student, it["choices"], it["gold_idx"])
        n = len(items)
        row = {"benchmark": name, "n": n, "n_choices": sorted({len(i["choices"]) for i in items}),
               "chance": round(sum(1 / len(i["choices"]) for i in items) / n, 3),
               "baseline_acc": round(base / n, 3),
               "oracle_acc": round(oracle / n_or, 3) if n_or else None,
               "oracle_headroom": round(oracle / n_or - base / n, 3) if n_or else None,
               # the fraction the ZPD screen would keep, before the free-text screen
               "zpd_keep": round(zpd / n_or, 3) if n_or else None}
        if args.probe and n_or:
            row.update(hint_only=round(hint_only / n_or, 3),
                       choices_only=round(choices_only / n_or, 3),
                       leak_above_floor=round((hint_only - choices_only) / n_or, 3))
        rows.append(row)
        print(json.dumps(row), flush=True)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}")
    extra = f"{'hint_only':>11}{'floor':>8}{'above':>8}" if args.probe else ""
    print(f"{'benchmark':<22}{'chance':>8}{'baseline':>10}{'oracle':>9}"
          f"{'headroom':>10}{'zpd_keep':>10}{extra}")
    for r in rows:
        o = f"{r['oracle_acc']:.3f}" if r["oracle_acc"] is not None else "-"
        h = f"{r['oracle_headroom']:+.3f}" if r["oracle_headroom"] is not None else "-"
        k = f"{r['zpd_keep']:.3f}" if r["zpd_keep"] is not None else "-"
        line = (f"{r['benchmark']:<22}{r['chance']:>8.3f}{r['baseline_acc']:>10.3f}"
                f"{o:>9}{h:>10}{k:>10}")
        if args.probe and "hint_only" in r:
            line += (f"{r['hint_only']:>11.3f}{r['choices_only']:>8.3f}"
                     f"{r['leak_above_floor']:>+8.3f}")
        print(line)


if __name__ == "__main__":
    main()
