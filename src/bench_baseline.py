"""Measure the student's UNAIDED accuracy on each external benchmark.

A benchmark is only useful for measuring teaching if the student neither aces it
(no headroom) nor floors it (nothing to build on). This reports baseline accuracy
plus the oracle-hint ceiling where the dataset provides one, so we can pick eval
sets on evidence rather than vibes.
"""

from __future__ import annotations

import argparse
import json

import benchmarks
import paths
from zpd_filter import HFStudent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--names", nargs="*", default=sorted(benchmarks.REGISTRY))
    ap.add_argument("--out", default=str(paths.RUNS / "bench_baseline.json"))
    args = ap.parse_args()

    student = HFStudent(args.student)
    rows = []
    for name in args.names:
        items = benchmarks.load_benchmark(name, limit=args.n)
        base = oracle = n_or = 0
        for it in items:
            base += int(student.choose(it["question"], it["choices"]) == it["gold_idx"])
            if it.get("hint"):
                n_or += 1
                oracle += int(student.choose(it["question"], it["choices"],
                                             hint=it["hint"]) == it["gold_idx"])
        n = len(items)
        row = {"benchmark": name, "n": n, "n_choices": sorted({len(i["choices"]) for i in items}),
               "chance": round(sum(1 / len(i["choices"]) for i in items) / n, 3),
               "baseline_acc": round(base / n, 3),
               "oracle_acc": round(oracle / n_or, 3) if n_or else None,
               "oracle_headroom": round(oracle / n_or - base / n, 3) if n_or else None}
        rows.append(row)
        print(json.dumps(row), flush=True)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}")
    print(f"{'benchmark':<15}{'chance':>8}{'baseline':>10}{'oracle':>9}{'headroom':>10}")
    for r in rows:
        o = f"{r['oracle_acc']:.3f}" if r["oracle_acc"] is not None else "-"
        h = f"{r['oracle_headroom']:+.3f}" if r["oracle_headroom"] is not None else "-"
        print(f"{r['benchmark']:<15}{r['chance']:>8.3f}{r['baseline_acc']:>10.3f}{o:>9}{h:>10}")


if __name__ == "__main__":
    main()
