"""Trend over a run's metrics.jsonl.

Default prints binned averages across the run, because first-vs-last hides
non-monotonic behaviour - a metric that falls then climbs back looks flat.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st

WATCH = ["reward", "leak_rate", "solved_rate", "hint_only_leak", "tokens",
         "gen/mean_len", "entropy", "kl", "zero_adv_frac", "clip_frac", "loss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", nargs="?", default=None)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--all", action="store_true", help="every key, not just WATCH")
    args = ap.parse_args()

    path = args.metrics or sorted(glob.glob("runs/*/metrics.jsonl"))[-1]
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "step" in r]
    if not rows:
        raise SystemExit(f"no metrics in {path}")
    rows.sort(key=lambda r: r["step"])

    keys = (sorted({k for r in rows for k in r
                    if isinstance(r.get(k), (int, float)) and k != "step"})
            if args.all else [k for k in WATCH if any(k in r for r in rows)])

    n = len(rows)
    size = max(1, n // args.bins)
    chunks = [rows[i:i + size] for i in range(0, n, size)][: args.bins]
    print(f"{path}  rows={n}  bins of {size} steps\n")
    hdr = "".join(f"{c[0]['step']:>9}" for c in chunks)
    print(f"{'step ->':<18}{hdr}")
    for k in keys:
        cells = ""
        for c in chunks:
            vals = [r[k] for r in c if k in r]
            cells += f"{st.fmean(vals):>9.3f}" if vals else f"{'-':>9}"
        print(f"{k:<18}{cells}")


if __name__ == "__main__":
    main()
