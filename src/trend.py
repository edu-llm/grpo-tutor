"""Quick trend over a run's metrics.jsonl: first vs last window for every key."""

from __future__ import annotations

import glob
import json
import statistics as st
import sys


def main():
    path = (sys.argv[1] if len(sys.argv) > 1
            else sorted(glob.glob("runs/*/metrics.jsonl"))[-1])
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        raise SystemExit(f"no metrics in {path}")
    keys = sorted({k for r in rows for k in r
                   if isinstance(r.get(k), (int, float)) and k != "step"})
    w = max(3, len(rows) // 5)
    print(f"{path}   rows={len(rows)}  window={w}")
    print(f"{'metric':<26}{'first':>10}{'last':>10}{'delta':>10}")
    for k in keys:
        vals = [(r["step"], r[k]) for r in rows if k in r]
        if len(vals) < 4:
            continue
        a = st.fmean(v for _, v in vals[:w])
        b = st.fmean(v for _, v in vals[-w:])
        star = "  <<<" if abs(b - a) > 0.15 * max(1e-9, abs(a)) else ""
        print(f"{k:<26}{a:>10.4f}{b:>10.4f}{b - a:>+10.4f}{star}")


if __name__ == "__main__":
    main()
