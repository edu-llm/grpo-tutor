"""Separate "stopped cheating" from "stopped teaching".

Overall solve rate conflates two very different things: dialogues solved because
the tutor leaked the answer, and dialogues solved because the tutor taught. When
leaking falls, the first kind disappears and the total necessarily drops - that
is the cheat being removed, not a regression.

The honest signal is the solve rate among dialogues that did NOT leak. If that
holds or rises while leaking falls, the tutor is converting cheating into
teaching. If it falls too, the tutor is going quiet.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="?", default=None)
    ap.add_argument("--bins", type=int, default=8)
    args = ap.parse_args()

    path = args.traces or sorted(glob.glob("runs/*/traces.jsonl"))[-1]
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "step" in r]
    if not rows:
        raise SystemExit(f"no traces with a step field in {path}")
    rows.sort(key=lambda r: r["step"])

    size = max(1, len(rows) // args.bins)
    chunks = [rows[i:i + size] for i in range(0, len(rows), size)][: args.bins]
    print(f"{path}   {len(rows)} dialogues\n")
    print(f"{'step':>7}{'leak':>8}{'solved':>8}{'clean-solved':>14}{'n_clean':>9}")
    for c in chunks:
        clean = [r for r in c if not r.get("leaked")]
        leak = st.fmean(float(bool(r.get("leaked"))) for r in c)
        solved = st.fmean(float(bool(r.get("solved"))) for r in c)
        clean_solved = (st.fmean(float(bool(r.get("solved"))) for r in clean)
                        if clean else float("nan"))
        print(f"{c[0]['step']:>7}{leak:>8.3f}{solved:>8.3f}{clean_solved:>14.3f}{len(clean):>9}")

    print("\nclean-solved = solve rate among dialogues the tutor did NOT leak in.")
    print("Rising while leak falls  -> cheating is being replaced by teaching.")
    print("Falling while leak falls -> the tutor is just saying less.")


if __name__ == "__main__":
    main()
