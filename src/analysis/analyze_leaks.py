"""Categorise how the teacher leaks, from a run's traces.

The aggregate leak rate says how often, not how. The fix differs completely by
mode: naming the answer needs a task/reward change, enumerating the option list
needs the choices hidden from the teacher.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re


def tutor_lines(rec):
    return [l[len("Tutor:"):].strip()
            for l in rec.get("completion", "").split("\n") if l.startswith("Tutor:")]


def classify(rec):
    text = " ".join(tutor_lines(rec)) or rec.get("completion", "")
    low = text.lower()
    gold = str(rec.get("gold", "")).lower().strip()
    names_gold = bool(gold) and gold in low
    # quoting several option-looking strings, or explicitly talking about choices
    quoted = len(re.findall(r'"[^"]{2,40}"', text))
    enumerates = quoted >= 2 or any(p in low for p in
                                    ("each choice", "each option", "the options",
                                     "look at the choices", "rule out"))
    if names_gold and enumerates:
        return "names gold AND walks the option list"
    if names_gold:
        return "names the gold answer outright"
    if enumerates:
        return "walks/eliminates the option list"
    return "paraphrase or overlap rule fired"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=None, help="traces.jsonl (default: newest run)")
    ap.add_argument("--show", type=int, default=2, help="examples per mode")
    args = ap.parse_args()

    path = args.traces or sorted(glob.glob("runs/*/traces.jsonl"))[-1]
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        raise SystemExit(f"no traces in {path}")

    n = len(rows)
    leaked = [r for r in rows if r.get("leaked")]
    solved = [r for r in rows if r.get("solved")]
    print(f"file    : {path}")
    print(f"traces  : {n}")
    print(f"leaked  : {len(leaked)} ({len(leaked)/n:.0%})")
    print(f"solved  : {len(solved)} ({len(solved)/n:.0%})")
    both = sum(1 for r in rows if r.get("leaked") and r.get("solved"))
    print(f"leaked AND solved: {both} ({both/max(1,len(leaked)):.0%} of leaks)")
    clean_solved = sum(1 for r in rows if not r.get("leaked") and r.get("solved"))
    clean = n - len(leaked)
    print(f"solved WITHOUT leaking: {clean_solved}/{clean}"
          f" ({clean_solved/max(1,clean):.0%} of clean dialogues)")

    modes = collections.Counter(classify(r) for r in leaked)
    print("\nleak modes:")
    for k, v in modes.most_common():
        print(f"  {v:4d}  ({v/max(1,len(leaked)):3.0%})  {k}")

    by_mode: dict[str, list] = collections.defaultdict(list)
    for r in leaked:
        by_mode[classify(r)].append(r)
    for mode, recs in by_mode.items():
        print(f"\n=== {mode} ===")
        for r in recs[: args.show]:
            print(f"  gold={r.get('gold')!r}  answered={r.get('student_answer')!r}")
            for line in r.get("completion", "").split("\n"):
                if line.strip():
                    print(f"    {line.strip()[:160]}")
            print()


if __name__ == "__main__":
    main()
