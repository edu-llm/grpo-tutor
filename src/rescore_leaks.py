"""Re-score saved traces with the leak rules, old vs new, and report what moved.

The leak rate written into a run is only as good as the rules at the time it
ran. After changing those rules you want to know two things before trusting a
new number: how many labels changed, and whether the change is concentrated in
the cases the fix was aimed at. This prints both.

    python src/rescore_leaks.py runs/<id>/traces.jsonl [--limit N] [--show 8]

"old" is the pre-fix decision (verbatim / length-normalised overlap /
elimination). "new" adds the identifying-token rule.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rewards import (leak_signals, _identifying_words,  # noqa: E402
                     _identifying_numbers)


def tutor_only(row: dict) -> str:
    """The tutor's words. Scoring the whole transcript charges it for the
    student's blurting, which is a different bug we already fixed once."""
    if row.get("teacher_text"):
        return row["teacher_text"]
    text = row.get("completion", "")
    turns = [ln.split(":", 1)[1] for ln in text.split("\n") if ln.startswith("Tutor:")]
    return "\n".join(turns) if turns else text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows")
    ap.add_argument("--show", type=int, default=6, help="examples of newly caught leaks")
    ap.add_argument("--step", type=int, default=None, help="only rows from this step")
    args = ap.parse_args()

    rows = []
    with open(args.path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if args.step is not None and r.get("step") != args.step:
                continue
            if r.get("choices") and r.get("gold_idx") is not None:
                rows.append(r)
            if args.limit and len(rows) >= args.limit:
                break

    if not rows:
        print("no scorable rows found (need 'choices' and 'gold_idx')")
        return

    old_n = new_n = 0
    caught, no_signal = [], 0
    for r in rows:
        gold = r.get("gold") or r["choices"][r["gold_idx"]]
        distractors = [c for j, c in enumerate(r["choices"]) if j != r["gold_idx"]]
        question = r.get("prompt", "")
        text = tutor_only(r)
        sig = leak_signals(text, gold, distractors, question)
        old = (sig["verbatim"] >= 1.0 or sig["overlap"] >= 0.6
               or sig["elimination"] >= 0.5)
        new = old or sig["identifying_hits"] >= 1
        old_n += old
        new_n += new
        ident = (_identifying_words(gold, distractors, question)
                 | _identifying_numbers(gold, distractors, question))
        if not ident:
            no_signal += 1
        if new and not old:
            caught.append((gold, sorted(ident), text, sig))

    n = len(rows)
    print(f"rows scored: {n}   file: {args.path}")
    print(f"  old rule : {old_n / n:.2%}  ({old_n})")
    print(f"  new rule : {new_n / n:.2%}  ({new_n})")
    print(f"  changed  : {len(caught) / n:.2%}  ({len(caught)} newly flagged, 0 unflagged)")
    print(f"  items with no identifying token at all: {no_signal / n:.1%} "
          f"(new rule structurally cannot fire on these)")

    for gold, ident, text, sig in caught[: args.show]:
        print("-" * 88)
        print("GOLD :", str(gold)[:110])
        print("IDENT:", ident, f"| old overlap {sig['overlap']:.3f} (needed 0.6)")
        print("TUTOR:", " ".join(text.split())[:280])


if __name__ == "__main__":
    main()
