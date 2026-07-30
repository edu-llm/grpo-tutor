"""Is the student actually unable to answer, or only unable through choose()?

The ZPD set is curated with student.choose(), a log-prob comparison over the
options with no room to reason. student.reply() is free-form generation. If the
model can name gold in free text on problems where choose() scores 0%, then the
"student cannot solve this" premise is a property of the SCORING CHANNEL, not of
the student's knowledge - and the teaching gain we measure is partly the hint
unlocking something already there.
"""

from __future__ import annotations

import argparse
import json
import re

import paths
import tasks
from zpd_filter import HFStudent


ASK = ("Question you're stuck on:\n{q}\n{choices}\n\n"
       "Just say which option you think it is and why, in one sentence.")


def names_gold(text: str, gold: str, distractors) -> bool:
    """Gold mentioned and no distractor mentioned (avoid crediting a scattershot)."""
    low = text.lower()
    g = gold.lower().strip()
    if not g or g not in low:
        return False
    return not any(d.lower().strip() and d.lower().strip() in low for d in distractors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default=str(paths.RUNS / "student_knows.json"))
    args = ap.parse_args()

    problems = tasks.load_zpd()
    train, _ = tasks.split_problems(problems, test_frac=0.15, seed=0)
    items = train[: args.n]
    student = HFStudent(args.student)

    # free-text: no tutor, no help, just answer it
    views = [ASK.format(q=p["question"], choices=tasks.format_choices(p["choices"]))
             for p in items]
    replies = []
    B = 32
    for i in range(0, len(views), B):
        replies.extend(student.reply(views[i:i + B], max_new_tokens=60))

    choose_ok = free_ok = 0
    rows = []
    for p, rep in zip(items, replies):
        gold_idx = p["gold_idx"]
        gold = p["choices"][gold_idx]
        distractors = [c for j, c in enumerate(p["choices"]) if j != gold_idx]
        c_ok = student.choose(p["question"], p["choices"]) == gold_idx
        f_ok = names_gold(rep, gold, distractors)
        choose_ok += c_ok
        free_ok += f_ok
        rows.append({"question": p["question"], "gold": gold,
                     "choose_correct": bool(c_ok), "free_text_correct": bool(f_ok),
                     "reply": rep})

    m = len(items)
    print(f"ZPD training problems tested : {m}")
    print(f"  (curated so choose() fails on all of them)")
    print(f"choose()   correct           : {choose_ok}/{m} ({choose_ok/m:.0%})")
    print(f"free-text  correct           : {free_ok}/{m} ({free_ok/m:.0%})")
    both = sum(1 for r in rows if r["free_text_correct"] and not r["choose_correct"])
    print(f"knows it in TEXT but fails choose(): {both}/{m} ({both/m:.0%})")
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\nexamples where free text is right but choose() is wrong:")
    for r in [x for x in rows if x["free_text_correct"] and not x["choose_correct"]][:3]:
        print(f"  gold={r['gold']!r}")
        print(f"    {re.sub(chr(10), ' ', r['reply'])[:150]}")


if __name__ == "__main__":
    main()
