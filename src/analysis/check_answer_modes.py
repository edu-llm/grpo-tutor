"""How much would the reward move if the student answered in free text?

`HFStudent.choose()` - the reward channel - ranks the options by length-normalized
log-prob behind a bare `Fact: ...\\nQuestion: ...\\nAnswer:` prompt: no chat
template, no room to reason. `HFStudent.choose_free()` lets the same frozen
student write a sentence and maps it back to an option. The two disagree, and
everything downstream (the 731 curated ZPD problems, the QASC/OpenBookQA
baselines, `solved_rate`) is defined by the first one.

This script measures the size of that gap before anyone flips the switch. It
reports both channels under the two conditions the ZPD screen is built on:

    alone   no hint            - the "student fails it alone" premise
    helped  dataset oracle hint - the "but succeeds with help" premise

and the keep rate each channel would produce, which is the number that decides
whether the curated set survives the change.

Measured over 100 QASC train items with Qwen2.5-0.5B-Instruct: alone 0.200 vs
0.250, helped 0.940 vs 0.620, and the two channels agree on only 17% of unaided
items - about what two independent 8-way guesses would manage. The keep rate goes
0.74 -> 0.41.

    python src/check_answer_modes.py --stub --limit 40          # plumbing only
    python src/check_answer_modes.py --source qasc --limit 150   # needs the student
"""

from __future__ import annotations

import argparse
import json

import paths
import seeding
from zpd_filter import SOURCES, StubStudent, load_source


def _rate(xs) -> float:
    return sum(xs) / max(1, len(xs))


def compare(student, items, hint_key: str | None) -> dict:
    """Both channels over `items`; hint_key=None runs the unaided condition."""
    lp, free, agree, rows = [], [], [], []
    for it in items:
        hint = str(it.get(hint_key) or "") if hint_key else ""
        a = student.choose(it["question"], it["choices"], hint=hint)
        b = student.choose_free(it["question"], it["choices"], hint=hint)
        lp.append(int(a == it["gold_idx"]))
        free.append(int(b == it["gold_idx"]))
        agree.append(int(a == b))
        rows.append({"question": it["question"], "gold": it["choices"][it["gold_idx"]],
                     "logprob": it["choices"][a], "free": it["choices"][b]})
    return {"acc_logprob": _rate(lp), "acc_free": _rate(free), "agreement": _rate(agree),
            "correct_logprob": lp, "correct_free": free, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="qasc", choices=sorted(SOURCES))
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stub", action="store_true",
                    help="StubStudent: no model, no GPU - checks the script runs")
    ap.add_argument("--show", type=int, default=5, help="disagreements to print")
    ap.add_argument("--out", default=str(paths.RUNS / "answer_modes.json"))
    args = ap.parse_args()

    seeding.seed_everything(args.seed)
    if args.stub:
        items = [{"question": f"toy question {i}?", "choices": ["a", "b", "c", "d"],
                  "gold_idx": i % 4, "hint": f"fact {i}"} for i in range(args.limit)]
        student = StubStudent()
    else:
        import torch

        from zpd_filter import HFStudent

        items = load_source(args.source, args.limit, seed=args.seed)
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # bf16 on CPU falls back to slow kernels for most ops (see Config.resolve_dtype)
        dtype = torch.float32 if device in ("cpu", "mps") else torch.bfloat16
        student = HFStudent(args.student, device=device, dtype=dtype)

    alone = compare(student, items, hint_key=None)
    helped = compare(student, items, hint_key="hint")

    # the ZPD screen in one line, per channel: fails alone AND solves with help.
    # If these two numbers differ, switching the channel rebuilds the training set.
    keep = {ch: _rate([(1 - a) * h for a, h in zip(alone[f"correct_{ch}"],
                                                   helped[f"correct_{ch}"])])
            for ch in ("logprob", "free")}
    calls = getattr(student, "free_calls", 0)
    unmapped = getattr(student, "free_unmapped", 0)

    print(f"\n=== answer channels: {'stub' if args.stub else args.source}, "
          f"{len(items)} items ===")
    print(f"{'condition':<10}{'logprob':>10}{'free':>10}{'agreement':>12}")
    for name, m in (("alone", alone), ("helped", helped)):
        print(f"{name:<10}{m['acc_logprob']:>10.3f}{m['acc_free']:>10.3f}"
              f"{m['agreement']:>12.3f}")
    print(f"\nZPD keep rate  logprob={keep['logprob']:.3f}  free={keep['free']:.3f}"
          "   (fails alone AND solves with help)")
    if calls:
        # a free reply that names no option falls back to choose(), so a high rate
        # here means the two channels agree partly by construction
        print(f"unparseable free replies: {unmapped}/{calls} "
              f"({unmapped / calls:.1%}) - these fell back to choose()")

    shown = 0
    for a, h in zip(alone["rows"], helped["rows"]):
        if shown >= args.show or a["logprob"] == a["free"]:
            continue
        shown += 1
        print(f"\n  Q: {a['question'][:90]}")
        print(f"     gold={a['gold']!r} logprob={a['logprob']!r} free={a['free']!r}"
              f" | with hint: free={h['free']!r}")

    payload = {"source": "stub" if args.stub else args.source, "n": len(items),
               "seed": args.seed, "keep_rate": keep, "free_calls": calls,
               "free_unmapped": unmapped,
               **{f"{cond}_{k}": v for cond, m in (("alone", alone), ("helped", helped))
                  for k, v in m.items() if k.startswith("acc") or k == "agreement"}}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
