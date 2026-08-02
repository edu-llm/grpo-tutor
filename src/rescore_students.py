"""Re-score existing transcripts with different students and answer channels.

    sbatch scripts/rescore.sbatch

Three runs found nothing to learn, hand-written expert tutoring scored no better
than a mediocre policy (0.293 vs 0.310), and a hint written for a DIFFERENT
problem is worth +0.006. Every one of those is consistent with a student that
cannot use the tutoring it is given.

"Cannot" has two candidate causes and they need different fixes:

  capacity   0.5B may simply lack the ability.
  channel    `choose()` ranks options by log-prob after a bare
             "Fact: <transcript>\\nQuestion: ...\\nAnswer:" - no chat template, no
             room to reason. Told "divide 45.5 by 7", nothing in that channel
             divides anything.

This separates them without generating a single new dialogue: the SAME
transcripts are re-read by bigger students, through both channels. The dialogues
are fixed, so any change in the numbers is the reader, not the tutoring.

The decisive comparison is hand-written vs policy WITHIN each cell. If the gap
opens up as the student grows, the student was the bottleneck and the project has
somewhere to go. If it stays at zero everywhere, the task cannot register
teaching quality and no reward design will rescue it.
"""

from __future__ import annotations

import argparse
import json
import re
import time

import paths

ANSWER_RE = re.compile(r"\b([A-D])\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--good", default="runs/good_traces_all.jsonl")
    ap.add_argument("--policy", default="runs/gen_traces.jsonl")
    ap.add_argument("--students", nargs="+",
                    default=["Qwen/Qwen2.5-0.5B-Instruct",
                             "Qwen/Qwen2.5-3B-Instruct",
                             "Qwen/Qwen2.5-7B-Instruct"])
    ap.add_argument("--max-policy", type=int, default=600)
    ap.add_argument("--out", default="runs/rescore.json")
    args = ap.parse_args()

    import torch
    from zpd_filter import HFStudent

    good = [json.loads(l) for l in open(args.good)]
    good_qs = {g["prompt"] for g in good}
    policy = [json.loads(l) for l in open(args.policy) if json.loads(l)["prompt"] in good_qs]
    policy = policy[: args.max_policy]
    print(f"{len(good)} hand-written and {len(policy)} policy transcripts "
          f"on the same {len(good_qs)} questions", flush=True)

    results = {}
    for name in args.students:
        t0 = time.time()
        student = HFStudent(name, device="cuda")
        cell = {}
        for tier, rows in (("good", good), ("policy", policy)):
            logprob = free = 0
            for r in rows:
                q, ch, gi = r["prompt"], r["choices"], r["gold_idx"]
                logprob += int(student.choose(q, ch, hint=r["completion"]) == gi)
                # the reasoning channel: chat template, room to think, then commit
                view = (f"A tutor just helped a student with this question.\n\n"
                        f"Conversation:\n{r['completion']}\n\n"
                        f"Question: {q}\n"
                        + "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(ch))
                        + "\n\nThink it through in one sentence, then finish with "
                          "'Answer: X' where X is the letter.")
                out = student.reply([view], max_new_tokens=90)[0]
                tail = out.split("Answer:")[-1] if "Answer:" in out else out[-12:]
                m = ANSWER_RE.search(tail.upper())
                free += int(bool(m) and (ord(m.group(1)) - 65) == gi)
            cell[tier] = {"n": len(rows),
                          "logprob": logprob / len(rows),
                          "free_text": free / len(rows)}
        cell["gap_logprob"] = cell["good"]["logprob"] - cell["policy"]["logprob"]
        cell["gap_free_text"] = cell["good"]["free_text"] - cell["policy"]["free_text"]
        results[name] = cell
        print(f"\n{name}   ({time.time() - t0:.0f}s)", flush=True)
        print(f"  log-prob   good {cell['good']['logprob']:.3f}  "
              f"policy {cell['policy']['logprob']:.3f}  gap {cell['gap_logprob']:+.3f}",
              flush=True)
        print(f"  free-text  good {cell['good']['free_text']:.3f}  "
              f"policy {cell['policy']['free_text']:.3f}  gap {cell['gap_free_text']:+.3f}",
              flush=True)
        del student
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {args.out}")
    print("\nA gap that grows with student size means the student was the "
          "bottleneck.\nA gap near zero everywhere means the task cannot see "
          "teaching quality at all.")


if __name__ == "__main__":
    main()
