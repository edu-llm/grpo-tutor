"""Generate a GOOD tutoring tier to sit alongside the run's own traces.

    export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY
    python src/gen_good_traces.py --n 120 --out runs/good_traces.jsonl

Why this exists: every trace we have comes from one mediocre policy, so a reward
model trained on them learns "least bad" and GRPO climbs to exactly that ceiling.
Preference pairs of mid-vs-mid are mostly annotator noise; good-vs-mid is the
comparison that carries signal.

The student is the REAL frozen Qwen2.5-0.5B replying in the loop, not a scripted
reply, and the output uses the run's trace schema. If the student turns looked
different here, a reward model could separate the tiers on surface cues instead
of on teaching quality, which is the one thing it must not learn.

Runs on a laptop: the student is 0.5B and goes on MPS or CPU. Only the tutor
needs the network.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

import paths
import rewards

TUTOR_SYSTEM = """You are an expert tutor working with a struggling 7th grader.

Your goal is that the student can answer THIS question themselves. You have one
hard rule and a few habits.

THE RULE: never state the answer, never name the correct option, and never rule
out options for them. If your message would let someone pick the answer without
thinking, rewrite it.

HABITS OF GOOD TUTORING:
- Diagnose first. Their confusion is specific; find it before explaining.
- Ask one question at a time, and make it a question they can actually answer.
- Give them the missing PIECE, not the conclusion: the principle, the unit, the
  definition, the thing to compare.
- For a maths item, walk the procedure on the numbers in front of them.
- Build from what they just said, even when it is wrong - especially then.
- Two or three sentences. A wall of text is not teaching.
"""

STUDENT_OPENERS = [
    "I'm stuck on this one.",
    "I don't really get what it's asking.",
    "hmm i'm not sure where to start",
]


def tutor_reply(client, kind, question, choices, transcript):
    view = (f"The student is working on this question:\n{question}\n"
            + "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
            + "\n\nDO NOT reveal or name the correct option.\n\n"
            + f"Conversation so far:\n{transcript or '(nothing yet)'}\n\n"
            "Write your next tutor message only.")
    if kind == "anthropic":
        r = client.messages.create(
            model="claude-sonnet-4-5", max_tokens=220, system=TUTOR_SYSTEM,
            messages=[{"role": "user", "content": view}])
        return r.content[0].text.strip()
    r = client.chat.completions.create(
        model="gpt-4o", max_tokens=220,
        messages=[{"role": "system", "content": TUTOR_SYSTEM},
                  {"role": "user", "content": view}])
    return r.choices[0].message.content.strip()


def make_client():
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        return "anthropic", anthropic.Anthropic()
    if os.environ.get("OPENAI_API_KEY"):
        import openai
        return "openai", openai.OpenAI()
    raise SystemExit("Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(paths.DATA / "state_tests" / "train_items.jsonl"))
    ap.add_argument("--n", type=int, default=120, help="how many problems to tutor")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default=None, help="mps / cuda / cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/good_traces.jsonl")
    args = ap.parse_args()

    kind, client = make_client()
    print(f"tutor: {kind}")

    import torch
    from zpd_filter import HFStudent
    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"student: {args.student} on {device}")
    student = HFStudent(args.student, device=device)

    rng = random.Random(args.seed)
    items = [json.loads(l) for l in open(args.items)]
    rng.shuffle(items)
    items = items[: args.n]

    done = 0
    with open(args.out, "w") as f:
        for it in items:
            q, choices, gi = it["question"], it["choices"], it["gold_idx"]
            gold = choices[gi]
            distractors = [c for j, c in enumerate(choices) if j != gi]

            lines = [f"Student: {rng.choice(STUDENT_OPENERS)}"]
            tutor_only = []
            for _ in range(args.turns):
                try:
                    msg = tutor_reply(client, kind, q, choices, "\n".join(lines))
                except Exception as e:
                    print(f"  tutor call failed ({e}); skipping item")
                    lines = None
                    break
                msg = re.sub(r"\s+", " ", msg).strip()
                lines.append(f"Tutor: {msg}")
                tutor_only.append(msg)
                # the real student replies, so these traces are indistinguishable
                # from training traces except in the tutor's quality
                reply = student.reply(["\n".join(lines)], max_new_tokens=48)[0]
                lines.append(f"Student: {reply.strip()}")
            if lines is None:
                continue

            transcript = "\n".join(lines)
            tutor_text = "\n".join(tutor_only)
            solved = float(student.choose(q, choices, hint=transcript) == gi)
            leaked = rewards.leaked_answer(tutor_text, gold, distractors, question=q)
            f.write(json.dumps({
                "step": -1,                      # not from a training step
                "tier": "good",
                "prompt": q,
                "completion": transcript,
                "choices": choices,
                "gold_idx": gi,
                "gold": gold,
                "solved": solved,
                "leaked": leaked,
                "turns": args.turns,
                "reward": -1.0 if leaked else solved,
                "student_answer": choices[student.choose(q, choices, hint=transcript)],
            }) + "\n")
            f.flush()
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(items)}")

    print(f"\nwrote {done} good traces -> {args.out}")


if __name__ == "__main__":
    main()
