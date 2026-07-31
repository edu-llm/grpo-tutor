"""Build SFT data to make the student sound like a stuck 7th grader.

Prompting does not work: tightening STUDENT_SYSTEM from a loose persona to five
explicit rules moved the median reply length by zero words (25 before, 25 after).
A 0.5B instruct model is trained to be a helpful explainer and will not hold a
persona against that. So we fine-tune instead.

Contexts come from REAL dialogues in a run's traces, not synthetic ones, so the
student learns to respond to the tutor prompts it will actually see. Only the
replies are generated, by a larger model, and then filtered hard:

  - must be short (a stuck kid does not lecture)
  - must NOT contain the gold answer (the student blurting it is what corrupted
    the reward in the first place)
  - must not read like a textbook

Output: JSONL of {"system", "user", "assistant"} ready for sft_student.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re

import paths
import tasks
from zpd_filter import HFStudent

GEN_SYSTEM = (
    "You write realistic dialogue for a struggling 7th-grade student talking to "
    "a tutor. You never break character and never solve the problem."
)

GEN_PROMPT = """Here is how this student talks:
{examples}

Now here is what the student is looking at:
---
{view}
---

Write ONLY the student's next line. Requirements:
- ONE sentence, under 15 words
- casual and unsure, like a kid typing quickly
- say what is confusing, guess vaguely, or ask something back
- NEVER state the correct answer, and never explain like a textbook
Student:"""


def load_seeds(path=None, k=5):
    path = path or str(paths.DATA / "student_persona_synthetic.jsonl")
    rows = [json.loads(l)["text"] for l in open(path) if l.strip()]
    return rows, k


def contexts_from_traces(trace_paths, limit):
    """Rebuild the view the student had at each point in real dialogues."""
    out = []
    for tp in trace_paths:
        for line in open(tp):
            if not line.strip():
                continue
            r = json.loads(line)
            convo, choices = r.get("completion", ""), r.get("choices")
            if not r.get("turns") or not choices:
                continue
            problem = {"question": r.get("prompt", ""), "choices": choices,
                       "gold_idx": r.get("gold_idx", 0)}
            gold = r.get("gold", "")
            # the opening turn, before the tutor has said anything
            out.append((tasks.student_opening_view(problem), gold))
            # and after each tutor turn
            partial = ""
            for ln in convo.split("\n"):
                if ln.startswith("Tutor:"):
                    partial += ln + "\n"
                    out.append((tasks.student_dialogue_view(problem, partial), gold))
                elif ln.startswith("Student:"):
                    partial += ln + "\n"
            if len(out) >= limit * 3:
                return out
    return out


def acceptable(reply: str, gold: str) -> bool:
    r = reply.strip()
    if not r or len(r.split()) > 18:
        return False
    if gold and gold.lower().strip() in r.lower():
        return False                       # the whole point: do not blurt it
    if r.count(".") > 1 or ":" in r or "\n" in r:
        return False                       # multi-sentence or list-like
    if re.search(r"\b(therefore|thus|in conclusion|is defined as|refers to)\b", r, re.I):
        return False                       # textbook register
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--traces", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=1500, help="target number of pairs")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--out", default=str(paths.DATA / "student_sft.jsonl"))
    args = ap.parse_args()

    seeds, k = load_seeds()
    traces = args.traces or sorted(glob.glob(str(paths.RUNS / "*" / "traces.jsonl")))
    if not traces:
        raise SystemExit("no traces found; point --traces at a run's traces.jsonl")
    ctxs = contexts_from_traces(traces, args.n)
    rng = random.Random(0)
    rng.shuffle(ctxs)
    ctxs = ctxs[: args.n * 2]              # over-generate, the filter is strict
    print(f"[persona] {len(ctxs)} contexts from {len(traces)} trace file(s)", flush=True)

    import vllm
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.generator)
    llm = vllm.LLM(model=args.generator, dtype="bfloat16",
                   gpu_memory_utilization=args.gpu_mem_util)
    params = vllm.SamplingParams(temperature=0.9, top_p=0.95, max_tokens=40, n=1)

    prompts = []
    for view, _ in ctxs:
        ex = "\n".join(f"- {s}" for s in rng.sample(seeds, k))
        user = GEN_PROMPT.format(examples=ex, view=view)
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))

    outs = llm.generate(prompts, params)
    kept, rejected = [], 0
    for (view, gold), o in zip(ctxs, outs):
        reply = o.outputs[0].text.strip().split("\n")[0].lstrip("-").strip()
        reply = reply.removeprefix("Student:").strip().strip('"')
        if acceptable(reply, gold):
            kept.append({"system": HFStudent.STUDENT_SYSTEM, "user": view,
                         "assistant": reply})
        else:
            rejected += 1
        if len(kept) >= args.n:
            break

    with open(args.out, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    lens = [len(r["assistant"].split()) for r in kept]
    print(f"[persona] kept {len(kept)}, rejected {rejected} "
          f"({rejected / max(1, rejected + len(kept)):.0%})")
    print(f"[persona] median reply length: {sorted(lens)[len(lens) // 2]} words "
          f"(current student: 25)")
    print(f"[persona] wrote {args.out}")
    for r in kept[:5]:
        print(f"    {r['assistant']!r}")


if __name__ == "__main__":
    main()
