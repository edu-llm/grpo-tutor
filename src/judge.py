"""LLM-judge labelling of teacher hints, for CALIBRATING the cheap detectors.

Not a reward. The point is to label a few hundred hints once, then measure how
well the free detectors (LeakGuard rules, hint_only_leak) agree with a careful
judgement. After that the free detectors keep running in the loop, but with
known precision and recall instead of guessed ones.

Putting a judge inside the reward is a different and worse idea: the policy
would optimise against the judge, which is a subtler hack than the one we are
already chasing, and it would add ~10k API/GPU calls per training run.

Rubric follows the AI-tutor evaluation taxonomy of Maurya et al. (NAACL 2025,
MRBench), which defines these dimensions with 3-way labels. Using their
definitions rather than inventing our own keeps the numbers comparable to
published work.
"""

from __future__ import annotations

import argparse
import collections
import json
import re

import paths

DIMENSIONS = {
    "revealing": (
        "Revealing of the Answer",
        "Does the tutor response disclose the final answer?\n"
        "  Yes = states or unambiguously names the correct answer\n"
        "  To some extent = strongly narrows it, e.g. rules out all but one option\n"
        "  No = does not give the answer away",
    ),
    "guidance": (
        "Providing Guidance",
        "Does the response offer guidance the student can learn from?\n"
        "  Yes = explanation, worked example, hint, or a probing question\n"
        "  To some extent = some guidance but vague, generic or partly off-target\n"
        "  No = no useful guidance (filler, encouragement only, irrelevant)",
    ),
    "actionability": (
        "Actionability",
        "Does the response make clear what the student should do next?\n"
        "  Yes = a concrete next step\n"
        "  To some extent = implies a direction but leaves it fuzzy\n"
        "  No = conversation would stall here",
    ),
}

PROMPT = """You are grading a tutor's reply for pedagogical quality.

Question the student is stuck on:
{question}

Correct answer: {gold}

Tutor's reply:
{hint}

Dimension: {dim_name}
{dim_def}

Answer with exactly one of: Yes | To some extent | No
Then a short reason on the same line, after a dash.
Format: <label> - <reason>"""

_LABELS = ["to some extent", "yes", "no"]   # check 2-word label first


def parse_label(text: str) -> str | None:
    t = text.strip().lower()
    for lab in _LABELS:
        if t.startswith(lab):
            return lab
    for lab in _LABELS:                      # fall back to first mention
        if re.search(rf"\b{re.escape(lab)}\b", t):
            return lab
    return None


def build_prompts(rows, dim, tok=None):
    name, definition = DIMENSIONS[dim]
    out = []
    for r in rows:
        user = PROMPT.format(question=r.get("prompt", ""), gold=r.get("gold", ""),
                             hint=r.get("completion", ""), dim_name=name,
                             dim_def=definition)
        if tok is not None and hasattr(tok, "apply_chat_template"):
            user = tok.apply_chat_template([{"role": "user", "content": user}],
                                           tokenize=False, add_generation_prompt=True)
        out.append(user)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="path to traces.jsonl")
    ap.add_argument("--judge", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--n", type=int, default=200, help="hints to label")
    ap.add_argument("--dims", nargs="*", default=list(DIMENSIONS))
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--out", default=str(paths.RUNS / "judge_labels.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.traces) if l.strip()]
    rows = [r for r in rows if r.get("completion")][: args.n]
    print(f"[judge] labelling {len(rows)} hints with {args.judge}", flush=True)

    import vllm
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.judge)
    llm = vllm.LLM(model=args.judge, dtype="bfloat16",
                   gpu_memory_utilization=args.gpu_mem_util)
    params = vllm.SamplingParams(temperature=0.0, max_tokens=64)

    labels: dict[str, list] = {}
    for dim in args.dims:
        outs = llm.generate(build_prompts(rows, dim, tok), params)
        labels[dim] = [parse_label(o.outputs[0].text) for o in outs]
        counts = collections.Counter(labels[dim])
        print(f"[judge] {dim}: {dict(counts)}", flush=True)

    # --- agreement of the cheap detectors with the judge ---
    report = {"n": len(rows), "judge": args.judge,
              "distribution": {d: dict(collections.Counter(labels[d])) for d in labels}}

    if "revealing" in labels:
        # judge says leaked if Yes; "to some extent" counted separately since it
        # is exactly the elimination case the rules try to catch
        judge_leak = [l == "yes" for l in labels["revealing"]]
        judge_soft = [l in ("yes", "to some extent") for l in labels["revealing"]]
        for det_name, det in [("leaked_rule", [bool(r.get("leaked", 0.0)) for r in rows]),
                              ("hint_only_leak", [bool(r.get("hint_only_leak", 0.0)) for r in rows])]:
            if not any(det):
                continue
            for strict, jl in [("strict", judge_leak), ("incl_partial", judge_soft)]:
                tp = sum(d and j for d, j in zip(det, jl))
                fp = sum(d and not j for d, j in zip(det, jl))
                fn = sum((not d) and j for d, j in zip(det, jl))
                prec = tp / (tp + fp) if tp + fp else None
                rec = tp / (tp + fn) if tp + fn else None
                report[f"{det_name}/{strict}"] = {
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision": round(prec, 3) if prec is not None else None,
                    "recall": round(rec, 3) if rec is not None else None}

    with open(args.out, "w") as f:
        json.dump({"report": report,
                   "rows": [{**{k: r.get(k) for k in ("prompt", "completion", "gold",
                                                      "leaked", "hint_only_leak")},
                             **{d: labels[d][i] for d in labels}}
                            for i, r in enumerate(rows)]}, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
