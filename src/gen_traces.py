"""Generate tutoring traces without training, for labelling.

    sbatch scripts/gen_traces.sbatch

Two changes from how v2's traces were produced, both aimed at the same problem:
the student had nothing to say and nothing to be moved off.

  student state   Each item gets a one-line state: the grade, and the option
                  `choose()` actually picks for it. Every training item was
                  screened on that pick being WRONG, so it is a real named
                  misconception rather than an invented one, and the tutor has
                  something specific to work against.

  student stops   The oracle stop consulted gold to decide when the dialogue was
                  over. Here the STUDENT ends it by saying [READY] when something
                  clicks. No gold is involved, so stopping measures whether the
                  tutoring landed rather than whether the answer was reachable.

Each dialogue is scored on the transcript:

  solved          the transcript alone, exactly as v0-v2 measured it

An earlier version also scored with the stated belief prefixed. That is dropped:
the anchor made the student cling to its wrong option (0.058 solved against
0.177), so it measured its own stickiness rather than teaching. The belief stays
in the dialogue, where it gives the tutor something to argue against.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import paths
import rewards
import tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(paths.DATA / "state_tests" / "train_items.jsonl"))
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--teacher-adapter", default=None,
                    help="LoRA to load, e.g. checkpoints-v2/teacher-final")
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--persona-adapter", default="checkpoints/student-persona")
    ap.add_argument("--k", type=int, default=4, help="dialogues per problem")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64, help="dialogues in flight")
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--student-ready", action="store_true",
                    help="let the student end the dialogue. Off by default: both "
                         "versions of this measured agreeableness rather than "
                         "understanding")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/gen_traces.jsonl")
    args = ap.parse_args()

    import config as config_mod
    import seeding
    import train_h100 as T
    from engine import build_engine
    from transformers import AutoTokenizer
    from zpd_filter import HFStudent

    seeding.seed_everything(args.seed)

    cfg = config_mod.Config()
    cfg.backend = "vllm"
    cfg.teacher_model = args.teacher
    cfg.student_model = args.student
    cfg.turns = args.turns
    cfg.temperature = args.temperature
    cfg.gpu_mem_util = args.gpu_mem_util
    cfg.no_sleep = True
    cfg.self_stop = False
    cfg.stop_when_solved = False    # no oracle anywhere in this run
    # Asking the student when to stop was tried twice and failed in opposite
    # directions: told to emit [READY] it complied in 1 dialogue of 1,000; asked
    # yes/no it said yes 86% of the time with the persona on and 98.5% with it
    # off, ending 560 of 614 dialogues after a single tutor turn. Neither
    # measured comprehension - a 0.5B has no metacognition to query. Fixed turns
    # is unbiased and matches every previous run.
    cfg.student_ready = args.student_ready

    problems = [json.loads(l) for l in open(args.problems)]
    print(f"[data] {len(problems)} problems x {args.k} = {len(problems) * args.k} dialogues",
          flush=True)

    # "" or "none" disables it: the persona LoRA is SFT'd toward sounding
    # confused, so whether it makes the student behave worse is testable only
    # if it can be switched off from the command line
    persona = args.persona_adapter if args.persona_adapter not in ("", "none", None) else None
    student = HFStudent(args.student, device="cuda", persona_adapter=persona)
    print(f"[student] persona adapter: {persona or 'OFF'}", flush=True)

    # --- student state: what does this student currently believe, per item? ---
    t0 = time.time()
    for p in problems:
        pick = student.choose(p["question"], p["choices"])
        p["student_state"] = {
            "grade": p.get("grade"),
            "believes": p["choices"][pick],
            "believes_idx": pick,
            "believes_is_gold": bool(pick == p["gold_idx"]),
        }
    wrong = sum(1 for p in problems if not p["student_state"]["believes_is_gold"])
    print(f"[state] built in {time.time() - t0:.0f}s; "
          f"{wrong}/{len(problems)} currently believe a WRONG option", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg.teacher_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eng = build_engine(cfg)
    if args.teacher_adapter:
        # the trained policy lives in a LoRA; without this we would be generating
        # from the base model and calling the result a policy trace
        eng.load_adapter(args.teacher_adapter)
    print(f"[engine] vllm up, adapter={args.teacher_adapter}", flush=True)

    jobs = [p for p in problems for _ in range(args.k)]
    n_done = 0
    with open(args.out, "w") as f:
        for i in range(0, len(jobs), args.batch):
            chunk = jobs[i: i + args.batch]
            transcripts, _, tutor_texts, _, ready, _ = T.run_dialogues(
                cfg, eng, student, chunk, tok, cfg.turns, cfg.temperature,
                states=[p["student_state"] for p in chunk])

            for p, transcript, tutor, rdy in zip(chunk, transcripts, tutor_texts, ready):
                gold = p["choices"][p["gold_idx"]]
                distractors = [c for j, c in enumerate(p["choices"]) if j != p["gold_idx"]]
                # scored on the transcript alone. Prefixing the stated belief made
                # the student cling to it - 0.058 solved against 0.177 - so the
                # anchor was measuring its own stickiness rather than teaching.
                # The belief stays in the dialogue, where it gives the tutor
                # something to argue against, and out of the measurement.
                plain = student.choose(p["question"], p["choices"], hint=transcript)
                leaked = rewards.leaked_answer(tutor, gold, distractors, question=p["question"])
                f.write(json.dumps({
                    "step": -1,
                    "tier": "policy",
                    "prompt": p["question"],
                    "completion": transcript.strip(),
                    "choices": p["choices"],
                    "gold_idx": p["gold_idx"],
                    "gold": gold,
                    "solved": float(plain == p["gold_idx"]),
                    "leaked": leaked,
                    "student_ready": rdy,
                    "turns": transcript.count("Tutor:"),
                    "reward": -1.0 if leaked else float(plain == p["gold_idx"]),
                    "student_answer": p["choices"][plain],
                    "student_believed": p["student_state"]["believes"],
                    "subject": p.get("subject"),
                    "grade": p.get("grade"),
                }) + "\n")
            f.flush()
            n_done += len(chunk)
            print(f"  {n_done}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)

    rows = [json.loads(l) for l in open(args.out)]
    n = len(rows)
    print(f"\nwrote {n} traces -> {args.out}")
    print("solved (plain)    %.3f" % (sum(r["solved"] for r in rows) / n))
    print("leaked            %.3f" % (sum(r["leaked"] for r in rows) / n))
    print("student said READY %.3f" % (sum(r["student_ready"] for r in rows) / n))
    print("mean tutor turns  %.2f" % (sum(r["turns"] for r in rows) / n))
    print("\n(v2 for comparison: solved 0.304, leaked 0.385, turns 2.57)")


if __name__ == "__main__":
    main()
