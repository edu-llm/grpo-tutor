"""Build the reward-model labelling bundle from a run's traces.

    python src/build_label_set.py --run runs/20260731-212530

The unit of labelling is a TUTOR TURN, not a dialogue. v2 produced 8,000
dialogues but 20,562 tutor turns, and the loss is already masked to teacher
tokens, so a per-turn label is what credit assignment can actually use.

Two label formats come out of one bundle, because they train different heads:

  turns   each tutor turn rated on the existing 3-way leak and 3-way help
          rubric. Absolute labels, used for the leak head (binary/ordinal) and
          for calibrating the rule-based detector against human judgement.
  pairs   two completions of the SAME problem from the SAME group, side by
          side, "which one teaches better". GRPO's advantage is exactly a
          within-group comparison, so a usefulness head trained on within-group
          pairs discriminates the distinction the algorithm uses. Cross-problem
          pairs teach it an easier question it never has to answer.

Turns are stratified on the rule's own verdict so both error directions are
measurable: over-sampling the boundary is what lets you estimate the rule's
precision AND recall, which is the open question the `hint_only_leak` probe has
failed to settle for three runs (v2 ended at rule 0.281 vs probe 0.447).

Pairs deliberately draw from dialogues the rule calls CLEAN on both sides, so a
"which taught better" judgement is not silently answering "which one leaked".

LICENSING: unlike `build_review_set.py`, this bundle CAN contain state-assessment
content, because the v2 run is state-assessment content and there is no version
of this dataset without it. That is a publishing decision made deliberately -
see review_app/README. Attribution is emitted with the bundle.
"""

from __future__ import annotations

import argparse
import base64
import collections
import itertools
import json
import os
import random

import paths
import rewards

TUTOR = "Tutor:"
STUDENT = "Student:"


def split_turns(completion: str):
    """Transcript -> [(speaker, text)] in order.

    The trace stores one flat string; leak attribution and per-turn labelling
    both need the tutor's words separated from the student's.
    """
    turns, speaker, buf = [], None, []
    for line in completion.splitlines():
        if line.startswith(TUTOR) or line.startswith(STUDENT):
            if speaker is not None:
                turns.append((speaker, "\n".join(buf).strip()))
            speaker = "tutor" if line.startswith(TUTOR) else "student"
            buf = [line.split(":", 1)[1].strip()]
        elif speaker is not None:
            buf.append(line)
    if speaker is not None:
        turns.append((speaker, "\n".join(buf).strip()))
    return [(s, t) for s, t in turns if t]


def stratum(sig: dict, flagged: bool) -> str:
    """Where this turn sits relative to the rule's decision boundary."""
    if flagged:
        if sig["verbatim"] >= 1.0:
            return "flag_verbatim"
        if sig.get("identifying", 0) > 0:
            return "flag_identifying"
        return "flag_other"
    # not flagged, but close enough that a human might disagree
    if sig["overlap"] >= 0.35 or sig.get("identifying", 0) >= 0.5 or sig["elimination"] >= 0.35:
        return "borderline"
    return "clean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/20260731-212530")
    ap.add_argument("--good", default="runs/good_traces_all.jsonl",
                    help="hand-written good tier; pairs it against the policy on "
                         "the same question so the comparison spans a real quality "
                         "range instead of being mid-vs-mid")
    ap.add_argument("--items", default=str(paths.DATA / "state_tests" / "train_items.jsonl"))
    ap.add_argument("--n-turns", type=int, default=1200)
    ap.add_argument("--n-pairs", type=int, default=400)
    ap.add_argument("--n-cross-pairs", type=int, default=150,
                    help="good-vs-policy pairs, capped by how many good traces exist")
    ap.add_argument("--pairs-per-group", type=int, default=1,
                    help="comparisons drawn from each problem's group of completions")
    ap.add_argument("--cross-per-good", type=int, default=1,
                    help="how many policy rivals each hand-written trace faces")
    ap.add_argument("--n-overlap-turns", type=int, default=80)
    ap.add_argument("--n-overlap-pairs", type=int, default=40,
                    help="items every labeller sees first, in the same order. "
                         "Agreement is only measurable on items more than one "
                         "person judged, and leaving that to a random shuffle "
                         "means discovering you have no overlap after the fact")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="review_app/data/label_items.json")
    ap.add_argument("--key", default="review_app/data/label_key.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    meta = {json.loads(l)["question"]: json.loads(l) for l in open(args.items)}
    # --run takes a run directory or a flat traces file, so a generation-only job
    # (src/gen_traces.py) can feed the labelling set directly
    trace_path = args.run if os.path.isfile(args.run) else f"{args.run}/traces.jsonl"
    traces = [json.loads(l) for l in open(trace_path)]
    # the items file is the source of truth: an item removed from it (a PDF
    # extraction that came out damaged, say) must not reach a labeller through
    # the traces, which were written before the item was withdrawn
    dropped = [t for t in traces if t["prompt"] not in meta]
    traces = [t for t in traces if t["prompt"] in meta]
    for t in traces:
        t.setdefault("tier", "policy")
    print(f"{len(traces)} dialogues from {trace_path}"
          + (f" ({len(dropped)} dropped: item withdrawn)" if dropped else ""))

    # Hand-written dialogues go into the SAME turn pool. A 1-5 goodness scale
    # rated only on one mediocre policy has nothing at its top end, so labellers
    # recalibrate "5" downwards to mean "best of a bad set" - which is exactly the
    # ceiling a reward model would then learn.
    if args.good and os.path.exists(args.good):
        good_traces = [json.loads(l) for l in open(args.good)]
        good_traces = [g for g in good_traces if g["prompt"] in meta]
        for g in good_traces:
            g["tier"] = "good"
        traces += good_traces
        print(f"  + {len(good_traces)} hand-written dialogues folded in")

    # ---- expand dialogues into tutor turns, scored by the rule ----
    turns, groups = [], collections.defaultdict(list)
    for di, tr in enumerate(traces):
        gold = tr["gold"]
        distractors = [c for j, c in enumerate(tr["choices"]) if j != tr["gold_idx"]]
        parsed = split_turns(tr["completion"])
        groups[(tr["step"], tr["prompt"])].append(di)
        ctx = []
        for ti, (who, text) in enumerate(parsed):
            if who != "tutor":
                ctx.append(("student", text))
                continue
            sig = rewards.leak_signals(text, gold, distractors, question=tr["prompt"])
            flagged = bool(rewards.leaked_answer(text, gold, distractors,
                                                 question=tr["prompt"]))
            turns.append({
                "id": f"t{di}_{ti}",
                "dialogue": di,
                "turn_index": ti,
                "step": tr["step"],
                "question": tr["prompt"],
                "choices": tr["choices"],
                "gold_b64": base64.b64encode(str(gold).encode()).decode(),
                "context": [{"who": w, "text": x} for w, x in ctx],
                "tutor_turn": text,
                "subject": meta.get(tr["prompt"], {}).get("subject"),
                "grade": meta.get(tr["prompt"], {}).get("grade"),
                "_stratum": stratum(sig, flagged),
                "_tier": tr.get("tier", "policy"),
                "_rule_flagged": flagged,
                "_signals": sig,
            })
            ctx.append(("tutor", text))

    counts = collections.Counter(t["_stratum"] for t in turns)
    print(f"{len(turns)} tutor turns; strata {dict(counts)}")

    # ---- stratified sample: the boundary is worth more than the bulk ----
    quota = {"flag_verbatim": 0.20, "flag_identifying": 0.20, "flag_other": 0.10,
             "borderline": 0.25, "clean": 0.25}
    by_stratum = collections.defaultdict(list)
    for t in turns:
        by_stratum[t["_stratum"]].append(t)
    picked = [t for t in turns if t["_tier"] == "good"]
    by_stratum = {k: [x for x in v if x["_tier"] != "good"] for k, v in by_stratum.items()}
    for s, frac in quota.items():
        pool = by_stratum.get(s, [])
        want = int(args.n_turns * frac)
        picked.extend(rng.sample(pool, min(want, len(pool))))
    rng.shuffle(picked)
    print(f"sampled {len(picked)} turns; "
          f"{dict(collections.Counter(t['_stratum'] for t in picked))}")

    # ---- within-group pairs, both sides clean by the rule ----
    pairs = []
    keys = list(groups)
    rng.shuffle(keys)
    for key in keys:
        if len(pairs) >= args.n_pairs:
            break
        members = [traces[i] for i in groups[key] if not traces[i]["leaked"]]
        if len(members) < 2:
            continue
        # prefer pairs that disagreed on the outcome: the interesting question is
        # whether a human agrees with the student's verdict
        solved = [m for m in members if m["solved"]]
        unsolved = [m for m in members if not m["solved"]]
        combos = ([(x, y) for x in solved for y in unsolved]
                  or list(itertools.combinations(members, 2)))
        rng.shuffle(combos)
        for a, b in combos[: args.pairs_per_group]:
            if len(pairs) >= args.n_pairs:
                break
            if rng.random() < 0.5:        # never let position encode the outcome
                a, b = b, a
            pairs.append({
                "id": f"p{len(pairs)}",
                "step": key[0],
                "question": a["prompt"],
                "choices": a["choices"],
                "gold_b64": base64.b64encode(str(a["gold"]).encode()).decode(),
                "a": a["completion"],
                "b": b["completion"],
                "subject": meta.get(a["prompt"], {}).get("subject"),
                "grade": meta.get(a["prompt"], {}).get("grade"),
                "_a_solved": bool(a["solved"]),
                "_b_solved": bool(b["solved"]),
            })
    print(f"built {len(pairs)} within-group pairs")

    # ---- cross-tier pairs: hand-written good vs the policy, same question ----
    # Every trace from the run is one mediocre policy, so within-group pairs are
    # mid-vs-mid and a preference model trained only on them learns "least bad".
    # These pairs span an actual quality range. The tier is recorded ONLY in the
    # key - a labeller who could see which side was hand-written would be rating
    # the label, not the teaching.
    n_cross = 0
    if args.good and os.path.exists(args.good):
        good = [json.loads(l) for l in open(args.good) if json.loads(l)["prompt"] in meta]
        by_q = collections.defaultdict(list)
        for tr in traces:
            by_q[tr["prompt"]].append(tr)
        rng.shuffle(good)
        for g in good:
            if n_cross >= args.n_cross_pairs:
                break
            rivals = by_q.get(g["prompt"], [])
            if not rivals:
                continue
            for rival in rng.sample(rivals, min(args.cross_per_good, len(rivals))):
                if n_cross >= args.n_cross_pairs:
                    break
                good_is_a = rng.random() < 0.5
                a, b = (g, rival) if good_is_a else (rival, g)
                pairs.append({
                    "id": f"x{n_cross}",
                    "step": rival["step"],
                    "question": g["prompt"],
                    "choices": g["choices"],
                    "gold_b64": base64.b64encode(str(g["gold"]).encode()).decode(),
                    "a": a["completion"],
                    "b": b["completion"],
                    "subject": meta.get(g["prompt"], {}).get("subject"),
                    "grade": meta.get(g["prompt"], {}).get("grade"),
                    "_a_solved": bool(a["solved"]),
                    "_b_solved": bool(b["solved"]),
                    "_good_side": "a" if good_is_a else "b",
                })
                n_cross += 1
        print(f"built {n_cross} cross-tier pairs (hand-written vs policy)")
    else:
        print(f"no good-tier file at {args.good}; pairs are policy-only")

    public_turns = [{k: v for k, v in t.items() if not k.startswith("_")} for t in picked]
    public_pairs = [{k: v for k, v in p.items() if not k.startswith("_")} for p in pairs]

    # Items every labeller sees first, in the same order. Everything after this is
    # shuffled per person, so without a fixed prefix two people can label 200 items
    # each and share almost none - and inter-annotator agreement, which caps what
    # any reward model trained on this data can achieve, becomes unmeasurable.
    # Drawn across the strata rather than off the top so the overlap is not all
    # flagged turns.
    overlap_turns = [t["id"] for t in picked[: args.n_overlap_turns]]
    overlap_pairs = [p["id"] for p in pairs[: args.n_overlap_pairs]]

    bundle = {
        "schema": "grpo-tutor-label/v1",
        "source": args.run,
        "seed": args.seed,
        "overlap": {"turns": overlap_turns, "pairs": overlap_pairs},
        "turns": public_turns,
        "pairs": public_pairs,
        "attribution": (
            "Questions and answer options are released items from public state "
            "assessments (California CAASPP/CST, Texas STAAR, Massachusetts MCAS, "
            "New Jersey NJSLA). Rights remain with the issuing state agencies. "
            "Tutor and student turns are model output."
        ),
    }
    with open(args.out, "w") as f:
        json.dump(bundle, f)
    with open(args.key, "w") as f:
        json.dump({
            "schema": "grpo-tutor-label-key/v1",
            "population": dict(counts),
            "turns": [{"id": t["id"], "stratum": t["_stratum"], "tier": t["_tier"],
                       "rule_flagged": t["_rule_flagged"], "signals": t["_signals"]}
                      for t in picked],
            "pairs": [{"id": p["id"], "a_solved": p["_a_solved"],
                       "b_solved": p["_b_solved"],
                       "good_side": p.get("_good_side")} for p in pairs],
        }, f, indent=1)
    print(f"\nwrote {args.out} and {args.key}")
    print("subject:", dict(collections.Counter(t["subject"] for t in picked)))


if __name__ == "__main__":
    main()
