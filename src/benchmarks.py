"""External multiple-choice benchmarks for held-out evaluation.

WHY THIS EXISTS
---------------
The training set is ZPD-filtered: it only keeps problems the student already
fails alone. Evaluating on a held-out slice of it means `baseline_acc` is 0.0
by construction, so `teaching_gain` collapses to `teacher_acc` and measures
nothing independent.

These sets are NOT filtered, so the student starts with real, non-zero accuracy
and a hint has to actually move it. Most also come from different corpora than
the training pool, which is the only way to tell whether the teacher learned to
teach or learned this dataset. Where they share a corpus (`qasc` vs the
`qasc_train` curation pool) they are at least different SPLITS - see REGISTRY.

Everything is normalized to the project schema:
    {question: str, choices: list[str], gold_idx: int, hint: str|None, source: str}
`hint` is a dataset-provided oracle fact where one exists (used as a ceiling
reference, never shown to the teacher).
"""

from __future__ import annotations

import argparse
import json
import random

import paths


def _from_labeled(ex, q_key="question"):
    """ARC / QASC / OpenBookQA / CommonsenseQA share a {text,label} choices dict."""
    labels = list(ex["choices"]["label"])
    texts = list(ex["choices"]["text"])
    key = ex["answerKey"]
    if key not in labels:            # a few items use numeric keys against letter labels
        return None
    return {"question": ex[q_key], "choices": texts, "gold_idx": labels.index(key)}


def _arc(cfg):
    def load(limit):
        from datasets import load_dataset

        ds = load_dataset("allenai/ai2_arc", cfg, split="validation")
        out = []
        for ex in ds:
            item = _from_labeled(ex)
            if item:
                out.append({**item, "hint": None})
        return out
    return load


def _sciq(limit):
    from datasets import load_dataset

    ds = load_dataset("allenai/sciq", split="validation")
    out = []
    for ex in ds:
        choices = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        # gold is always index 0 before shuffling; shuffle per-item so position
        # carries no signal (a student could otherwise learn "always pick A")
        order = list(range(4))
        random.Random(hash(ex["question"]) & 0xFFFF).shuffle(order)
        shuffled = [choices[i] for i in order]
        out.append({"question": ex["question"], "choices": shuffled,
                    "gold_idx": order.index(0), "hint": (ex["support"] or "").strip() or None})
    return out


def _qasc(split):
    """QASC ships `combinedfact`: the two facts the item needs, already composed
    into one sentence. That is the oracle hint.

    CAVEAT, measured over all 8,134 train items with `rewards.leak_signals`:
    `combinedfact` states the gold option VERBATIM in 88.5% of items and trips the
    leak rule in 96.7% (OpenBookQA's `fact1`: 5.8% / 10.1%). So the QASC oracle
    ceiling is largely "student copies the answer out of the hint", and a tutor
    that never leaks cannot reach it. `fact1` alone trips the rule on 37.2% and is
    the closer analogue of a legitimate scaffold if that ceiling ever needs to be
    an honest one.
    """
    def load(limit):
        from datasets import load_dataset

        ds = load_dataset("allenai/qasc", split=split)
        out = []
        for ex in ds:
            item = _from_labeled(ex)
            if item:
                out.append({**item, "hint": ex.get("combinedfact") or None})
        return out
    return load


def _openbookqa(split):
    """`fact1` is the science fact the item turns on - OpenBookQA's oracle hint."""
    def load(limit):
        from datasets import load_dataset

        ds = load_dataset("allenai/openbookqa", "additional", split=split)
        out = []
        for ex in ds:
            item = _from_labeled(ex, q_key="question_stem")
            if item:
                out.append({**item, "hint": ex.get("fact1") or None})
        return out
    return load


def _commonsense_qa(limit):
    from datasets import load_dataset

    ds = load_dataset("tau/commonsense_qa", split="validation")
    out = []
    for ex in ds:
        item = _from_labeled(ex)
        if item:
            out.append({**item, "hint": None})
    return out


def _mmlu(subject):
    def load(limit):
        from datasets import load_dataset

        ds = load_dataset("cais/mmlu", subject, split="test")
        return [{"question": ex["question"], "choices": list(ex["choices"]),
                 "gold_idx": int(ex["answer"]), "hint": None} for ex in ds]
    return load


# name -> (loader, one-line description)
#
# The `*_train` entries are the CURATION pools `zpd_filter.py` draws training
# problems from; the plain names are eval sets. They are deliberately different
# splits of the same corpus: curating training items out of the rows that
# `--eval-benchmark qasc` later scores would put the eval inside the training set.
REGISTRY = {
    "arc_easy":      (_arc("ARC-Easy"),      "grade-school science, 4-way (closest sibling of the training set)"),
    "arc_challenge": (_arc("ARC-Challenge"), "harder grade-school science, 4-way"),
    "sciq":          (_sciq,                 "science, 4-way, ships a support passage as an oracle hint"),
    "qasc":          (_qasc("validation"),   "science, 8-way, needs two facts composed - good for multi-turn"),
    "qasc_train":    (_qasc("train"),        "QASC train split: the default ZPD curation pool (8k items)"),
    "obqa_test":     (_openbookqa("test"),   "OpenBookQA test split: same corpus, unfiltered (control)"),
    "obqa_train":    (_openbookqa("train"),  "OpenBookQA train split: the original ZPD curation pool"),
    "commonsense":   (_commonsense_qa,       "commonsense reasoning, 5-way, non-science"),
    "geography":     (_mmlu("high_school_geography"), "MMLU social studies, 4-way"),
    "us_history":    (_mmlu("high_school_us_history"), "MMLU social studies, 4-way"),
}


def load_benchmark(name: str, limit: int | None = None, seed: int = 0):
    if name not in REGISTRY:
        raise SystemExit(f"unknown benchmark {name!r}; choose from {sorted(REGISTRY)}")
    items = REGISTRY[name][0](limit)
    for it in items:
        it["source"] = name
    if limit and len(items) > limit:
        idx = list(range(len(items)))
        random.Random(seed).shuffle(idx)
        items = [items[i] for i in sorted(idx[:limit])]
    return items


def main():
    ap = argparse.ArgumentParser(description="Fetch an external MC benchmark as JSONL.")
    ap.add_argument("--list", action="store_true", help="show available benchmarks")
    ap.add_argument("--name", help="benchmark to dump")
    ap.add_argument("--n", type=int, default=200, help="how many items")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.list or not args.name:
        width = max(len(k) for k in REGISTRY)
        for k, (_, desc) in REGISTRY.items():
            print(f"  {k:<{width}}  {desc}")
        return

    items = load_benchmark(args.name, limit=args.n, seed=args.seed)
    out = args.out or str(paths.DATA / f"bench_{args.name}.jsonl")
    with open(out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    n_hint = sum(1 for it in items if it.get("hint"))
    sizes = sorted({len(it["choices"]) for it in items})
    print(f"wrote {len(items)} items -> {out}")
    print(f"  choices per item: {sizes} | items with an oracle hint: {n_hint}")


if __name__ == "__main__":
    main()
