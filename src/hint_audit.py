"""Score a corpus's suitability as a TUTORING task - with no model and no GPU.

WHY THIS EXISTS
---------------
Run v0 measured specificity ~0: a hint written for a different problem helped the
student as much as the right one. The diagnosis was structural, not a tuning
problem - OpenBookQA answers are single words, so "correct the misconception" and
"reveal the answer" are the same sentence and no scaffolding move exists that is
not leaking. Picking a replacement corpus by oracle-hint headroom alone repeats
the QASC mistake: QASC's `combinedfact` states the gold option verbatim in 88.5%
of items, so most of its +0.64 headroom is the student COPYING the answer out of
the hint, and a tutor that never leaks can never reach it.

Two properties decide this and both are pure string manipulation:

  LEAK    what fraction of oracle hints already contain the answer
          (`rewards.leak_signals`, the same rules the training reward uses)
  SHAPE   how long the gold answer is, how many options, whether the answer is
          already sitting in the question stem

Everything here runs on a laptop in seconds. Baseline accuracy, oracle headroom
and ZPD keep rate need the real 0.5B student and belong in `bench_baseline.py` /
`zpd_filter.py` on the GPU - see docs/dataset_choice.md.

    python src/hint_audit.py                  # every registered benchmark
    python src/hint_audit.py --name qasc_train_f1
    python src/hint_audit.py --candidates     # + corpora considered and rejected
"""

from __future__ import annotations

import argparse
import json
import statistics

import benchmarks
import rewards


def _shape(items: list[dict]) -> dict:
    """Structural facts that decide whether a tutor has a move that is not leaking."""
    gold_words, single, in_stem = [], 0, 0
    for it in items:
        gold = it["choices"][it["gold_idx"]]
        w = len(str(gold).split())
        gold_words.append(w)
        single += w == 1
        # the answer already sitting in the prompt (common in passage-grounded
        # sets) means a hint that quotes the passage cannot help but leak
        in_stem += rewards._norm(gold) in rewards._norm(it["question"])
    n = len(items)
    return {"gold_words_median": statistics.median(gold_words),
            "single_word_gold": single / n,
            "gold_in_stem": in_stem / n}


def _leak(items: list[dict]) -> dict:
    """Fraction of ORACLE hints that already give the answer away.

    Items with no hint are counted as non-leaking but tracked separately: a
    corpus with no oracle hint field cannot be ZPD-screened at all, which is a
    different failure from one whose hints happen to be clean.
    """
    n = len(items)
    hinted = [it for it in items if (it.get("hint") or "").strip()]
    verb = trip = elim = 0
    overlaps, hint_words = [], []
    for it in hinted:
        gold = it["choices"][it["gold_idx"]]
        distractors = [c for j, c in enumerate(it["choices"]) if j != it["gold_idx"]]
        sig = rewards.leak_signals(it["hint"], gold, distractors)
        verb += sig["verbatim"] >= 1.0
        elim += sig["elimination"] >= 0.5
        overlaps.append(sig["overlap"])
        hint_words.append(len(it["hint"].split()))
        trip += rewards.leaked_answer(it["hint"], gold, distractors)
    h = len(hinted)
    return {"has_hint": h / n,
            "hint_words_median": statistics.median(hint_words or [0]),
            "verbatim": verb / h if h else 0.0,
            "elimination": elim / h if h else 0.0,
            "mean_overlap": statistics.mean(overlaps) if overlaps else 0.0,
            "trips_leak_rule": trip / h if h else 0.0,
            "honest_items": h - trip}


def audit(name: str, items: list[dict]) -> dict:
    n = len(items)
    sizes = sorted({len(it["choices"]) for it in items})
    row = {"name": name, "n": n, "n_choices": sizes,
           "chance": round(sum(1 / len(it["choices"]) for it in items) / n, 3)}
    row.update(_shape(items))
    row.update(_leak(items))
    return row


HEADER = (f"{'corpus':<30}{'n':>7}{'chance':>8}{'gold_w':>8}{'1word':>7}"
          f"{'hint':>6}{'verb':>7}{'elim':>7}{'trips':>7}{'honest':>8}")


def fmt(r: dict) -> str:
    return (f"{r['name']:<30}{r['n']:>7}{r['chance']:>8.3f}"
            f"{r['gold_words_median']:>8.1f}{r['single_word_gold']:>7.2f}"
            f"{r['has_hint']:>6.2f}{r['verbatim']:>7.3f}{r['elimination']:>7.3f}"
            f"{r['trips_leak_rule']:>7.3f}{r['honest_items']:>8}")


# Corpora that were surveyed for docs/dataset_choice.md and NOT adopted. They
# stay here rather than in benchmarks.REGISTRY so the registry keeps meaning
# "sets we actually train or evaluate on", while the rejection evidence stays
# reproducible with one command.
def _rejected():
    from datasets import load_dataset

    out = {}

    ds = load_dataset("allenai/sciq", split="train")
    out["sciq_train/support"] = [
        {"question": ex["question"],
         "choices": [ex["correct_answer"], ex["distractor1"], ex["distractor2"],
                     ex["distractor3"]],
         "gold_idx": 0, "hint": (ex["support"] or "").strip()}
        for ex in ds if (ex["support"] or "").strip()]

    ds = load_dataset("yangdong/ecqa", split="train")
    ecqa = []
    for ex in ds:
        ops = [ex[f"q_op{i}"] for i in range(1, 6)]
        if ex["q_ans"] in ops:
            ecqa.append({"question": ex["q_text"], "choices": ops,
                         "gold_idx": ops.index(ex["q_ans"]), "hint": ex["taskA_pos"]})
    out["ecqa_train/taskA_pos"] = ecqa

    from datasets import Image
    ds = load_dataset("derek-thomas/ScienceQA", split="train").cast_column(
        "image", Image(decode=False))
    sqa = []
    for ex in ds:
        if ex["image"] is not None or not ex["lecture"]:
            continue
        q = (ex["hint"].strip() + "\n" + ex["question"]) if ex["hint"] else ex["question"]
        sqa.append({"question": q, "choices": list(ex["choices"]),
                    "gold_idx": int(ex["answer"]), "hint": ex["lecture"].strip()})
    out["scienceqa_train/lecture"] = sqa
    out["scienceqa_train/lecture-4way"] = [s for s in sqa if len(s["choices"]) == 4]

    ds = load_dataset("allenai/quartz", split="train")
    out["quartz_train/para"] = [
        {"question": ex["question"], "choices": list(ex["choices"]["text"]),
         "gold_idx": list(ex["choices"]["label"]).index(ex["answerKey"]),
         "hint": ex["para"]}
        for ex in ds if ex["answerKey"] in list(ex["choices"]["label"])]

    ds = load_dataset("ehovy/race", "middle", split="train")
    out["race_middle/article"] = [
        {"question": f"{ex['article']}\n\n{ex['question']}", "choices": list(ex["options"]),
         "gold_idx": "ABCD".find(ex["answer"]), "hint": ex["article"]}
        for ex in ds if "ABCD".find(ex["answer"]) >= 0 and len(ex["options"]) == 4]

    ds = load_dataset("dataset-org/dream", split="train", revision="refs/convert/parquet")
    out["dream_train/dialogue"] = [
        {"question": " ".join(ex["dialogue"]) + "\n\n" + ex["question"],
         "choices": list(ex["choice"]), "gold_idx": ex["choice"].index(ex["answer"]),
         "hint": " ".join(ex["dialogue"])}
        for ex in ds if ex["answer"] in ex["choice"]]

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", nargs="*", default=None,
                    help="registry entries to audit (default: all of them)")
    ap.add_argument("--limit", type=int, default=None, help="cap items per corpus")
    ap.add_argument("--candidates", action="store_true",
                    help="also audit the corpora considered and rejected")
    ap.add_argument("--out", default=None, help="write the rows as JSON")
    args = ap.parse_args()

    pools = {}
    for name in (args.name or sorted(benchmarks.REGISTRY)):
        pools[name] = benchmarks.load_benchmark(name, limit=args.limit)
    if args.candidates:
        pools.update(_rejected())

    print(HEADER)
    rows = []
    for name, items in pools.items():
        if not items:
            continue
        row = audit(name, items)
        rows.append(row)
        print(fmt(row), flush=True)

    print("\ngold_w = median words in the gold answer · 1word = fraction of "
          "single-word answers\ntrips = fraction of oracle hints the leak rule "
          "fires on · honest = items left after dropping those")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
