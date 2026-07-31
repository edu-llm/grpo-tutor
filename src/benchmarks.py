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
`hint` is an oracle fact - dataset-provided where one exists, derived where the
corpus ships a passage instead. It is a ceiling reference, never shown to the
teacher.

AN ORACLE HINT THAT CONTAINS THE ANSWER IS NOT AN ORACLE HINT. It defines a
ceiling of "the student can copy", which no honest tutor can reach, and the ZPD
screen in `zpd_filter.py` actively selects for it - measured on the live 549-item
set, 28.2% of its hints trip the leak rule against 10.1% in the OpenBookQA pool
it was drawn from. The `*_honest` entries below drop those items at load time.
Run `python src/hint_audit.py --candidates` for the full table, and see
docs/dataset_choice.md for how each corpus was chosen or rejected.
"""

from __future__ import annotations

import argparse
import json
import random
import re

import paths
import rewards


def _leaks(item) -> bool:
    """Does this item's oracle hint already give its answer away?

    Same rules the training reward uses (`rewards.leaked_answer`), so a pool
    screened with this cannot contain an item whose oracle ceiling the LeakGuard
    would have punished the teacher for reaching.
    """
    hint = (item.get("hint") or "").strip()
    if not hint:
        return False
    gold = item["choices"][item["gold_idx"]]
    distractors = [c for i, c in enumerate(item["choices"]) if i != item["gold_idx"]]
    return bool(rewards.leaked_answer(hint, gold, distractors))


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


def _qasc(split, hint_field="combinedfact", honest_only=False):
    """QASC items are built from TWO facts, plus `combinedfact` - the two already
    composed into one sentence. Which of those is the oracle hint decides whether
    the ceiling it defines is reachable by a tutor that never leaks.

    Measured over all 8,134 train items with `rewards.leak_signals`
    (`hint_audit.py` reproduces it), fraction stating the gold option VERBATIM /
    tripping the leak rule:

        combinedfact  88.5% / 96.7%     fact2  49.6% / 61.5%
        fact1         30.6% / 37.2%     OpenBookQA fact1  5.8% / 10.1%

    So `combinedfact` is not an oracle hint, it is the answer: the QASC
    0.253 -> 0.893 ceiling is mostly the student copying, and no honest tutor can
    reach it. `fact1` with `honest_only` keeps 5,112 items whose ceiling an honest
    tutor could in principle reach.

    QASC is NOT the recommended pool despite that, because its answers are the
    shortest of any candidate - 60% are a single word against OpenBookQA's 31% -
    which is the structure run v0 blamed for specificity ~0. See
    docs/dataset_choice.md.
    """
    def load(limit):
        from datasets import load_dataset

        ds = load_dataset("allenai/qasc", split=split)
        out = []
        for ex in ds:
            item = _from_labeled(ex)
            if item:
                out.append({**item, "hint": ex.get(hint_field) or None})
        return [it for it in out if not (honest_only and _leaks(it))]
    return load


def _openbookqa(split, honest_only=False):
    """`fact1` is the science fact the item turns on - OpenBookQA's oracle hint.

    The cleanest oracle hint of every corpus surveyed: it trips the leak rule on
    10.1% of train items, against 37.2% for QASC's `fact1`, 96.4% for SciQ's
    `support` and 93.6% for ECQA's explanation. It reads as a general principle
    the student then has to APPLY - "as distance to an object increases, that
    object will appear smaller" for gold "the mountains seem smaller than in
    photographs" - which is a scaffolding move that is not a reveal.

    `honest_only` drops the remaining 10.1% and leaves 4,454 train items.
    """
    def load(limit):
        from datasets import load_dataset

        ds = load_dataset("allenai/openbookqa", "additional", split=split)
        out = []
        for ex in ds:
            item = _from_labeled(ex, q_key="question_stem")
            if item:
                out.append({**item, "hint": ex.get("fact1") or None})
        return [it for it in out if not (honest_only and _leaks(it))]
    return load


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _mask_gold(text: str, gold: str) -> str:
    """Blank the gold answer out of a passage sentence, content word by content
    word, so quoting the passage cannot become a reveal."""
    out = re.sub(re.escape(gold), "___", text, flags=re.IGNORECASE)
    for word in sorted(rewards._content(gold), key=len, reverse=True):
        if len(word) >= 4:
            out = re.sub(rf"\b{re.escape(word)}\w*", "___", out, flags=re.IGNORECASE)
    return re.sub(r"(___[\s,]*)+", "___ ", out).strip()


def _locator(passage: str, question: str, gold: str) -> str:
    """Derive an oracle hint for a passage-grounded item: the passage sentence
    that best matches the QUESTION, with the gold answer masked out.

    Honest BY CONSTRUCTION, in two independent ways. The sentence is chosen by
    overlap with the question and never with the options, so the answer plays no
    part in selecting it; and whatever survives selection is then masked. Trips
    the leak rule on 3.3% of RACE-middle, against 77.9% for handing over the
    whole article.

    It is also the scaffolding move the science corpora cannot offer. The student
    already HAS the passage, so "reread the line where the writer describes the
    weather" adds no information it did not have - it directs attention, which is
    teaching that is definitionally not telling.
    """
    q_words = rewards._content(question)
    best, best_score = "", -1
    for sentence in _SENTENCE.split(passage):
        sentence = sentence.strip()
        if len(sentence.split()) < 4:
            continue
        score = len(q_words & rewards._content(sentence))
        if score > best_score:
            best, best_score = sentence, score
    return _mask_gold(best, gold)


def _race(cfg, split):
    """RACE: English exam reading comprehension, 4-way, passage supplied.

    Middle-school RACE is the structural opposite of OpenBookQA on the axis run
    v0 blamed: gold answers run a median of 4 words and only 19% are a single
    word (QASC 60%, OpenBookQA 31%), so naming the concept and naming the option
    stay distinct acts. 24,587 honest train items, ~10x any science pool.
    """
    def load(limit):
        from datasets import load_dataset

        ds = load_dataset("ehovy/race", cfg, split=split)
        out = []
        for ex in ds:
            gold_idx = "ABCD".find(ex["answer"])
            if gold_idx < 0 or len(ex["options"]) != 4:
                continue
            article = ex["article"].replace("\n", " ").strip()
            gold = ex["options"][gold_idx]
            hint = _locator(article, ex["question"], gold)
            out.append({"question": f"{article}\n\n{ex['question']}",
                        "choices": list(ex["options"]), "gold_idx": gold_idx,
                        "hint": hint or None})
        return [it for it in out if it["hint"] and not _leaks(it)]
    return load


def _staar(subjects=None):
    """Released Texas STAAR exam items, grades 3-8, four subjects.

    NOT ON THE HUB AND NOT IN THIS REPO. The PDFs are TEA copyright and say so
    ("Reproduction of all or portions of this work is prohibited without
    express written permission"), this repository is public, so `data/staar/`
    is gitignored and the file has to be rebuilt locally:

        python src/staar_extract.py --download

    What makes it different from everything else in this table is that the
    questions are real exam items with human-written distractors, and that it
    covers social studies and maths, which no other registered set does.

    What it does NOT have is an oracle hint, so `hint` is None for every item.
    That is not a gap to be filled in later with something convenient - it
    means STAAR cannot go through `zpd_filter.py` as it stands, because the
    screen is defined as "fails alone AND solves with the oracle hint" and the
    second half has nothing to evaluate. Use it as an eval set, where no hint
    is needed and its unfiltered baseline is exactly the point. See
    docs/dataset_choice.md for the two ways to get a hint if one is wanted.
    """
    def load(limit):
        path = paths.DATA / "staar" / "staar_items.jsonl"
        if not path.exists():
            raise SystemExit(
                f"{path} is missing. STAAR content is TEA copyright and is not "
                f"committed to this public repo - rebuild it with\n"
                f"    python src/staar_extract.py --download\n"
                f"(needs pdfplumber: python -m venv /tmp/pdfenv && "
                f"/tmp/pdfenv/bin/python -m pip install pdfplumber)")
        with open(path) as f:
            items = [json.loads(line) for line in f if line.strip()]
        if subjects:
            items = [it for it in items if it["subject"] in subjects]
        return [{"question": it["question"], "choices": it["choices"],
                 "gold_idx": it["gold_idx"], "hint": None} for it in items]
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
def _state_file(name: str, label: str):
    """Load an extracted state-assessment file.

    These are NOT in git - the items are state-copyright and this repo is public
    (see docs/state_test_sources.md). Rebuild them with the extractor rather than
    expecting them to be present.
    """
    def load(limit):
        p = paths.DATA / "state_tests" / f"{name}_items.jsonl"
        legacy = paths.DATA / "staar" / "staar_items.jsonl"
        if not p.exists() and name == "tx" and legacy.exists():
            p = legacy
        if not p.exists():
            raise SystemExit(
                f"{p} not found. State-test items are gitignored (state copyright); "
                f"rebuild with the matching src/extract_*.py - see "
                f"docs/state_test_sources.md")
        return [json.loads(l) for l in open(p) if l.strip()]
    return load, label


REGISTRY = {
    "pa": _state_file("pa", "Pennsylvania PSSA - the eval set; carries p-values"),
    "ca": _state_file("ca", "California CST released questions"),
    "tx": _state_file("tx", "Texas STAAR released questions"),
    "ma": _state_file("ma", "Massachusetts MCAS released items"),
    "nj": _state_file("nj", "New Jersey released items"),
    "arc_easy":      (_arc("ARC-Easy"),      "grade-school science, 4-way (closest sibling of the training set)"),
    "arc_challenge": (_arc("ARC-Challenge"), "harder grade-school science, 4-way"),
    "sciq":          (_sciq,                 "science, 4-way - support passage LEAKS the answer 96% of the time"),
    "qasc":          (_qasc("validation"),   "science, 8-way - combinedfact LEAKS the answer 97% of the time"),
    "qasc_train":    (_qasc("train"),        "QASC train, combinedfact hint - CONTAMINATED, see _qasc"),
    "qasc_honest":   (_qasc("validation", "fact1", honest_only=True),
                      "QASC validation, fact1 hint, leaky items dropped"),
    "qasc_train_honest": (_qasc("train", "fact1", honest_only=True),
                      "QASC train, fact1 hint, leaky items dropped (5,112) - curation pool"),
    "obqa_test":     (_openbookqa("test"),   "OpenBookQA test split: same corpus, unfiltered (control)"),
    "obqa_train":    (_openbookqa("train"),  "OpenBookQA train split: the original ZPD curation pool"),
    "obqa_honest":   (_openbookqa("test", honest_only=True),
                      "OpenBookQA test, leaky items dropped - honest oracle ceiling"),
    "obqa_train_honest": (_openbookqa("train", honest_only=True),
                      "OpenBookQA train, leaky items dropped (4,454) - the DEFAULT curation pool"),
    "race_middle":   (_race("middle", "validation"),
                      "middle-school reading comprehension, 4-way, derived locator hint"),
    "race_middle_train": (_race("middle", "train"),
                      "RACE-middle train (24,587): phrasal answers, ~10x any science pool"),
    "staar":         (_staar(),
                      "released Texas STAAR exam items, grades 3-8, 4-way, NO hint - local file, see _staar"),
    "staar_math":    (_staar({"math"}),      "STAAR maths only - the only maths set registered"),
    "staar_science": (_staar({"science"}),   "STAAR science, grades 5 and 8"),
    "staar_social":  (_staar({"social_studies"}), "STAAR social studies, grade 8"),
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
