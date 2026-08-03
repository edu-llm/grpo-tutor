"""Build the human-review item set for `review_app/` from a run's traces.

    python src/build_review_set.py

Reads `runs/from-cluster/traces.jsonl` (OpenBookQA-derived dialogues from run v0)
and writes two files:

  review_app/data/items.json         what the browser fetches - question, choices,
                                     the dialogue, and the gold answer (base64'd so
                                     it is not readable in the network tab during
                                     stage 1). NO LeakGuard verdict: reviewers must
                                     not be anchored by the label we are validating.
  review_app/data/analysis_key.json  the join key - LeakGuard's verdict and signals,
                                     the stratum, and population stratum counts for
                                     reweighting. Not fetched by the app.

LICENSING: OpenBookQA only. No state-assessment content (STAAR/PSSA/MCAS/CAASPP/
NJSLA) may enter this bundle - it is state-copyright and cannot be republished.
`--assert-clean` (on by default) fails the build if any such string appears.

SAMPLING
--------
The review set is deliberately NOT a random sample of the run. It is stratified on
LeakGuard's own decision so that both error directions are measurable:

  flag_verbatim / flag_overlap / flag_elim   flagged - measures PRECISION
  borderline                                 not flagged but close to a threshold
  clean                                      not flagged and nowhere near one
                                             (the last two measure RECALL)

Each stratum's population size is recorded in the key so estimates can be
reweighted back to the run's real distribution.
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from rewards import leak_signals  # noqa: E402

TRACES = REPO / "runs" / "from-cluster" / "traces.jsonl"
OUT_DIR = REPO / "review_app" / "data"

# One tutor turn, one student turn, one tutor turn, ... Anything else is the model
# running off the rails past its stop token; those transcripts are ambiguous to
# label and are dropped rather than shown to a human.
CANONICAL = ("Tutor", "Student", "Tutor", "Student", "Tutor")

MIN_TUTOR_CHARS = 200
MAX_TUTOR_CHARS = 1400

# stratum -> how many items to put in the review set
QUOTAS = {
    "flag_verbatim": 55,
    "flag_overlap": 40,
    "flag_elim": 35,
    "borderline": 80,
    "clean": 100,
}
# The everyone-sees-these agreement set. Served FIRST to every reviewer, so even
# somebody who does five items and stops still contributes agreement data. Small
# for that reason, and one item per stratum so it spans LeakGuard's decision
# boundary rather than sitting on one side of it.
OVERLAP_QUOTAS = {
    "flag_verbatim": 1,
    "flag_overlap": 1,
    "flag_elim": 1,
    "borderline": 1,
    "clean": 1,
}

BANNED = ("staar", "pssa", "mcas", "caaspp", "njsla", "texas education agency")


def parse_dialogue(completion: str) -> list[tuple[str, str]]:
    parts = re.split(r"^(Tutor:|Student:)", completion, flags=re.M)
    turns = []
    i = 1
    while i < len(parts):
        role = parts[i].rstrip(":")
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        turns.append((role, text))
        i += 2
    return turns


def stratum_of(sig: dict, leaked: float) -> str:
    if leaked:
        if sig["verbatim"] >= 1.0:
            return "flag_verbatim"
        if sig["overlap"] >= 0.6:
            return "flag_overlap"
        return "flag_elim"
    if sig["overlap"] >= 0.34 or sig["elimination"] >= 0.25:
        return "borderline"
    return "clean"


def stable_key(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()


def load_candidates(path: Path) -> list[dict]:
    """Rows that are safe to show a human, with LeakGuard recomputed from the
    tutor turns only (the same text LeakGuard scores in training)."""
    out = []
    for lineno, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        turns = parse_dialogue(r["completion"])
        if tuple(t[0] for t in turns) != CANONICAL:
            continue
        if any(not text for _, text in turns):
            continue

        tutor_text = " ".join(text for role, text in turns if role == "Tutor")
        if not (MIN_TUTOR_CHARS <= len(tutor_text) <= MAX_TUTOR_CHARS):
            continue

        distractors = [c for i, c in enumerate(r["choices"]) if i != r["gold_idx"]]
        sig = leak_signals(tutor_text, r["gold"], distractors)
        leaked = float(
            sig["verbatim"] >= 1.0 or sig["overlap"] >= 0.6 or sig["elimination"] >= 0.5
        )
        # Where the recomputation disagrees with what training logged, the tutor
        # text we can reconstruct is not the text that was scored. Skip: a human
        # label on it would not be comparable to LeakGuard's.
        if leaked != float(r["leaked"]):
            continue

        out.append(
            {
                "id": f"v0-{lineno:05d}",
                "question": r["prompt"],
                "choices": r["choices"],
                "gold_idx": r["gold_idx"],
                "gold": r["gold"],
                "dialogue": [{"role": role.lower(), "text": text} for role, text in turns],
                "step": r["step"],
                "solved": r["solved"],
                "reward": r["reward"],
                "sim_student_answer": r["student_answer"],
                "hint_only_leak": r["hint_only_leak"],
                "leakguard_leaked": leaked,
                "leak_signals": sig,
                "stratum": stratum_of(sig, leaked),
            }
        )
    return out


def select(cands: list[dict], seed: str) -> list[dict]:
    """Fill each stratum's quota, at most one item per distinct question, and
    alternating solved/unsolved so the 'did this help?' question sees both."""
    by_stratum = collections.defaultdict(lambda: {0.0: [], 1.0: []})
    for c in cands:
        by_stratum[c["stratum"]][c["solved"]].append(c)

    used_questions: set[str] = set()
    chosen: list[dict] = []
    # Scarcest strata first, so they are not starved of questions by the plentiful ones.
    for stratum in sorted(QUOTAS, key=lambda s: len(by_stratum[s][0.0]) + len(by_stratum[s][1.0])):
        buckets = {
            k: sorted(v, key=lambda c: stable_key(seed, c["id"]))
            for k, v in by_stratum[stratum].items()
        }
        want = QUOTAS[stratum]
        got = 0
        turn = 1.0
        while got < want and (buckets[0.0] or buckets[1.0]):
            src = buckets[turn] or buckets[1.0 - turn]
            item = src.pop(0)
            if item["question"] not in used_questions:
                used_questions.add(item["question"])
                chosen.append(item)
                got += 1
            turn = 1.0 - turn
        if got < want:
            print(f"  ! {stratum}: only {got}/{want} available", file=sys.stderr)
    return chosen


def pick_overlap(chosen: list[dict], seed: str) -> list[str]:
    """The shared set, in the order everyone sees it.

    Deliberately led by a blatant leak: it is the first item anybody reviews, and
    an unambiguous example calibrates what the three buttons mean. The order is
    identical for every reviewer, so it cannot skew agreement.
    """
    by_stratum = collections.defaultdict(list)
    for c in chosen:
        by_stratum[c["stratum"]].append(c)
    ids = []
    for stratum in ("flag_verbatim", "clean", "borderline", "flag_overlap", "flag_elim"):
        pool = sorted(by_stratum[stratum], key=lambda c: stable_key(seed, "overlap", c["id"]))
        ids += [c["id"] for c in pool[: OVERLAP_QUOTAS[stratum]]]
    return ids


def assert_no_state_tests(blobs: list[str]) -> None:
    for blob in blobs:
        low = blob.lower()
        for term in BANNED:
            if term in low:
                raise SystemExit(f"REFUSING TO BUILD: state-assessment term {term!r} in output")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", type=Path, default=TRACES)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", default="review-v1")
    args = ap.parse_args()

    print(f"reading {args.traces}")
    cands = load_candidates(args.traces)
    pop = collections.Counter(c["stratum"] for c in cands)
    print(f"  {len(cands)} labelable dialogues")
    for k, v in pop.most_common():
        print(f"    {k:<14} {v}")

    chosen = select(cands, args.seed)
    chosen.sort(key=lambda c: stable_key(args.seed, "order", c["id"]))
    overlap_ids = pick_overlap(chosen, args.seed)
    print(f"selected {len(chosen)} items ({len(overlap_ids)} of them in the shared overlap set)")

    items = [
        {
            "id": c["id"],
            "question": c["question"],
            "choices": c["choices"],
            # Base64 keeps the gold answer out of plain sight in devtools/the raw
            # JSON. It is not security - it stops an idle glance during stage 1,
            # which is the only threat that matters here.
            "gold_b64": base64.b64encode(str(c["gold_idx"]).encode()).decode(),
            "dialogue": c["dialogue"],
        }
        for c in chosen
    ]
    items_doc = {
        "schema": "grpo-tutor-review-items/v1",
        "source": "runs/from-cluster/traces.jsonl - run v0, OpenBookQA-derived. "
                  "Questions and options are OpenBookQA (openly licensed); tutor and "
                  "student turns are our own models' output. No state-assessment content.",
        "seed": args.seed,
        # Served first, in this order, to everybody. Nobody is assigned a quota:
        # the app queues these five and then walks the rest of the pool in a
        # name-seeded order for as long as the reviewer wants to keep going.
        "overlap_ids": overlap_ids,
        "items": items,
    }
    items_json = json.dumps(items_doc, indent=1, ensure_ascii=False)
    items_doc["items_version"] = hashlib.sha256(items_json.encode()).hexdigest()[:12]

    key_doc = {
        "schema": "grpo-tutor-review-key/v1",
        "items_version": items_doc["items_version"],
        "note": "Join on id. Population counts let stratified estimates be reweighted "
                "back to the run's real distribution; the review set oversamples "
                "flagged and borderline dialogues on purpose.",
        "population_stratum_counts": dict(pop),
        "population_total": len(cands),
        "review_set_stratum_counts": dict(collections.Counter(c["stratum"] for c in chosen)),
        "overlap_ids": overlap_ids,
        "key": [
            {
                "id": c["id"],
                "stratum": c["stratum"],
                "leakguard_leaked": c["leakguard_leaked"],
                "leak_verbatim": c["leak_signals"]["verbatim"],
                "leak_overlap": c["leak_signals"]["overlap"],
                "leak_elimination": c["leak_signals"]["elimination"],
                "hint_only_leak": c["hint_only_leak"],
                "gold": c["gold"],
                "gold_idx": c["gold_idx"],
                "solved": c["solved"],
                "reward": c["reward"],
                "step": c["step"],
                "sim_student_answer": c["sim_student_answer"],
            }
            for c in chosen
        ],
    }

    final_items = json.dumps(items_doc, indent=1, ensure_ascii=False)
    final_key = json.dumps(key_doc, indent=1, ensure_ascii=False)
    assert_no_state_tests([final_items, final_key])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "items.json").write_text(final_items)
    (args.out_dir / "analysis_key.json").write_text(final_key)
    print(f"wrote {args.out_dir/'items.json'} ({len(final_items)/1024:.0f} KB, "
          f"version {items_doc['items_version']})")
    print(f"wrote {args.out_dir/'analysis_key.json'}")

    solved = sum(1 for c in chosen if c["solved"])
    print(f"  solved after tutoring: {solved}/{len(chosen)}")
    print(f"  flagged by LeakGuard:  {sum(1 for c in chosen if c['leakguard_leaked'])}/{len(chosen)}")


if __name__ == "__main__":
    main()
