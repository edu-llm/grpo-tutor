"""Recompute the four leak signals on the labelled turns and check the stored verdict.

Measurement only: never imports or touches training code.
Run with PYTHONPATH=src.
"""

import base64
import json
import os
import sys

import rewards

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    rm = json.load(open(os.path.join(ROOT, "data/rm_dataset.json")))["rows"]
    items = json.load(open(os.path.join(ROOT, "label_app/data/label_items.json")))["turns"]
    by_id = {t["id"]: t for t in items}
    rows = []
    for r in rm:
        t = by_id[r["id"]]
        gold = base64.b64decode(t["gold_b64"]).decode("utf-8")
        choices = list(t.get("choices") or [])
        distractors = [c for c in choices if c.strip().lower() != gold.strip().lower()]
        sig = rewards.leak_signals(t["tutor_turn"], gold, distractors, t.get("question", ""))
        rows.append({
            "id": r["id"],
            "leak": r["leak"],
            "n_raters": r["n_raters"],
            "rule_flagged": bool(r["rule_flagged"]),
            "subject": r.get("subject"),
            "grade": r.get("grade"),
            "question": t.get("question", ""),
            "gold": gold,
            "choices": choices,
            "distractors": distractors,
            "tutor_turn": t["tutor_turn"],
            "verbatim": sig["verbatim"],
            "overlap": sig["overlap"],
            "elimination": sig["elimination"],
            "identifying": sig["identifying"],
            "identifying_hits": sig["identifying_hits"],
            "identifying_n": sig["identifying_n"],
            # the four-signal rule this calibration was run against, so the
            # numbers in docs/leak_calibration.md stay reproducible now that
            # overlap and elimination are off by default in the reward
            "recomputed": rewards.leaked_answer(
                t["tutor_turn"], gold, distractors, question=t.get("question", ""),
                use_overlap=True, use_elimination=True,
            ),
        })
    return rows


if __name__ == "__main__":
    rows = load()
    print("n rows:", len(rows))
    n_choices_missing = sum(1 for r in rows if not r["choices"])
    print("rows with no choices:", n_choices_missing)
    print("rows where gold not in choices:",
          sum(1 for r in rows if r["choices"] and len(r["distractors"]) == len(r["choices"])))
    mism = [r for r in rows if bool(r["recomputed"]) != r["rule_flagged"]]
    print("mismatches vs stored rule_flagged:", len(mism), f"({len(mism)/len(rows):.4f})")
    for r in mism[:20]:
        print(" ", r["id"], "stored", r["rule_flagged"], "recomputed", r["recomputed"],
              {k: round(r[k], 3) for k in ("verbatim", "overlap", "elimination", "identifying_hits")})
