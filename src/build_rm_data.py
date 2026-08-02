"""Assemble the reward-model training set from every rating collected.

    python src/build_rm_data.py --out data/rm_dataset.json

Sources, in order of authority:
  Firestore `labels`   human ratings from the app
  data/agent_labels.json    the lead agent's 153
  data/label_slices/out_*.json   six independent raters, one rubric

An item rated by more than one rater gets the MEAN of their scores, not a vote.
These are ordinal judgements of degree, so two raters saying 3 and 5 genuinely
means 4 - taking a majority would throw away the disagreement instead of
representing it, and disagreement is real signal about how clear-cut a turn is.
`n_raters` is kept so training can weight confident items higher if wanted.

The text handed to the model is what a rater saw: question, correct answer, the
conversation before, and the turn being judged. Gold is included deliberately -
the leak head cannot judge leakage without knowing the answer, and gold is
available at training time anyway.
"""

from __future__ import annotations

import argparse
import base64
import collections
import glob
import json
import statistics as st
import sys
import urllib.request

sys.path.insert(0, "src")


def firestore_labels():
    try:
        from peek_labels import access_token, plain
    except ImportError:
        return {}
    try:
        tok = access_token()
    except SystemExit:
        return {}
    url = ("https://firestore.googleapis.com/v1/projects/grpo-tutor-label"
           "/databases/(default)/documents/labels?pageSize=300")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    out = collections.defaultdict(list)
    with urllib.request.urlopen(req, timeout=60) as r:
        for doc in json.load(r).get("documents", []):
            f = {k: plain(v) for k, v in doc.get("fields", {}).items()}
            if (f.get("kind") == "turn" and isinstance(f.get("leak"), int)
                    and isinstance(f.get("goodness"), int)):
                out[f["itemId"]].append(
                    {"leak": f["leak"], "goodness": f["goodness"],
                     "rater": f"human:{f.get('who')}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="label_app/data/label_items.json")
    ap.add_argument("--key", default="label_app/data/label_key.json")
    ap.add_argument("--out", default="data/rm_dataset.json")
    args = ap.parse_args()

    ratings = collections.defaultdict(list)
    for iid, rs in firestore_labels().items():
        ratings[iid].extend(rs)
    n_human = sum(len(v) for v in ratings.values())

    for iid, v in json.load(open("data/agent_labels.json"))["labels"].items():
        ratings[iid].append({"leak": v["leak"], "goodness": v["goodness"],
                             "rater": "agent:lead"})
    for f in sorted(glob.glob("data/label_slices/out_*.json")):
        who = "agent:" + f.split("_")[-1].split(".")[0]
        for iid, v in json.load(open(f)).items():
            ratings[iid].append({"leak": v["leak"], "goodness": v["goodness"],
                                 "rater": who})

    bundle = json.load(open(args.bundle))
    key = {t["id"]: t for t in json.load(open(args.key))["turns"]}
    byid = {t["id"]: t for t in bundle["turns"]}

    rows, skipped = [], 0
    for iid, rs in ratings.items():
        item = byid.get(iid)
        if not item:
            skipped += 1          # rated against an earlier bundle
            continue
        gold = base64.b64decode(item["gold_b64"]).decode()
        ctx = "\n".join(f"{c['who'].capitalize()}: {c['text']}" for c in item["context"])
        text = (f"Question: {item['question']}\n"
                f"Correct answer: {gold}\n"
                f"Conversation so far:\n{ctx or '(none)'}\n"
                f"Tutor message to rate: {item['tutor_turn']}")
        rows.append({
            "id": iid,
            "text": text,
            "leak": st.fmean(r["leak"] for r in rs),
            "goodness": st.fmean(r["goodness"] for r in rs),
            "n_raters": len(rs),
            "raters": sorted(r["rater"] for r in rs),
            "leak_spread": (max(r["leak"] for r in rs) - min(r["leak"] for r in rs)),
            "goodness_spread": (max(r["goodness"] for r in rs)
                                - min(r["goodness"] for r in rs)),
            "tier": key.get(iid, {}).get("tier"),
            "rule_flagged": key.get(iid, {}).get("rule_flagged"),
            "subject": item.get("subject"),
            "grade": item.get("grade"),
        })

    multi = [r for r in rows if r["n_raters"] > 1]
    print(f"{len(rows)} items ({skipped} rated against an older bundle, dropped)")
    print(f"  {n_human} human ratings, {sum(r['n_raters'] for r in rows) - n_human} agent")
    print(f"  {len(multi)} items have more than one rater")
    if multi:
        print(f"  mean spread where they overlap: leak "
              f"{st.fmean(r['leak_spread'] for r in multi):.2f}, goodness "
              f"{st.fmean(r['goodness_spread'] for r in multi):.2f}")
    for t in ("good", "policy"):
        sub = [r for r in rows if r["tier"] == t]
        if sub:
            print(f"  {t:7s} n={len(sub):4d}  mean goodness "
                  f"{st.fmean(r['goodness'] for r in sub):.2f}  mean leak "
                  f"{st.fmean(r['leak'] for r in sub):.2f}")

    json.dump({"schema": "grpo-tutor-rm/v1", "rows": rows}, open(args.out, "w"))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
