"""Pull the collected labels out of Firestore into a local JSONL.

    export GOOGLE_APPLICATION_CREDENTIALS=~/keys/grpo-tutor-label.json
    python src/fetch_labels.py --out data/labels.jsonl

The `labels` collection is write-only from the browser - the rules deny client
reads on purpose - so reading it needs a service account:

    Firebase console -> Project settings -> Service accounts -> Generate new
    private key. Keep the file OUT of this repo; it is a credential.

Joins each label back to what the labeller was actually looking at, and to the
rule's own verdict from label_key.json, which is what makes the two useful:

  agreement   how often a human agrees with the leak RULE. Both error directions
              are estimable because the sample over-weights the boundary.
  training    the joined rows are the reward model's dataset.
"""

from __future__ import annotations

import argparse
import collections
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="grpo-tutor-label")
    ap.add_argument("--bundle", default="label_app/data/label_items.json")
    ap.add_argument("--key", default="label_app/data/label_key.json")
    ap.add_argument("--out", default="data/labels.jsonl")
    args = ap.parse_args()

    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise SystemExit(
            "GOOGLE_APPLICATION_CREDENTIALS is not set.\n"
            "Firebase console -> Project settings -> Service accounts -> "
            "Generate new private key, then point this at the file.")
    try:
        from google.cloud import firestore
    except ImportError:
        raise SystemExit("pip install google-cloud-firestore")

    db = firestore.Client(project=args.project)
    docs = [d.to_dict() for d in db.collection("labels").stream()]
    # connectivity checks are written straight to the live collection rather than
    # to a staging one, so that what is verified is the real path
    probes = [d for d in docs if str(d.get("who", "")).startswith("__selftest")]
    docs = [d for d in docs if not str(d.get("who", "")).startswith("__selftest")]
    print(f"{len(docs)} labels in Firestore"
          + (f" ({len(probes)} self-test rows ignored)" if probes else ""))
    if not docs:
        return

    bundle = json.load(open(args.bundle))
    key = json.load(open(args.key))
    turns = {t["id"]: t for t in bundle["turns"]}
    pairs = {p["id"]: p for p in bundle["pairs"]}
    kturn = {t["id"]: t for t in key["turns"]}
    kpair = {p["id"]: p for p in key["pairs"]}

    rows, agree, n_rule = [], 0, 0
    for d in docs:
        iid = d.get("itemId")
        row = {k: v for k, v in d.items() if k != "createdAt"}
        row["createdAt"] = str(d.get("createdAt"))
        if d.get("kind") == "turn" and iid in turns:
            row["item"] = turns[iid]
            row["rule"] = kturn.get(iid)
            if row["rule"]:
                n_rule += 1
                human_leak = d.get("leak") == "names_it"
                agree += int(human_leak == bool(row["rule"]["rule_flagged"]))
        elif d.get("kind") == "pair" and iid in pairs:
            row["item"] = pairs[iid]
            row["outcome"] = kpair.get(iid)
        rows.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    kinds = collections.Counter(r.get("kind") for r in rows)
    who = collections.Counter(r.get("who") for r in rows)
    print(f"wrote {args.out}")
    print("kinds  :", dict(kinds))
    print("people :", dict(who))
    if n_rule:
        print(f"\nhuman vs leak RULE: agree on {agree}/{n_rule} ({agree / n_rule:.1%})")
        print("  (human 'names_it' treated as a leak; 'hints_at_it' is not, which is "
              "the judgement call the rule cannot make)")


if __name__ == "__main__":
    main()
