"""Delete label documents that predate the current schema.

    python src/clean_labels.py            # dry run, lists what would go
    python src/clean_labels.py --apply    # actually delete

The rating scales changed from string enums ('no' / 'too_vague') to numbers
(leak 1-3, goodness 1-5), and the pairwise comparisons were dropped. Documents in
the old shape cannot be joined to the new ones, and a mixed collection silently
poisons any aggregate computed over it - so they go, rather than being carried
along and filtered forever after.

Connectivity self-tests (who = __selftest) go too. Deletion is by document, never
by collection: `firestore:delete labels` would take the real labels with it.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from peek_labels import access_token, plain

PROJECT = "grpo-tutor-label"
BASE = (f"https://firestore.googleapis.com/v1/projects/{PROJECT}"
        "/databases/(default)/documents/labels")


def verdict(f: dict) -> str | None:
    """Why this document should go, or None to keep it."""
    if str(f.get("who", "")).startswith("__selftest"):
        return "self-test"
    if f.get("kind") == "pair":
        return "pairwise (comparisons dropped)"
    if not isinstance(f.get("leak"), int):
        return f"old leak format ({f.get('leak')!r})"
    if not isinstance(f.get("goodness"), int):
        return "no goodness rating (pre-1-5 scale)"
    if not (1 <= f["leak"] <= 3 and 1 <= f["goodness"] <= 5):
        return "rating out of range"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="delete for real")
    args = ap.parse_args()

    tok = access_token()
    hdr = {"Authorization": f"Bearer {tok}"}

    docs, page = [], None
    while True:
        url = BASE + "?pageSize=300" + (f"&pageToken={page}" if page else "")
        with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60) as r:
            data = json.load(r)
        docs.extend(data.get("documents", []))
        page = data.get("nextPageToken")
        if not page:
            break

    doomed, keep = [], []
    for d in docs:
        fields = {k: plain(v) for k, v in d.get("fields", {}).items()}
        why = verdict(fields)
        (doomed if why else keep).append((d["name"], fields, why))

    print(f"{len(docs)} documents: {len(keep)} conform, {len(doomed)} do not\n")
    for name, f, why in doomed:
        print(f"  DELETE {name.split('/')[-1]:14s} who={f.get('who')!r:14s} {why}")
    for name, f, _ in keep:
        print(f"  keep   {name.split('/')[-1]:14s} who={f.get('who')!r:14s} "
              f"leak={f.get('leak')} goodness={f.get('goodness')}")

    if not args.apply:
        print("\ndry run - pass --apply to delete")
        return
    for name, _, _ in doomed:
        req = urllib.request.Request(
            f"https://firestore.googleapis.com/v1/{name}", method="DELETE", headers=hdr)
        urllib.request.urlopen(req, timeout=30)
    print(f"\ndeleted {len(doomed)}; {len(keep)} remain")


if __name__ == "__main__":
    main()
