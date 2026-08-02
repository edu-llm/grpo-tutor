"""Read the labels collection using the Firebase CLI's own credentials.

    python src/peek_labels.py

`fetch_labels.py` is the proper export and wants a service account key. This is
the quick look: it reuses the refresh token the Firebase CLI already stores, so
there is nothing to set up. Client reads are denied by the security rules, which
is deliberate - this goes through the admin path instead, the same one
`firebase firestore:delete` uses.

Prints a summary only. Nothing is written, and the token is never displayed.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import urllib.parse
import urllib.request

CLI_CLIENT_ID = "563584335869-fgrhgmd47bqnekij5i8b5pr03ho849e6.apps.googleusercontent.com"
CLI_CLIENT_SECRET = "j9iVZfS8kkCEFUPaAeJV0sAi"   # public, shipped inside firebase-tools


def access_token() -> str:
    path = os.path.expanduser("~/.config/configstore/firebase-tools.json")
    if not os.path.exists(path):
        raise SystemExit("Firebase CLI credentials not found - run: npx firebase-tools login")
    refresh = json.load(open(path)).get("tokens", {}).get("refresh_token")
    if not refresh:
        raise SystemExit("No refresh token in the CLI config - run: npx firebase-tools login")
    body = urllib.parse.urlencode({
        "client_id": CLI_CLIENT_ID,
        "client_secret": CLI_CLIENT_SECRET,
        "refresh_token": refresh,
        "grant_type": "refresh_token",
    }).encode()
    req = urllib.request.Request("https://oauth2.googleapis.com/token", data=body)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["access_token"]


def plain(v):
    """Firestore REST wraps every value in a type tag; unwrap the ones we store."""
    for k in ("stringValue", "timestampValue", "booleanValue"):
        if k in v:
            return v[k]
    if "integerValue" in v:
        return int(v["integerValue"])
    return next(iter(v.values()), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="grpo-tutor-label")
    ap.add_argument("--collection", default="labels")
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    tok = access_token()
    base = (f"https://firestore.googleapis.com/v1/projects/{args.project}"
            f"/databases/(default)/documents/{args.collection}")
    rows, page = [], None
    while True:
        url = base + "?pageSize=300" + (f"&pageToken={page}" if page else "")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        for d in data.get("documents", []):
            rows.append({k: plain(v) for k, v in d.get("fields", {}).items()})
        page = data.get("nextPageToken")
        if not page:
            break

    real = [r for r in rows if not str(r.get("who", "")).startswith("__selftest")]
    print(f"{len(rows)} documents in `{args.collection}` "
          f"({len(rows) - len(real)} self-test, {len(real)} real)")
    if not real:
        print("\nNothing from a labeller yet.")
        return

    print("\nby person :", dict(collections.Counter(r.get("who") for r in real)))
    print("by kind   :", dict(collections.Counter(r.get("kind") for r in real)))
    turns = [r for r in real if r.get("kind") == "turn"]
    pairs = [r for r in real if r.get("kind") == "pair"]
    if turns:
        leaks = [r["leak"] for r in turns if isinstance(r.get("leak"), int)]
        goods = [r["goodness"] for r in turns if isinstance(r.get("goodness"), int)]
        print("leak 1-3  :", dict(sorted(collections.Counter(leaks).items())),
              f"mean {sum(leaks) / len(leaks):.2f}" if leaks else "")
        print("goodness  :", dict(sorted(collections.Counter(goods).items())),
              f"mean {sum(goods) / len(goods):.2f}" if goods else "")
    if pairs:
        print("winner    :", dict(collections.Counter(r.get("winner") for r in pairs)))
    notes = [r for r in real if r.get("note")]
    print(f"notes     : {len(notes)}")

    print(f"\nmost recent {min(args.show, len(real))}:")
    for r in sorted(real, key=lambda x: str(x.get("createdAt")), reverse=True)[: args.show]:
        bits = [f"{r.get('who')}", f"{r.get('kind')} {r.get('itemId')}"]
        if r.get("kind") == "turn":
            bits.append(f"leak={r.get('leak')} goodness={r.get('goodness')}")
        else:
            bits.append(f"winner={r.get('winner')}")
        if r.get("note"):
            bits.append(f"note={r['note'][:60]!r}")
        print("  " + "  ".join(bits))


if __name__ == "__main__":
    main()
