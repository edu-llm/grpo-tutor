"""Structural audit of extracted state-assessment items.

    python src/audit_items.py data/state_tests/train_items.jsonl

These items come out of PDFs, and PDF extraction fails quietly: an option loses
its tail, a fraction like 1/10 comes apart into a stray '1', a question refers to
a diagram that was never captured. None of that raises an error - it just puts a
broken problem into training, the eval, and the labelling app.

Every check here is structural. It cannot tell you whether gold_idx is the right
answer; it can only find items whose TEXT is damaged or unanswerable. Read the
flagged ones.
"""

from __future__ import annotations

import argparse
import collections
import json
import re

# a question that points at something the extractor cannot have captured
VISUAL = re.compile(
    r"\b(diagram|figure|picture|image|graph|chart|map|table below|the table|"
    r"shown below|above|following passage|the passage|illustration|"
    r"this model|the model below|photograph|drawing)\b", re.I)


def checks(r):
    out = []
    q, ch, gi = r.get("question", ""), r.get("choices", []), r.get("gold_idx")

    if not isinstance(gi, int) or not (0 <= gi < len(ch)):
        out.append("gold_idx out of range")
    if len(ch) != 4:
        out.append(f"{len(ch)} choices, expected 4")
    if len({str(c).strip().lower() for c in ch}) != len(ch):
        out.append("duplicate choices")
    if len(q.split()) < 4:
        out.append("question too short")
    if any(not str(c).strip() for c in ch):
        out.append("empty choice")

    # A fraction set as a stacked numerator/denominator comes out of the PDF as a
    # loose digit stranded after the sentence, and the fraction itself vanishes:
    #   "...is 1/10 the value..."  ->  "...is the value... place. 1"
    for i, c in enumerate(ch):
        s = str(c).strip()
        if re.search(r"[.,;:]\s+\d{1,2}$", s) or re.search(r"\b\d{1,2}\s+(ten|hundred|thousand)s?\b", s):
            out.append(f"choice {chr(65+i)} has a stranded number (torn fraction?)")
            break

    # an option that stops mid-clause. Sentence-completion stems legitimately end
    # in 'the' or 'of', so only flag CHOICES, and only long ones.
    for i, c in enumerate(ch):
        s = str(c).strip().rstrip(".")
        if len(s.split()) >= 6 and re.search(
                r"\b(the|of|in|an|to|and|is|that|for|with|by)$", s, re.I):
            out.append(f"choice {chr(65+i)} looks truncated")
            break

    # one option that is a prefix of another usually means one got cut short
    norm = [re.sub(r"[^a-z0-9 ]", "", str(c).lower()).strip() for c in ch]
    for i, a in enumerate(norm):
        for j, b in enumerate(norm):
            if i != j and a and b and a != b and b.startswith(a) and len(a.split()) >= 5:
                out.append(f"choice {chr(65+i)} is a truncated copy of {chr(65+j)}")
                break
        else:
            continue
        break

    # Two options that differ by only a word or two are usually one option that
    # got garbled, and sometimes two distractors with the same VALUE - which a
    # real test item would not ship. Exact-duplicate matching misses these:
    # "four tens" and "forty ones" share no words but are both 40.
    toks = [set(re.findall(r"[a-z0-9]+", str(c).lower())) for c in ch]
    for i in range(len(ch)):
        for j in range(i + 1, len(ch)):
            a, b = toks[i], toks[j]
            # Only identical word sets are a real smell. A 1-2 word difference is
            # how good distractors are built - faster/slower, increase/decrease -
            # and flagging those buries the genuine cases.
            if len(a) >= 5 and a == b:
                out.append(f"choices {chr(65+i)} and {chr(65+j)} use the same words")
                break
        else:
            continue
        break

    if VISUAL.search(q):
        out.append("refers to something visual")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    for path in args.files:
        rows = [json.loads(l) for l in open(path)]
        flagged, tally = [], collections.Counter()
        for r in rows:
            probs = checks(r)
            if probs:
                flagged.append((r, probs))
                for p in probs:
                    tally[re.sub(r"choice [A-D]", "a choice", p)] += 1

        print(f"\n=== {path}: {len(flagged)}/{len(rows)} flagged "
              f"({len(flagged) / len(rows):.1%}) ===")
        for k, v in tally.most_common():
            print(f"  {v:4d}  {k}")

        # duplicates across the file
        qs = collections.Counter(r["question"] for r in rows)
        dupes = [q for q, n in qs.items() if n > 1]
        if dupes:
            print(f"  {len(dupes)} questions appear more than once")

        for r, probs in flagged[: args.show]:
            print(f"\n  -- {', '.join(probs)}")
            print(f"     Q: {r['question'][:150]}")
            for i, c in enumerate(r["choices"]):
                mark = " <-- gold" if i == r.get("gold_idx") else ""
                print(f"        {chr(65+i)}. {str(c)[:110]}{mark}")


if __name__ == "__main__":
    main()
