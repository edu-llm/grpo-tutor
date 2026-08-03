"""Does the student answer its OWN question during the dialogue?

The final score comes from student.choose(question, choices, hint=transcript),
and the transcript includes the student's own turns. So if the student blurts
the gold answer mid-conversation, it then reads that back and scores correct -
and the teacher collects the reward for something it never said.

This is the mirror of the leak-attribution bug: that one wrongly PUNISHED the
teacher for the student's words, this one wrongly PAYS it.
"""

from __future__ import annotations

import glob
import json
import sys


def split_speakers(convo: str):
    tutor, stud = [], []
    for line in convo.split("\n"):
        if line.startswith("Tutor:"):
            tutor.append(line[len("Tutor:"):].strip())
        elif line.startswith("Student:"):
            stud.append(line[len("Student:"):].strip())
    return " ".join(tutor).lower(), " ".join(stud).lower()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("runs/*/traces.jsonl"))[-1]
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("turns")]
    if not rows:
        raise SystemExit("no multi-turn traces")

    n = len(rows)
    self_said = [r for r in rows
                 if str(r.get("gold", "")).lower() in split_speakers(r.get("completion", ""))[1]]
    tutor_silent = [r for r in self_said
                    if str(r.get("gold", "")).lower()
                    not in split_speakers(r.get("completion", ""))[0]]

    solved_all = sum(1 for r in rows if r.get("solved"))
    solved_self = sum(1 for r in self_said if r.get("solved"))
    solved_self_only = sum(1 for r in tutor_silent if r.get("solved"))

    print(f"dialogues                          : {n}")
    print(f"student said gold in its own turns : {len(self_said)} ({len(self_said)/n:.0%})")
    print(f"  ...and tutor never said it       : {len(tutor_silent)} ({len(tutor_silent)/n:.0%})")
    print()
    print(f"solved overall                     : {solved_all} ({solved_all/n:.0%})")
    print(f"solved | student self-said gold    : {solved_self}/{len(self_said)}"
          f" ({solved_self/max(1,len(self_said)):.0%})")
    print(f"solved | ONLY the student said it  : {solved_self_only}/{len(tutor_silent)}"
          f" ({solved_self_only/max(1,len(tutor_silent)):.0%})")
    rest = [r for r in rows if r not in self_said]
    solved_rest = sum(1 for r in rest if r.get("solved"))
    print(f"solved | student never said it     : {solved_rest}/{len(rest)}"
          f" ({solved_rest/max(1,len(rest)):.0%})")


if __name__ == "__main__":
    main()
