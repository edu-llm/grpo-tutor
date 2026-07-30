"""How many 'leaks' are the STUDENT's words, not the teacher's?

LeakGuard scores trajectory.transcript, which in multi-turn contains both
speakers. If the student blurts the answer, the teacher is penalised for it.
"""

from __future__ import annotations

import glob
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rewards import leak_signals  # noqa: E402


def split_speakers(convo: str):
    tutor, stud = [], []
    for line in convo.split("\n"):
        if line.startswith("Tutor:"):
            tutor.append(line[len("Tutor:"):].strip())
        elif line.startswith("Student:"):
            stud.append(line[len("Student:"):].strip())
    return " ".join(tutor), " ".join(stud)


def fires(text, gold, distractors):
    s = leak_signals(text, gold, distractors)
    return s["verbatim"] >= 1.0 or s["overlap"] >= 0.6 or s["elimination"] >= 0.5


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("runs/*/traces.jsonl"))[-1]
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("turns")]
    if not rows:
        raise SystemExit("no multi-turn traces")

    both = tutor_only = student_only = neither = 0
    for r in rows:
        gold = r.get("gold", "")
        # distractors are not stored in traces; verbatim+overlap still fire
        t_txt, s_txt = split_speakers(r.get("completion", ""))
        t = fires(t_txt, gold, ())
        s = fires(s_txt, gold, ())
        if t and s:
            both += 1
        elif t:
            tutor_only += 1
        elif s:
            student_only += 1
        else:
            neither += 1

    n = len(rows)
    flagged_now = both + tutor_only + student_only     # what scoring the whole convo catches
    correct = both + tutor_only                        # what scoring tutor-only would catch
    print(f"traces                         : {n}")
    print(f"flagged scoring WHOLE convo    : {flagged_now} ({flagged_now/n:.0%})")
    print(f"flagged scoring TUTOR ONLY     : {correct} ({correct/n:.0%})")
    print(f"  teacher blamed for STUDENT's words: {student_only} "
          f"({student_only/max(1,flagged_now):.0%} of current flags)")
    print()
    print(f"  both speakers said it : {both}")
    print(f"  tutor only            : {tutor_only}")
    print(f"  student only          : {student_only}")
    print(f"  neither               : {neither}")


if __name__ == "__main__":
    main()
