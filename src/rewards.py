"""Composable reward rules.

SolveReward  - did the student solve the problem after tutoring? (the real signal)
LeakGuard    - wraps ANY RewardModel: if the teacher gave the answer away, the
               reward is slammed to a large negative value.

Leak detection is deliberately RULE-BASED (not a learned judge), so the policy
cannot hack it the way it could hack a reward model. It covers three modes seen
in real traces:

  verbatim    "the answer is copper wire"          - gold text stated outright
  paraphrase  gold "weigh the same"                - same content words, reworded
              hint "doesn't change its weight"
  elimination gold "polar opposite"                - names the WRONG options so
              hint "not wooden, salty, or wind"      the student can rule them out

What it deliberately does NOT flag: conceptual hints that point at the mechanism
without naming the option ("think about what melts snow" for `heat lamp`). That
is arguably good teaching, and rules cannot adjudicate it - the transfer check in
evals.py is the tool for that grey zone.
"""

from __future__ import annotations

import re

_STOP = {
    "the", "a", "an", "of", "to", "in", "is", "are", "was", "were", "be", "and",
    "or", "it", "its", "that", "this", "these", "those", "for", "on", "at", "as",
    "with", "by", "from", "than", "then", "so", "if", "not", "no", "do", "does",
    "you", "your", "they", "their", "what", "which", "when", "how", "why", "can",
    "will", "would", "about", "into", "more", "most", "some", "any", "all",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).strip()


def _stem(w: str) -> str:
    """Crude suffix stripping - enough to match weigh/weight, melt/melting."""
    for suf in ("ing", "edly", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _content(s: str) -> set[str]:
    return {_stem(w) for w in _norm(s).split() if w not in _STOP and len(w) > 2}


def _matches(gold_word: str, text_words: set[str], min_prefix: int = 4) -> bool:
    """Loose word match: exact, or one is a prefix of the other (>=4 chars).

    Suffix-stripping alone misses pairs like weigh/weight; prefix matching catches
    them without pulling in unrelated words.
    """
    if gold_word in text_words:
        return True
    return any(
        (w.startswith(gold_word) or gold_word.startswith(w))
        and min(len(w), len(gold_word)) >= min_prefix
        for w in text_words
    )


def leak_signals(teacher_text: str, gold: str, distractors=()) -> dict:
    """Three independent leak signals in [0,1]."""
    t_norm = _norm(teacher_text)
    t_words = _content(teacher_text)

    gold_norm = _norm(gold)
    verbatim = 1.0 if gold_norm and gold_norm in t_norm else 0.0

    gold_words = _content(gold)
    overlap = (sum(_matches(g, t_words) for g in gold_words) / len(gold_words)
               if gold_words else 0.0)

    named = 0
    for d in distractors:
        d_words = _content(d)
        d_norm = _norm(d)
        if (d_norm and d_norm in t_norm) or (d_words and all(_matches(w, t_words) for w in d_words)):
            named += 1
    elimination = named / len(distractors) if distractors else 0.0

    return {"verbatim": verbatim, "overlap": overlap, "elimination": elimination}


def leaked_answer(teacher_text: str, gold: str, distractors=(),
                  overlap_thresh: float = 0.6, elim_thresh: float = 0.5) -> float:
    """Combined 0/1 leak decision. Any single mode firing counts as a leak."""
    s = leak_signals(teacher_text, gold, distractors)
    return float(
        s["verbatim"] >= 1.0
        or s["overlap"] >= overlap_thresh
        or s["elimination"] >= elim_thresh
    )


def hint_only_leak(student, hint: str, choices, gold_idx: int) -> float:
    """Information-theoretic leak probe: can the student pick the gold answer from
    the HINT ALONE, with the question hidden?

    If yes, the hint carried the answer. This is strictly stronger than string rules:
    it catches verbatim, semantic paraphrase AND elimination in one number, with no
    similarity threshold to tune. Cost is one extra student forward pass.

    NOTE: use this as a MONITOR, not as a reward term. Optimizing against it pushes
    the teacher toward vacuous hints that say nothing at all.
    """
    idx = student.choose("(hidden)", list(choices), hint=hint)
    return float(idx == gold_idx)


def choices_only_baseline(student, choices, gold_idx: int) -> float:
    """The FLOOR for hint_only_leak: same prompt, hint removed.

    Without this, hint_only_leak is uninterpretable. LLMs routinely beat the
    majority baseline from the choices alone - distractors leak through style,
    length and topical coherence - so part of any hint-only score belongs to the
    answer options, not to the teacher. Report hint_only_leak MINUS this.
    """
    idx = student.choose("(hidden)", list(choices), hint="")
    return float(idx == gold_idx)


class SolveReward:
    """1.0 if the student's post-tutoring answer matches gold, else 0.0."""

    def __init__(self, partial_credit: bool = False):
        self.partial_credit = partial_credit

    def score(self, trajectory) -> dict:
        info = trajectory.info or {}
        pred = _norm(info.get("student_answer", ""))
        gold = _norm(info.get("gold", ""))
        if not gold:
            return {"reward": 0.0, "solved": 0.0}
        solved = 1.0 if (pred == gold or (self.partial_credit and gold in pred)) else 0.0
        return {"reward": solved, "solved": solved}


class LeakGuard:
    """Wrap a RewardModel; punish giving the answer away.

    `penalty` is kept modest (-1.0) on purpose: GRPO normalizes advantages within
    the group, so a huge penalty mostly inflates the group std and CRUSHES the
    solved-vs-failed signal rather than punishing leaks harder. -1.0 already means
    "strictly worse than honestly failing", which is the strongest statement the
    normalization can carry.
    """

    def __init__(self, inner, penalty: float = -1.0):
        self.inner = inner
        self.penalty = penalty

    def score(self, trajectory) -> dict:
        out = dict(self.inner.score(trajectory))
        info = trajectory.info or {}
        teacher_text = trajectory.transcript or " ".join(
            t.completion.text for t in trajectory.turns
        )
        gold = info.get("gold", "")
        distractors = info.get("distractors", ())
        sig = leak_signals(teacher_text, gold, distractors)
        leaked = float(sig["verbatim"] >= 1.0 or sig["overlap"] >= 0.6
                       or sig["elimination"] >= 0.5)
        out.update({"leaked": leaked, "leak_verbatim": sig["verbatim"],
                    "leak_overlap": sig["overlap"], "leak_elimination": sig["elimination"]})
        if leaked:
            out["reward"] = float(self.penalty)
        return out
