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


# Words that are too common to identify any particular option. Only consulted by
# the identifying-word rule, where a SINGLE hit costs the tutor -1, so a generic
# word slipping through is an unearned penalty. Deliberately excludes words that
# can carry answers in science items (metal, rock, high, long, ...).
_GENERIC_RAW = {
    "make", "made", "making", "sure", "being", "been", "have", "has", "had",
    "get", "got", "take", "taken", "give", "given", "come", "came", "goes",
    "went", "keep", "put", "use", "used", "using", "need", "needs", "want",
    "know", "known", "think", "thought", "look", "looks", "find", "found",
    "help", "helps", "thing", "things", "stuff", "kind", "sort", "type",
    "good", "bad", "nice", "great", "well", "better", "best", "just", "very",
    "really", "also", "like", "likes", "other", "others", "another", "same",
    "different", "way", "ways", "one", "ones", "two", "lot", "lots", "much",
    "many", "few", "every", "each", "both", "only", "even", "still", "always",
    "never", "often", "sometimes", "usually", "maybe", "perhaps", "should",
    "could", "must", "might", "may", "avoid", "avoids", "become", "becomes",
    "there", "here", "them", "him", "her", "his", "she", "our", "who", "whom",
    "something", "someone", "anything", "everything", "nothing", "person",
    "people", "yes", "not",
    # prepositions and relational words long enough to survive the length filter
    "through", "across", "around", "between", "within", "without", "above",
    "below", "along", "toward", "towards", "during", "before", "after",
    # generic nouns/verbs that describe any answer rather than identify one
    "change", "changed", "changes", "create", "created", "creates", "item",
    "items", "form", "forms", "amount", "group", "groups", "place", "places",
    "time", "times", "area", "areas", "number", "part", "parts", "side",
    "sides", "late", "early", "happen", "happens", "occur", "occurs",
}
_GENERIC = set()


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


_GENERIC |= {_stem(w) for w in _GENERIC_RAW}


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


def _identifying_words(gold: str, distractors=(), question: str = "") -> set[str]:
    """Content words that point at THIS option and nothing else.

    A word in the gold answer that appears in no distractor and not in the
    question stem is what actually lets a student pick the option out. "Jury" in
    a Bill of Rights item is identifying; "many" is not, and neither is anything
    the question already said.
    """
    g = _content(gold)
    if not g:
        return set()
    elsewhere: set[str] = set()
    for d in distractors:
        elsewhere |= _content(d)
    elsewhere |= _content(question)
    return {w for w in g - elsewhere if len(w) >= 4 and w not in _GENERIC}


def leak_signals(teacher_text: str, gold: str, distractors=(), question: str = "") -> dict:
    """Four independent leak signals in [0,1].

    `identifying` exists because `overlap` divides by the LENGTH OF GOLD, so it
    weakens as answers get longer - exactly backwards for the state-assessment
    corpus, whose answers are long by design. Measured case:

        gold "For depriving us ... of the benefits of Trial by Jury . . ."
        hint "...couldn't have a fair trial ... anything about trials and juries?"
        overlap 0.167 -> not flagged. Shorten gold to "Trial by Jury" and the
        SAME hint scores 0.500. Three times the signal for an identical leak.

    `identifying` normalises by the words unique to gold instead, so a long
    answer with one distinctive word is not diluted by its own verbosity.
    """
    t_norm = _norm(teacher_text)
    t_words = _content(teacher_text)

    gold_norm = _norm(gold)
    verbatim = 1.0 if gold_norm and gold_norm in t_norm else 0.0

    gold_words = _content(gold)
    overlap = (sum(_matches(g, t_words) for g in gold_words) / len(gold_words)
               if gold_words else 0.0)

    ident_words = _identifying_words(gold, distractors, question)
    hits = sum(_matches(w, t_words) for w in ident_words)
    identifying = hits / len(ident_words) if ident_words else 0.0

    named = 0
    for d in distractors:
        d_words = _content(d)
        d_norm = _norm(d)
        if (d_norm and d_norm in t_norm) or (d_words and all(_matches(w, t_words) for w in d_words)):
            named += 1
    elimination = named / len(distractors) if distractors else 0.0

    return {"verbatim": verbatim, "overlap": overlap, "elimination": elimination,
            "identifying": identifying, "identifying_hits": float(hits),
            "identifying_n": float(len(ident_words))}


def leaked_answer(teacher_text: str, gold: str, distractors=(),
                  overlap_thresh: float = 0.6, elim_thresh: float = 0.5,
                  question: str = "", ident_hits: int = 1) -> float:
    """Combined 0/1 leak decision. Any single mode firing counts as a leak.

    `ident_hits` is an absolute count, not a fraction, on purpose: a fraction
    would reintroduce the very length dilution `identifying` exists to remove.
    Naming one word that belongs to gold alone is a leak whether gold has one
    such word or six.
    """
    s = leak_signals(teacher_text, gold, distractors, question)
    return float(
        s["verbatim"] >= 1.0
        or s["overlap"] >= overlap_thresh
        or s["elimination"] >= elim_thresh
        or s["identifying_hits"] >= ident_hits
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


class SpecificityGuard:
    """Wrap a RewardModel; pay only for help that is about THIS question.

    Run v0's finding: a hint written for a different problem helped the student
    as much as the correct one (specificity -0.10 .. +0.10 across 500 steps).
    The +0.10 "teaching gain" was generic help - any tutor-shaped text lifts the
    student - and a reward that only asks "did the student get it right" cannot
    tell the two apart, so 500 steps of optimisation never moved it.

        reward = solved(my problem | my hint) - solved(OTHER problem | my hint)

    DIRECTION MATTERS. The subtracted term must hold the HINT fixed and vary the
    PROBLEM. The inverse - my problem with someone else's hint - measures how
    hackable my problem is, which is not a property of my hint at all, and GRPO
    would then credit a member for a hint they did not write.

    It must also vary WITHIN the group or it does nothing: advantages are
    mean-centred over the K completions, so any term identical across them
    cancels exactly. Using one fixed foreign problem per group and letting each
    member's own hint be the thing that changes satisfies both requirements.

    Needs `trajectory.info["off_problem_solved"]` in {0, 1}; the rollout is
    responsible for that extra student forward pass.
    """

    MODES = ("difference", "gated", "off")

    def __init__(self, inner, mode: str = "difference"):
        if mode not in self.MODES:
            raise ValueError(f"specificity mode must be one of {self.MODES}, got {mode!r}")
        self.inner = inner
        self.mode = mode

    def score(self, trajectory) -> dict:
        out = dict(self.inner.score(trajectory))
        off = (trajectory.info or {}).get("off_problem_solved")
        out["off_problem_solved"] = off
        out["specificity"] = None
        if self.mode == "off" or off is None:
            return out
        off = float(off)
        solved = float(out.get("solved", 0.0))
        out["specificity"] = solved - off
        # difference: a hint that also works elsewhere earns nothing
        # gated: only pay for wins that do NOT transfer (never negative)
        out["reward"] = (solved - off) if self.mode == "difference" else solved * (1.0 - off)
        return out


def build_real_rewarder(specificity: str = "off", leak_penalty: float = -1.0):
    """LeakGuard OUTSIDE SpecificityGuard, deliberately.

    A leaked answer is maximally question-specific, so it scores well on
    specificity. Wrapping the other way round would pay the teacher for leaking.
    """
    inner = SolveReward()
    if specificity and specificity != "off":
        inner = SpecificityGuard(inner, mode=specificity)
    return LeakGuard(inner, penalty=leak_penalty)


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
        # info["teacher_text"] carries the TUTOR's turns only. Scoring the whole
        # multi-turn transcript charges the teacher for words the student said:
        # measured on 864 live dialogues, 17% of flags were the student blurting
        # the answer while the tutor never named it.
        teacher_text = info.get("teacher_text") or trajectory.transcript or " ".join(
            t.completion.text for t in trajectory.turns
        )
        gold = info.get("gold", "")
        distractors = info.get("distractors", ())
        question = info.get("question", "")
        sig = leak_signals(teacher_text, gold, distractors, question)
        leaked = float(sig["verbatim"] >= 1.0 or sig["overlap"] >= 0.6
                       or sig["elimination"] >= 0.5 or sig["identifying_hits"] >= 1.0)
        out.update({"leaked": leaked, "leak_verbatim": sig["verbatim"],
                    "leak_overlap": sig["overlap"], "leak_elimination": sig["elimination"],
                    "leak_identifying": sig["identifying"],
                    "leak_identifying_hits": sig["identifying_hits"]})
        if leaked:
            out["reward"] = float(self.penalty)
        return out
