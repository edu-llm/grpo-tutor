"""Regression tests for the leak rules.

The property under test is the one that broke: a leak must be detected the same
way whether the gold answer is quoted in full or trimmed to its key phrase. The
old overlap rule divided by gold length, so quoting a whole clause from a state
test hid the leak. Run: python -m pytest tests/ -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rewards import leak_signals, leaked_answer  # noqa: E402

QUESTION = ("Which grievance from the Declaration of Independence is addressed "
            "by the Sixth Amendment to the U.S. Constitution?")
GOLD_FULL = "“For depriving us in many cases, of the benefits of Trial by Jury . . .”"
GOLD_TRIMMED = "Trial by Jury"
DISTRACTORS = [
    "“He has dissolved Representative Houses repeatedly.”",
    "“He has obstructed the Administration of Justice . . .”",
    "“For imposing Taxes on us without our Consent . . .”",
]
LEAKY = ("Think of it this way: If someone was upset because they couldn't have "
         "a fair trial when accused by the government, which part of the "
         "declaration fits that? Do you remember anything about trials and juries?")
CONCEPTUAL = ("A grievance is a complaint about being treated unfairly. Think "
              "about which of these is about what happens to a person after "
              "they are accused of something, rather than about money or "
              "lawmaking. Which one fits?")


def test_leak_detected_regardless_of_gold_length():
    """The defect: same hint, same leak, but detection depended on gold length."""
    assert leaked_answer(LEAKY, GOLD_FULL, DISTRACTORS, question=QUESTION) == 1.0
    assert leaked_answer(LEAKY, GOLD_TRIMMED, DISTRACTORS, question=QUESTION) == 1.0


def test_overlap_alone_would_have_missed_the_long_form():
    """Pin the reason the new signal is needed, so nobody deletes it as redundant."""
    long_sig = leak_signals(LEAKY, GOLD_FULL, DISTRACTORS, QUESTION)
    short_sig = leak_signals(LEAKY, GOLD_TRIMMED, DISTRACTORS, QUESTION)
    assert long_sig["overlap"] < 0.6
    assert long_sig["overlap"] < short_sig["overlap"]
    assert long_sig["identifying_hits"] >= 1


def test_conceptual_hint_does_not_flag():
    assert leaked_answer(CONCEPTUAL, GOLD_FULL, DISTRACTORS, question=QUESTION) == 0.0


def test_question_stem_words_are_not_identifying():
    """Echoing the question back cannot be a leak; the student already read it."""
    echo = "Look again at the grievance and the Sixth Amendment. Which fits?"
    assert leaked_answer(echo, GOLD_FULL, DISTRACTORS, question=QUESTION) == 0.0


def test_generic_words_are_not_identifying():
    """A single generic word costs -1 if it counts, so it must not count."""
    sig = leak_signals("Think about what makes people change over time.",
                       "a change that makes many people", [], "")
    assert sig["identifying_hits"] == 0


def test_verbatim_still_wins():
    assert leaked_answer("The answer is Trial by Jury.", GOLD_FULL, DISTRACTORS,
                         question=QUESTION) == 1.0
