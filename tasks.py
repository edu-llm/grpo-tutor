"""ZPD problem set: loading, a stable train/test split, and prompt construction.

These are the 731 problems from `zpd_filter.py` where the student FAILS ALONE but
SUCCEEDS WITH an oracle hint - the only items that can produce reward gradient.
"""

from __future__ import annotations

import json
import os
import random

DEFAULT_PATH = "data/zpd_problems.jsonl"

_FALLBACK = [
    {"question": "Which object is the best conductor of electricity?",
     "choices": ["a rubber band", "a copper wire", "a glass cup", "a wooden spoon"],
     "gold_idx": 1, "hint": "Metals let electric current flow easily."},
    {"question": "What causes day and night on Earth?",
     "choices": ["Earth orbiting the Sun", "Earth spinning on its axis",
                 "the Moon's phases", "clouds moving"],
     "gold_idx": 1, "hint": "Earth rotates once about every 24 hours."},
]


def load_zpd(path: str = DEFAULT_PATH):
    """Load the curated ZPD problems; falls back to a tiny built-in set if absent."""
    if not os.path.exists(path):
        print(f"[tasks] {path} not found - using {len(_FALLBACK)} fallback problems")
        return list(_FALLBACK)
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def split_problems(items, test_frac: float = 0.15, seed: int = 0):
    """Deterministic split so the held-out set is identical across runs/evals."""
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n_test = max(1, int(len(items) * test_frac)) if items else 0
    test = [items[i] for i in idx[:n_test]]
    train = [items[i] for i in idx[n_test:]]
    return train, test


def format_choices(choices) -> str:
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))


TEACHER_SYSTEM = (
    "You are a patient tutor working through a problem WITH a middle-school student.\n"
    "Your goal is to teach, not to answer. In each turn do whatever helps most:\n"
    "explain the underlying idea, give a small worked example from a similar\n"
    "situation, ask a question that makes them think, or correct a misconception.\n"
    "Give a direct hint only if they are truly stuck.\n"
    "Rules:\n"
    "1. NEVER reveal the answer. Do not name, quote, or rule out any option.\n"
    "2. Keep each turn to 2-3 sentences so it stays a conversation.\n"
    "3. Plain language a 12-year-old understands. No preamble."
)


def dialogue_prompt(problem, transcript: str, tokenizer=None) -> str:
    """Teacher's view mid-conversation: the problem plus the dialogue so far."""
    user = (
        f"The student is working on this question:\n\n{problem['question']}\n"
        f"{format_choices(problem['choices'])}\n\n"
    )
    user += (f"Conversation so far:\n{transcript.strip()}\n\nYour next turn:"
             if transcript.strip() else "Start the conversation - open the discussion:")
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": TEACHER_SYSTEM},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
    return f"{TEACHER_SYSTEM}\n\n{user}"


def student_dialogue_view(problem, transcript: str) -> str:
    """What the student sees when it is their turn to speak."""
    return (f"Question you're stuck on:\n{problem['question']}\n"
            f"{format_choices(problem['choices'])}\n\n"
            f"Conversation so far:\n{transcript.strip()}\n\nYour reply:")


def teacher_prompt(problem, tokenizer=None) -> str:
    """What the teacher sees. Deliberately EXCLUDES the gold answer.

    Applies the model's chat template when a tokenizer is given - an instruct model
    follows rules far better in its native chat format than as raw text (and this
    directly targets the answer-leaking failure mode).
    """
    user = (
        "A student is stuck on this question:\n\n"
        f"{problem['question']}\n{format_choices(problem['choices'])}\n\n"
        "Give your one-hint reply now."
    )
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": TEACHER_SYSTEM},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
    return f"{TEACHER_SYSTEM}\n\n{user}"


def gold_text(problem) -> str:
    return problem["choices"][problem["gold_idx"]]
