"""Placeholder reward until the real student/PRM lands.

Deliberately LEARNABLE so it doubles as a pipeline test: if the teacher starts
gaming the fake signal and mean reward climbs, your GRPO plumbing (generation ->
advantage -> loss -> optimizer -> weight sync) provably works. If reward stays
flat on a trivially-gameable target, something upstream is broken.

Modes:
  keyword  - +1 per target word present (easy to SEE in traces). Default.
  length   - prefers a target length (tests a smooth, non-lexical signal).
  random   - pure noise; reward should NOT climb. Use as a negative control.
Implements the RewardModel protocol from interfaces.py.
"""

from __future__ import annotations

import random
import re

DEFAULT_KEYWORDS = ("step", "because", "first", "check", "example")


class FakeReward:
    def __init__(self, mode: str = "keyword", keywords=DEFAULT_KEYWORDS,
                 target_len: int = 60, seed: int = 0):
        self.mode = mode
        self.keywords = tuple(w.lower() for w in keywords)
        self.target_len = target_len
        self.rng = random.Random(seed)

    def score(self, trajectory) -> dict:
        text = trajectory.transcript or " ".join(
            t.completion.text for t in trajectory.turns
        )
        if self.mode == "keyword":
            low = text.lower()
            hits = sum(1 for w in self.keywords if re.search(rf"\b{re.escape(w)}", low))
            reward = hits / len(self.keywords)
            info = {"hits": hits}
        elif self.mode == "length":
            n = len(text.split())
            reward = max(0.0, 1.0 - abs(n - self.target_len) / self.target_len)
            info = {"n_words": n}
        elif self.mode == "random":
            reward = self.rng.random()
            info = {}
        else:
            raise ValueError(f"unknown fake reward mode: {self.mode}")
        return {"reward": float(reward), **info}
