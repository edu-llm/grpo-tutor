"""Seams. These Protocols are the contract between the RL infra (yours) and the
components other people build (student model, PRM / reward model).

Nothing here imports torch or vLLM - it is pure types, so a teammate can build a
Student or RewardModel without touching the training stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# --------------------------------------------------------------------------
# data carried through the pipeline
# --------------------------------------------------------------------------

@dataclass
class Completion:
    """One generated continuation plus the info the GRPO update needs."""
    text: str
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)  # behavior ("old") logprobs


@dataclass
class Turn:
    """One teacher action: the context it saw and what it generated."""
    prompt: str
    completion: Completion


@dataclass
class Trajectory:
    """One rollout (a full tutoring episode) for one group member."""
    turns: list[Turn]
    transcript: str = ""
    reward: float = 0.0
    info: dict = field(default_factory=dict)   # anything the PRM/monitor wants to attach


# --------------------------------------------------------------------------
# seams
# --------------------------------------------------------------------------

@runtime_checkable
class InferenceEngine(Protocol):
    """Generation backend for the POLICY (teacher). vLLM / SGLang / HF / stub."""

    def generate(self, prompts: list[str], n: int, max_new_tokens: int,
                 temperature: float) -> list[list[Completion]]:
        """Return, per prompt, `n` completions (with token ids + behavior logprobs)."""
        ...

    def sync_weights(self, source) -> None:
        """Make the engine generate with the trainer's current weights."""
        ...

    def sleep(self) -> None:
        """Release GPU memory so the trainer can use it (colocated mode)."""
        ...

    def wake(self) -> None:
        """Re-acquire GPU memory before the next generation phase."""
        ...


@runtime_checkable
class Student(Protocol):
    """The environment. SOMEONE ELSE BUILDS THIS.

    This is the contract the training loop and evals ACTUALLY call
    (`zpd_filter.HFStudent` / `StubStudent` implement it): answer a multiple-choice
    question, optionally with the teacher's hint in context, and return the index
    of the chosen option.

    Scoring by index (rather than free text) keeps the reward exact - no answer
    parsing - and works with base models via length-normalized choice log-probs.

    NOTE: a conversational `reply(...)` method will be needed if/when multi-turn
    tutoring dialogue lands; it is deliberately NOT declared here because nothing
    calls it yet.

    A student MAY also offer `answer(question, choices, hint)` - a second, freer
    answering channel selected by `cfg.student_answer_mode`. It stays out of the
    contract because `zpd_filter.student_answer` falls back to `choose()` when it
    is absent, so implementing only `choose()` remains enough.
    """

    def choose(self, question: str, choices: list[str], hint: str = "") -> int:
        ...


@runtime_checkable
class RewardModel(Protocol):
    """Scores a trajectory. SOMEONE ELSE BUILDS THIS (PRM or verifier).

    Returning a dict lets a PRM expose per-step / diagnostic signals alongside
    the scalar the RL update consumes.
    """

    def score(self, trajectory: Trajectory) -> dict:
        """Must include {"reward": float}; may add any extra diagnostic keys."""
        ...
