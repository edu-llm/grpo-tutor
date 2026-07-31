"""One call that seeds every RNG a run actually draws from.

`cfg.seed` used to reach exactly two places: the problem sampler and the
train/test split. Everything else that moves was unseeded, so two runs with the
same seed diverged from the first dialogue onwards:

  - the student samples its turns (`do_sample=True, temperature=0.8`) off
    torch's GLOBAL generator - `HFStudent.reply` passes no generator, so seeding
    torch once at startup is enough to fix it without touching that method;
  - the teacher samples at `cfg.temperature=1.0` inside vLLM, which keeps its own
    per-request generator (see `engine.VLLMEngine`, which asks for a per-call
    seed derived from this one).

WHAT THIS DOES NOT BUY. Seeding makes the *stream* of random numbers fixed; it
does not make the computation bit-identical:

  - vLLM batches requests continuously, and the numerics of a token depend on
    what else is in the batch, so identical seeds can still produce different
    text under load. Nothing here fixes that.
  - `torch.use_deterministic_algorithms(True)` is deliberately NOT set: it makes
    several attention/scatter kernels fall back to slow paths or raise outright,
    and a 10x slower rollout is a worse trade than a reproducible one.
  - Python's `hash()` of a str is salted per process. `StubStudent` keys its
    answers off `hash(question)`, so stub numbers only repeat when
    PYTHONHASHSEED is exported in the shell (this cannot be fixed from inside
    the process - by the time we run, the salt is already chosen).
"""

from __future__ import annotations

import random

import torch

MAX_SEED = 2**31 - 1


def seed_everything(seed: int) -> int:
    """Seed python / torch (CPU + all CUDA devices) / numpy. Returns the seed."""
    seed = int(seed) % MAX_SEED
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:      # numpy rides in with transformers; the stub path can live without it
        pass
    return seed


def derive_seed(base: int, *parts) -> int:
    """A stable child seed for one call site.

    Reusing one seed for every generation call would make every step sample the
    SAME continuations; drawing a fresh random one would not survive a requeue.
    Mixing (base, call index) arithmetically - rather than through `hash()`,
    which is salted per process for strings - gives a value fixed by the run seed
    and still different for each call.
    """
    acc = int(base) % MAX_SEED
    for p in parts:
        acc = (acc * 1_000_003 + int(p)) % MAX_SEED
    return acc
