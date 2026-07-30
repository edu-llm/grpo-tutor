"""Repo-root-anchored paths.

Data and output locations must NOT depend on the working directory. A requeued
job that starts in a different cwd would otherwise find no checkpoints/ckpt.pt
and silently restart from step 0 instead of resuming.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
CHECKPOINTS = ROOT / "checkpoints"
