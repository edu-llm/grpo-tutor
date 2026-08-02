# Source this before any run:  source scripts/env.sh
# Keeps ALL caches/outputs off the (full) home dir.

source ~/venv_test/bin/activate

# Derived from this file's own location, so the repo can live anywhere and the
# scripts work from any submit directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SCRATCH="${SCRATCH:-$(dirname "$REPO_ROOT")}"

export HF_HOME=$SCRATCH/hf-cache
export HF_HUB_CACHE=$SCRATCH/hf-cache/hub
export PIP_CACHE_DIR=$SCRATCH/pip-cache

# vLLM compiles the model on first load and caches the result. That cache
# defaults to ~/.cache/vllm, and HOME is a 200GB quota that OTHER projects keep
# full - a job then dies with "Disk quota exceeded" during engine init, which
# reads like a vLLM bug and is not one. Everything cache-shaped goes to scratch.
export XDG_CACHE_HOME=$SCRATCH/xdg-cache
export XDG_CONFIG_HOME=$SCRATCH/xdg-config
export VLLM_CACHE_ROOT=$SCRATCH/vllm-cache
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor-cache
# vLLM's usage reporter writes ~/.config/vllm/usage_stats.json, which no cache
# variable covers - it killed a run at engine init with "Disk quota exceeded"
# that reads like a vLLM bug and is really a full home directory.
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# NB: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here. vLLM's
# sleep mode uses a CUDA memory pool and asserts on startup:
#   "Expandable segments are not compatible with memory pool"
export WANDB_ENTITY=eduLLM
# W&B stages run data before upload; left at its default that lands in HOME and
# fails the same way everything else does.
export WANDB_DIR=$SCRATCH/wandb
export WANDB_CACHE_DIR=$SCRATCH/wandb-cache
export WANDB_CONFIG_DIR=$SCRATCH/wandb-config
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# Preflight. Redirecting caches stops THIS job filling home, but other projects
# share the quota, and a home that is already full fails writes the job still
# makes - it cost a v3 attempt its W&B sample table, silently, at step 0.
#
# Test the failure mode itself rather than reading the quota. `quota -s` talks to
# the NFS quota service and blocks in uninterruptible I/O on the compute nodes:
# it wedged job 19467157 for 15 minutes in state D, having produced no output at
# all, which is a far worse bug than the one it was added to catch. A write is
# what actually fails when the quota is exhausted, so attempt a write.
if ! touch "$HOME/.write_probe" 2>/dev/null; then
  echo "=============================================================" >&2
  echo "WARNING: cannot write to HOME - quota is full." >&2
  echo "  W&B tables, plots and matplotlib config will be skipped." >&2
  echo "  Free space with: rm -rf ~/.cache/vllm ~/.cache/pip ~/.cache/torch ~/.triton" >&2
  echo "  (all regenerable). Then check the rest with: du -sh ~/.cache/*" >&2
  echo "=============================================================" >&2
else
  rm -f "$HOME/.write_probe"
fi

cd "$REPO_ROOT"
mkdir -p logs
