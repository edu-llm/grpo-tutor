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
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_ENTITY=eduLLM

cd "$REPO_ROOT"
mkdir -p logs
