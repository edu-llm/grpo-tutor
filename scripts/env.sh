# Source this before any run. Keeps ALL caches/outputs off the (full) home dir.
source ~/venv_test/bin/activate
export SCRATCH=/home/zsophia/orcd/scratch
export HF_HOME=$SCRATCH/hf-cache
export HF_HUB_CACHE=$SCRATCH/hf-cache/hub
export PIP_CACHE_DIR=$SCRATCH/pip-cache
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd $SCRATCH/grpo_tutor
