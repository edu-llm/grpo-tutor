"""Central configuration.

Backend/device auto-detect so the same code runs on a laptop (HF generate, CPU/MPS,
smoke tests) and on a CUDA box (vLLM, real training).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import paths


@dataclass
class Config:
    # --- models ---
    teacher_model: str = "Qwen/Qwen2.5-3B-Instruct"    # the policy we train (LoRA)
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct"  # frozen "environment"

    # --- backend / device ("auto" resolves at runtime) ---
    backend: str = "auto"    # "auto" | "stub" | "hf" | "vllm"
    device: str = "auto"     # "auto" | "cpu" | "mps" | "cuda"
    dtype: str = "bfloat16"

    # --- GRPO ---
    group_size: int = 8       # K completions sampled per prompt (the "group")
    lr: float = 5e-6        # LoRA+GRPO band is ~5e-6..2e-5 (full-FT RL's 1e-6 is too low here)
    kl_coef: float = 0.05     # weight on KL-to-reference
    clip_eps: float = 0.2     # PPO-style ratio clip
    update_epochs: int = 2    # gradient epochs per generation batch
    sync_every: int = 4       # push LoRA weights into the engine every N steps
    temperature: float = 1.0  # teacher sampling temperature (exploration)
    warmup_steps: int = 10    # linear lr warmup, then CONSTANT (no cosine)
    teacher_max_new_tokens: int = 256
    turns: int = 1            # teacher turns per dialogue; >1 enables multi-turn discussion

    gpu_mem_util: float = 0.45    # vLLM's share of GPU mem; rest is for the trainer
    no_sleep: bool = False        # keep the engine resident (fits on 80GB; saves sleep/wake)

    # --- LoRA ---
    lora_r: int = 32        # verl recommends r>=32 for RL; GRPO gradients are high-rank
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    # --- sequence / memory ---
    max_seq_len: int = 1024      # truncation budget for the loss batch
    micro_batch_size: int = 8    # forward/backward chunk size (memory control)

    # --- training loop ---
    total_steps: int = 500
    batch_prompts: int = 4       # distinct prompts per optimizer step
    seed: int = 0

    # --- held-out benchmarking ---
    eval_every: int = 25      # run a held-out benchmark every N steps (0 = off)
    eval_n: int = 30          # held-out problems per benchmark
    hint_probe: bool = False  # log the hint-only leak probe (1 extra student call/sample)
    eval_benchmark: str | None = None  # external eval set; None = ZPD held-out split

    # --- monitoring ---
    use_wandb: bool = False
    wandb_project: str = "grpo_tutor"
    # the *team* entity, not the org (philote-...-org): wandb rejects runs logged
    # directly to an organization with "please try using your team entity"
    wandb_entity: str = "eduLLM"
    print_samples_every: int = 10
    reward_mode: str = "keyword"  # fake-reward mode (pipeline test only)

    # --- io ---
    save_dir: str = str(paths.CHECKPOINTS)
    save_every: int = 50

    def resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        return "vllm" if torch.cuda.is_available() else "hf"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def resolve_dtype(self):
        if self.resolve_device() == "cpu":
            return torch.float32   # bf16 on CPU is slow/unsupported for many ops
        return getattr(torch, self.dtype)
