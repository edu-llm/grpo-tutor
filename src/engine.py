"""Inference engines for the policy (teacher).

Three implementations behind one interface:
  StubEngine  - deterministic fake generations. No GPU, no models. Used by the
                toy path to validate orchestration/monitoring end-to-end.
  HFEngine    - transformers .generate(). Small-scale / Mac dev.
  VLLMEngine  - production path. Batched group generation, prefix caching, and
                the colocated lifecycle (sleep/wake) needed to share 1 H100 with
                the trainer.

COLOCATION (single GPU)
-----------------------
Trainer and engine share one H100, so the default is to ALTERNATE:
    wake -> generate rollouts -> sleep -> train N epochs -> sync weights -> wake
`sleep()` offloads engine KV cache/weights so the optimizer has room.
In practice an 80GB H100 CAN hold both a 3B teacher (LoRA) and the engine at
once - `--no-sleep` keeps it resident and skips the ~0.3s cycle. Alternating is
the safe default that also holds for bigger teachers or a tighter card.

Generating while the engine is asleep does NOT fail cleanly: it dies inside
LoRA activation with `CUDA error: invalid argument`. Every generate() call in
train_h100.py goes through its awake() context manager for that reason.

SPEED NOTES (the things that actually matter)
  - ask for all K group samples in ONE call (n=K) - one prefill, K decodes
  - enable prefix caching: the K samples share the whole lesson prompt
  - keep max_new_tokens tight; generation dominates wall-clock
  - sync weights as rarely as correctness allows (see cfg.sync_every)
"""

from __future__ import annotations

import hashlib
import os

from interfaces import Completion


class StubEngine:
    """Deterministic fake engine - no models, no GPU. For toy/CI runs.

    Generations are a hash-derived function of (prompt, sample index) so runs are
    reproducible, and token ids/logprobs are synthetic but well-formed.
    """

    # word pool deliberately overlaps FakeReward's keywords so the toy loop has
    # real reward variance (otherwise advantages are all zero and nothing is tested)
    WORDS = ("step", "because", "first", "check", "example", "then", "number",
             "try", "look", "the", "answer", "so", "next", "value")

    def __init__(self, vocab: int = 128, reply_words: int = 24):
        self.vocab = vocab
        self.reply_words = reply_words
        self.n_sync = 0
        self.asleep = False

    def generate(self, prompts, n, max_new_tokens, temperature):
        out = []
        for p in prompts:
            samples = []
            for i in range(n):
                seed = int(hashlib.md5(f"{p}|{i}|{self.n_sync}".encode()).hexdigest(), 16)
                length = 4 + (seed % max(1, min(self.reply_words, max_new_tokens)))
                words = [self.WORDS[(seed >> (3 * b)) % len(self.WORDS)] for b in range(length)]
                ids = [(seed >> (b % 24)) % self.vocab for b in range(length)]
                lps = [-((seed >> b) % 300) / 100.0 - 0.01 for b in range(length)]
                samples.append(Completion(text=" ".join(words), token_ids=ids, logprobs=lps))
            out.append(samples)
        return out

    def sync_weights(self, source=None):
        self.n_sync += 1

    def sleep(self):
        self.asleep = True

    def wake(self):
        self.asleep = False


class HFEngine:
    """transformers .generate() backend (wraps generation.HFGenerator)."""

    def __init__(self, model, tokenizer, device: str):
        from generation import HFGenerator

        self._gen = HFGenerator(model, tokenizer, device)

    def generate(self, prompts, n, max_new_tokens, temperature):
        raw = self._gen.generate(prompts, n=n, max_new_tokens=max_new_tokens, temperature=temperature)
        return [
            [Completion(text=g.text, token_ids=g.token_ids, logprobs=g.logprobs) for g in per_prompt]
            for per_prompt in raw
        ]

    def sync_weights(self, source=None):
        # HF engine holds the live trainer module - updates are visible immediately.
        pass

    def sleep(self):
        pass

    def wake(self):
        pass


class VLLMEngine:
    """vLLM backend with colocated sleep/wake + LoRA weight sync.

    NOTE: vLLM's sleep-mode and weight-update APIs move between versions. This
    uses the portable path (sleep_mode + LoRARequest adapter reload) and degrades
    gracefully if the installed version lacks sleep support. Verify on your H100.
    """

    def __init__(self, model_name: str, dtype: str = "bfloat16", max_lora_rank: int = 16,
                 gpu_memory_utilization: float = 0.45, enable_prefix_caching: bool = True,
                 colocated: bool = True):
        import vllm
        from vllm.lora.request import LoRARequest

        self._LoRARequest = LoRARequest
        kwargs = dict(
            model=model_name,
            dtype=dtype,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
        )
        if colocated:
            # lets us hand VRAM back to the trainer between phases (single-GPU only)
            kwargs["enable_sleep_mode"] = True

        self.llm = vllm.LLM(**kwargs)
        self.SamplingParams = vllm.SamplingParams
        self._lora_req = None
        self._version = 0
        self._supports_sleep = hasattr(self.llm, "sleep") and colocated

    def generate(self, prompts, n, max_new_tokens, temperature):
        params = self.SamplingParams(
            n=n,                       # all K group samples in one call
            temperature=max(temperature, 1e-5),
            top_p=1.0,
            max_tokens=max_new_tokens,
            logprobs=0,                # logprob of each sampled token
        )
        outs = self.llm.generate(prompts, params, lora_request=self._lora_req)
        results = []
        for out in outs:
            samples = []
            for comp in out.outputs:
                ids = list(comp.token_ids)
                lps = [lp[t].logprob for lp, t in zip(comp.logprobs, ids)] if comp.logprobs else []
                samples.append(Completion(text=comp.text, token_ids=ids, logprobs=lps))
            results.append(samples)
        return results

    def sync_weights(self, source):
        """Hand vLLM the trainer's fresh LoRA adapter.

        The adapter is written to a RAM disk (/dev/shm) rather than the working
        directory: on a cluster the CWD is usually a network filesystem, so syncing
        there sends ~35MB over the wire every few steps. /dev/shm is local memory,
        so this is effectively an in-memory handoff. Override with LORA_SYNC_DIR.
        """
        import shutil

        self._version += 1
        base = os.environ.get("LORA_SYNC_DIR") or ("/dev/shm" if os.path.isdir("/dev/shm") else ".")
        adapter_dir = os.path.join(base, f"lora_sync_{os.getpid()}", f"v{self._version}")
        source.save_pretrained(adapter_dir)
        self._lora_req = self._LoRARequest(f"teacher-{self._version}", self._version, adapter_dir)
        # keep only the newest adapter so /dev/shm doesn't fill up over a long run
        prev = os.path.join(base, f"lora_sync_{os.getpid()}", f"v{self._version - 1}")
        shutil.rmtree(prev, ignore_errors=True)

    def sleep(self):
        if self._supports_sleep:
            self.llm.sleep(level=1)

    def wake(self):
        if self._supports_sleep:
            self.llm.wake_up()


# SGLang: same interface would wrap sgl.Engine(...).generate(...). SGLang has
# comparable throughput; vLLM is the default here only because its RLHF
# weight-sync / sleep-mode story is more established. Swapping is a ~40-line
# class implementing the same four methods.


def build_engine(cfg, model=None, tokenizer=None):
    backend = cfg.resolve_backend()
    if backend == "stub":
        return StubEngine()
    if backend == "vllm":
        return VLLMEngine(cfg.teacher_model, dtype=cfg.dtype, max_lora_rank=cfg.lora_r,
                          gpu_memory_utilization=getattr(cfg, "gpu_mem_util", 0.45),
                          colocated=not getattr(cfg, "no_sleep", False))
    return HFEngine(model, tokenizer, cfg.resolve_device())
