"""HuggingFace `.generate()` sampling, used by `engine.HFEngine`.

This is the small-scale / no-vLLM path (CPU, MPS, or a single GPU without an
inference server). The production path is `engine.VLLMEngine`.

IMPORTANT (consistency): behavior ("old") log-probs come from a clean forward
pass, NOT from warped sampling scores, so that the "old" and "new" log-probs in
the GRPO ratio are computed the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Generation:
    token_ids: list[int]     # generated tokens only (prompt excluded)
    logprobs: list[float]    # behavior log-prob per generated token
    text: str


class HFGenerator:
    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tok = tokenizer
        self.device = device
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    @torch.no_grad()
    def generate(self, prompts, n, max_new_tokens, temperature):
        results: list[list[Generation]] = []
        eos_id, pad_id = self.tok.eos_token_id, self.tok.pad_token_id
        for prompt in prompts:
            enc = self.tok(prompt, return_tensors="pt").to(self.device)
            prompt_len = enc.input_ids.shape[1]
            out = self.model.generate(
                **enc,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=pad_id,
                return_dict_in_generate=True,
            )
            seqs = out.sequences
            logp = self._sequence_logprobs(seqs, prompt_len)

            samples, gen = [], seqs[:, prompt_len:]
            for i in range(gen.shape[0]):
                ids, lps = _trim_at_eos(gen[i].tolist(), logp[i].tolist(), eos_id, pad_id)
                samples.append(Generation(ids, lps, self.tok.decode(ids, skip_special_tokens=True)))
            results.append(samples)
        return results

    @torch.no_grad()
    def _sequence_logprobs(self, seqs, prompt_len):
        attn = (seqs != self.tok.pad_token_id).long()
        logits = self.model(input_ids=seqs, attention_mask=attn).logits[:, :-1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(-1, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
        return tok_lp[:, prompt_len - 1:]


def _trim_at_eos(ids, logprobs, eos_id, pad_id):
    """Cut at the first EOS (kept) and drop trailing pad tokens."""
    out_ids, out_lp = [], []
    for tok, lp in zip(ids, logprobs):
        if tok == pad_id and tok != eos_id:
            break
        out_ids.append(tok)
        out_lp.append(lp)
        if tok == eos_id:
            break
    return out_ids, out_lp
