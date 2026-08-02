"""Learned teaching-quality score: frozen backbone, linear head, last position.

Trained on 1,104 tutor turns rated 1-5 for how much they would help a student
learn (data/rm_dataset.json, see docs/leak_calibration.md for the sibling leak
work). Used as a DENSE term in the GRPO reward, alongside - never instead of -
the outcome and the leak rule.

Three choices worth stating, because each was measured rather than assumed:

FROZEN backbone. GRPO moves the teacher's LoRA every step. A head reading
intermediate activations would be scoring a moving target; the last layer of a
frozen trunk is fixed for the whole run.

LAST POSITION. Attention is causal, so only the final token has read the entire
tutor message. Mean-pooling biases towards early tokens.

LINEAR head, not an MLP. The MLP reached train rho 0.976 against test 0.55 -
memorisation - and its AUC varied 0.87-0.94 across splits. The linear probe holds
AUC 0.932 with a third of the variance, and matches the MLP on the metric that
actually matters (same-question pairwise ranking, 0.72 vs 0.74, inside the noise
of 366 pairs). It also extrapolates predictably: RL will push the policy into
regions the head never saw, and a single direction in embedding space degrades
gracefully there where an MLP can have arbitrary maxima to climb.
"""

from __future__ import annotations

import os


class TeachingScorer:
    """Scores tutor turns for teaching quality.

    The output is an uncalibrated RANKING score, not a 1-5 prediction: fitting
    the absolute scale cost AUC (0.933 -> 0.901) and buys nothing, because the
    reward z-scores within the group and only the ordering survives.
    """

    def __init__(self, head_path: str, device: str = "cuda", batch: int = 16,
                 max_length: int = 1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not os.path.exists(head_path):
            raise SystemExit(f"reward-model head not found: {head_path}\n"
                             f"train it with: python src/train_rm.py --hidden 0")
        blob = torch.load(head_path, map_location="cpu")
        self.mu, self.sd = blob["mu"], blob["sd"]
        self.backbone_name = blob["backbone"]
        self.device, self.batch, self.max_length = device, batch, max_length

        self.tok = AutoTokenizer.from_pretrained(self.backbone_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.backbone_name, output_hidden_states=True).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        d = self.mu.shape[1]
        self.head = torch.nn.Linear(d, 2)
        self.head.load_state_dict(blob["head"])
        self.head.to(device).eval()
        self.torch = torch

    def view(self, question: str, gold: str, transcript: str, tutor_text: str) -> str:
        """Exactly the view the raters had. A different framing here would score
        text the head was never fitted on."""
        return (f"Question: {question}\n"
                f"Correct answer: {gold}\n"
                f"Conversation so far:\n{transcript.strip() or '(none)'}\n"
                f"Tutor message to rate: {tutor_text.strip()}")

    def score(self, views: list[str]) -> list[float]:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(views), self.batch):
                enc = self.tok(views[i:i + self.batch], return_tensors="pt",
                               padding=True, truncation=True,
                               max_length=self.max_length,
                               padding_side="left").to(self.device)
                hs = self.model(**enc).hidden_states[-1][:, -1, :].float()
                z = (hs.cpu() - self.mu) / self.sd
                out.extend(self.head(z.to(self.head.weight.device))[:, 1].tolist())
        return out


def group_normalize(values: list[float]) -> list[float]:
    """Z-score within the group.

    GRPO centres advantages inside a group anyway, so only the SPREAD of this
    term across one problem's completions survives. Feeding raw 1-5 values lets a
    problem the head happens to score high everywhere dominate the batch, which
    is between-problem variance the algorithm was never going to use.
    """
    n = len(values)
    if n < 2:
        return [0.0] * n
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n
    sd = var ** 0.5
    if sd < 1e-6:
        return [0.0] * n
    return [(v - m) / sd for v in values]
