"""Find problems in the student's Zone of Proximal Development.

The whole tutoring reward only has a gradient on problems where the student
FAILS ALONE but SUCCEEDS WITH HELP. Too easy -> ceiling, reward 0. Too hard ->
floor, reward 0. This script measures that band and writes out the usable set.

Uses OpenBookQA, which ships `fact1` - the science fact needed to answer. That's
an ideal ORACLE HINT: genuinely helpful, but not the answer itself (so a "solved
with help" here really does mean teachable, not leaked).

Multiple choice is scored by comparing the length-normalized log-prob of each
choice - deterministic and far more reliable for small models than parsing free
text.

    python zpd_filter.py --limit 200                     # needs a GPU-ish box
    python zpd_filter.py --limit 20 --stub               # no model: smoke the logic
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import paths


class StubStudent:
    """Deterministic fake student: gets it right only when the hint is present
    for ~half of items. Lets us validate the filter logic with no model."""

    def reply(self, dialogues, max_new_tokens: int = 48):
        return [f"i think i get part of it but im stuck on the {len(d) % 7}th bit"
                for d in dialogues]

    def choose(self, question, choices, hint=""):
        h = (hash(question) % 100) / 100.0
        gold_idx = hash(question) % len(choices)
        if hint and h < 0.5:
            return gold_idx           # hint rescues it
        return (gold_idx + 1) % len(choices)   # wrong without help


class HFStudent:
    """Small frozen model; answers MC by length-normalized choice log-prob."""

    def __init__(self, name: str, device: str = "cuda", dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
        self.model.eval()
        for p in self.model.parameters():   # the student is the ENVIRONMENT: never trained
            p.requires_grad_(False)
        self.device = device

    STUDENT_SYSTEM = (
        "You are a 7th grader talking to your tutor about a question you're stuck on.\n"
        "Rules:\n"
        "1. ONE short sentence, usually under 15 words.\n"
        "2. Sound like a kid: casual, unsure, plain words.\n"
        "3. SAY what CONFUSES you, try a half-formed idea, or ask something back.\n"
        "4. NEVER EXPLAIN like a textbook and never state a confident fact.\n"
        "5. No lists, no formal vocabulary, no definitions."
    )

    @staticmethod
    def _trim_to_sentence(text: str) -> str:
        """Drop a trailing half-sentence left by the token cap.

        A truncated student turn ends the transcript mid-sentence, and the
        teacher's next turn then CONTINUES that sentence instead of taking its
        own. Observed on GPU: a student turn cut at "...why people still" was
        completed by the tutor with the gold answer, scoring as a leak that the
        tutor never really chose to commit.
        """
        t = text.strip()
        cut = max(t.rfind("."), t.rfind("!"), t.rfind("?"))
        # only trim when a sentence boundary is far enough in; short bare
        # replies ("Rain") have no punctuation and should survive untouched
        return t[: cut + 1] if cut >= 20 else t

    @torch.no_grad()
    def reply(self, dialogues, max_new_tokens: int = 80):
        """Batched free-text student turns (the environment's side of the dialogue)."""
        texts = []
        for d in dialogues:
            msgs = [{"role": "system", "content": self.STUDENT_SYSTEM},
                    {"role": "user", "content": d}]
            texts.append(self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        enc = self.tok(texts, return_tensors="pt", padding=True,
                       padding_side="left").to(self.device)
        out = self.model.generate(**enc, do_sample=True, temperature=0.8, top_p=0.95,
                                  max_new_tokens=max_new_tokens,
                                  pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc.input_ids.shape[1]:]
        return [self._trim_to_sentence(self.tok.decode(g, skip_special_tokens=True))
                for g in gen]

    @torch.no_grad()
    def choose(self, question, choices, hint=""):
        head = f"Fact: {hint}\n" if hint else ""
        prompt = f"{head}Question: {question}\nAnswer:"
        p_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        scores = []
        for ch in choices:
            c_ids = self.tok(" " + ch, add_special_tokens=False,
                             return_tensors="pt").input_ids.to(self.device)
            full = torch.cat([p_ids, c_ids], dim=1)
            logits = self.model(full).logits[:, :-1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            tgt = full[:, 1:]
            tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
            choice_lp = tok_lp[p_ids.shape[1] - 1:]          # only the choice tokens
            scores.append(choice_lp.mean().item())           # length-normalized
        return int(max(range(len(scores)), key=lambda i: scores[i]))


FREE_TEXT_ASK = ("Question you're stuck on:\n{q}\n{choices}\n\n"
                 "Just say which option you think it is and why, in one sentence.")


def _format_choices(choices) -> str:
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))


def _names_gold(text: str, gold: str, distractors) -> bool:
    """Gold mentioned and no distractor mentioned, so a scattershot reply that
    lists several options does not count as knowing the answer."""
    low = text.lower()
    g = gold.lower().strip()
    if not g or g not in low:
        return False
    return not any(d.lower().strip() and d.lower().strip() in low for d in distractors)


def load_openbookqa(limit: int):
    from datasets import load_dataset

    ds = load_dataset("allenai/openbookqa", "additional", split="train")
    items = []
    for row in ds:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        if row["answerKey"] not in labels:
            continue
        items.append({
            "question": row["question_stem"],
            "choices": texts,
            "gold_idx": labels.index(row["answerKey"]),
            "hint": row.get("fact1", ""),
        })
        if len(items) >= limit:
            break
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--stub", action="store_true", help="no model; validate the logic")
    ap.add_argument("--no-free-text-screen", action="store_true",
                    help="keep items the student can already answer in free text "
                         "(the old, choose()-only criterion)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        # a --stub smoke test must never overwrite the real curated set
        args.out = str(paths.DATA / ("zpd_stub.jsonl" if args.stub
                                     else "zpd_problems.jsonl"))

    if args.stub:
        items = [{"question": f"toy question {i}?", "choices": ["a", "b", "c", "d"],
                  "gold_idx": i % 4, "hint": f"fact {i}"} for i in range(args.limit)]
        student = StubStudent()
    else:
        items = load_openbookqa(args.limit)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        student = HFStudent(args.student, device=device)

    # Free-text screen: choose() compares option log-probs with no room to reason,
    # so "fails alone" under it is not the same as "does not know". Measured on the
    # old set: 25% of curated items were answered correctly in free text. Keeping
    # those trains the tutor to teach things the student already knows, and lets it
    # blurt the answer mid-dialogue and score itself right.
    free_ok = [False] * len(items)
    if not args.stub and not args.no_free_text_screen:
        views = [FREE_TEXT_ASK.format(q=it["question"],
                                      choices=_format_choices(it["choices"]))
                 for it in items]
        replies = []
        for i in range(0, len(views), 32):
            replies.extend(student.reply(views[i:i + 32], max_new_tokens=60))
        for i, (it, rep) in enumerate(zip(items, replies)):
            gold = it["choices"][it["gold_idx"]]
            distractors = [c for j, c in enumerate(it["choices"]) if j != it["gold_idx"]]
            free_ok[i] = _names_gold(rep, gold, distractors)

    kept, n_base, n_help, n_free = [], 0, 0, 0
    for it, knows in zip(items, free_ok):
        alone = student.choose(it["question"], it["choices"]) == it["gold_idx"]
        helped = student.choose(it["question"], it["choices"], hint=it["hint"]) == it["gold_idx"]
        n_base += int(alone)
        n_help += int(helped)
        n_free += int(knows)
        if (not alone) and helped and not knows:      # <- the ZPD band, both channels
            kept.append({**it, "baseline_correct": False, "assisted_correct": True,
                         "free_text_correct": False})

    n = max(1, len(items))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for k in kept:
            f.write(json.dumps(k) + "\n")

    print("=== ZPD probe ===")
    print(f"items probed         : {len(items)}")
    print(f"baseline accuracy    : {n_base / n:.2%}   (student alone, choose())")
    print(f"assisted accuracy    : {n_help / n:.2%}   (student + oracle hint)")
    print(f"teaching gain        : {(n_help - n_base) / n:+.2%}")
    print(f"free-text correct    : {n_free / n:.2%}   (knows it when simply asked)")
    print(f"ZPD items kept       : {len(kept)} ({len(kept) / n:.1%}) -> {args.out}")
    if len(kept) / n < 0.05:
        print("\n[WARNING] almost no ZPD headroom. The tutoring reward will have "
              "little/no gradient. Try: an easier/harder problem set, or a "
              "different-sized student.")


if __name__ == "__main__":
    main()
