"""Find problems in the student's Zone of Proximal Development.

The whole tutoring reward only has a gradient on problems where the student
FAILS ALONE but SUCCEEDS WITH HELP. Too easy -> ceiling, reward 0. Too hard ->
floor, reward 0. This script measures that band and writes out the usable set.

Every supported corpus ships an ORACLE HINT - the fact the item turns on. It has
to be helpful WITHOUT being the answer, or "solved with help" degenerates into
"can copy the answer out of the hint". That is not hypothetical: 28.2% of the
hints in the set run v0 trained on contain their answer, against 10.1% in the
pool they were drawn from, because this screen prefers them. The `*_honest`
sources drop those items up front - see SOURCES. The pools come from
`benchmarks.py` so there is one loader per corpus.

Multiple choice is scored by comparing the length-normalized log-prob of each
choice - deterministic and far more reliable for small models than parsing free
text.

    python zpd_filter.py --limit 5000                        # openbookqa_honest
    python zpd_filter.py --limit 25000 --source race_middle  # the fallback corpus
    python zpd_filter.py --limit 20 --stub                   # no model: smoke the logic
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re

import torch

import paths
import seeding


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

    def choose_free(self, question, choices, hint=""):
        """Agrees with choose() most of the time and not always - the stub is here
        so the free-answer plumbing runs with no GPU, and a channel that always
        agreed would make `--student-answer-mode free` untestable."""
        idx = self.choose(question, choices, hint=hint)
        if hash(question) % 4 == 0:
            return (idx + 1) % len(choices)
        return idx

    def set_answer_mode(self, mode: str) -> None:
        self.answer_mode = mode

    def answer(self, question, choices, hint=""):
        if getattr(self, "answer_mode", "logprob") == "free":
            return self.choose_free(question, choices, hint=hint)
        return self.choose(question, choices, hint=hint)


class HFStudent:
    """Small frozen model; answers MC by length-normalized choice log-prob.

    An optional persona adapter changes how the student TALKS without changing
    how it SCORES: reply() runs with the adapter active, choose() runs with it
    disabled. choose() is the reward channel, and the ZPD curation and every
    reported baseline assume it is fixed.
    """

    def __init__(self, name: str, device: str = "cuda", dtype=torch.bfloat16,
                 persona_adapter: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
        self.has_persona = False
        if persona_adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, persona_adapter)
            self.has_persona = True
            print(f"[student] persona adapter: {persona_adapter} "
                  f"(reply only; choose() disables it)", flush=True)
        self.model.eval()
        for p in self.model.parameters():   # the student is the ENVIRONMENT: never trained
            p.requires_grad_(False)
        self.device = device

    @contextlib.contextmanager
    def _scoring_weights(self):
        """Base weights, so the persona never moves the reward channel."""
        if self.has_persona:
            with self.model.disable_adapter():
                yield
        else:
            yield

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
    def score_choices(self, question, choices, hint=""):
        """Length-normalized log-prob of each choice as a continuation.

        Exposed separately from choose() because the GAP between the top two
        scores is the only confidence this channel has: argmax alone cannot tell
        a coin flip from knowledge, and the call is deterministic so resampling
        it says nothing.
        """
        head = f"Fact: {hint}\n" if hint else ""
        prompt = f"{head}Question: {question}\nAnswer:"
        p_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        scores = []
        with self._scoring_weights():
            for ch in choices:
                c_ids = self.tok(" " + ch, add_special_tokens=False,
                                 return_tensors="pt").input_ids.to(self.device)
                full = torch.cat([p_ids, c_ids], dim=1)
                logits = self.model(full).logits[:, :-1, :].float()
                logp = torch.log_softmax(logits, dim=-1)
                tgt = full[:, 1:]
                tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
                choice_lp = tok_lp[p_ids.shape[1] - 1:]      # only the choice tokens
                scores.append(choice_lp.mean().item())       # length-normalized
        return scores

    def choose(self, question, choices, hint=""):
        scores = self.score_choices(question, choices, hint=hint)
        return int(max(range(len(scores)), key=lambda i: scores[i]))

    # ---- second answering channel (OFF by default; see set_answer_mode) ----

    FREE_ANSWER_SYSTEM = (
        "You are a student answering a multiple-choice question.\n"
        "Give at most ONE short sentence of reasoning, then a final line of the "
        "form 'Answer: X' where X is the letter of your choice."
    )

    @torch.no_grad()
    def choose_free(self, question, choices, hint="", max_new_tokens: int = 64) -> int:
        """Answer in free text with room to reason, then map back to an option.

        WHY: `choose()` ranks the options by log-prob behind the bare prompt
        `Fact: ...\\nQuestion: ...\\nAnswer:` - no chat template, no persona, no
        tokens to think in. Measured on 120 problems where choose() scored 1%, the
        same student answered 26% correctly when simply asked, so "the student
        cannot solve this" is partly a property of the channel, not the student.

        The prompt keeps choose()'s `Fact:` framing for the hint on purpose: the
        two channels then differ ONLY in how the answer is produced, which is what
        `check_answer_modes.py` is trying to isolate.

        Greedy, and run on the base weights like choose(), so the reward stays a
        function of the problem rather than of the persona or the sampling seed.
        Falls back to choose() when the reply commits to no single option -
        guessing would inject noise straight into the reward.
        """
        head = f"Fact: {hint}\n" if hint else ""
        user = (f"{head}Question: {question}\n{_format_choices(choices)}\n"
                "Which option is it?")
        text = self.tok.apply_chat_template(
            [{"role": "system", "content": self.FREE_ANSWER_SYSTEM},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        with self._scoring_weights():
            out = self.model.generate(**enc, do_sample=False,
                                      max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tok.pad_token_id)
        reply = self.tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        self.free_calls = getattr(self, "free_calls", 0) + 1
        idx = map_reply_to_choice(reply, choices)
        if idx is None:
            self.free_unmapped = getattr(self, "free_unmapped", 0) + 1
            return self.choose(question, choices, hint=hint)
        return idx

    ANSWER_MODES = ("logprob", "free")

    def set_answer_mode(self, mode: str) -> None:
        """Choose which channel `answer()` - the REWARD channel - uses.

        Set at construction time only, and left at "logprob" by default, because
        moving it moves the reward: the 731 curated ZPD problems were selected by
        choose() failing on them, and every baseline in the README (QASC 0.253 ->
        0.893 included) was measured through choose(). Switching to "free" makes
        all of those numbers describe a channel the run no longer uses; the
        curation has to be rebuilt before they mean anything again.
        """
        if mode not in self.ANSWER_MODES:
            raise ValueError(f"answer mode {mode!r} not in {self.ANSWER_MODES}")
        self.answer_mode = mode

    def answer(self, question, choices, hint="") -> int:
        if getattr(self, "answer_mode", "logprob") == "free":
            return self.choose_free(question, choices, hint=hint)
        return self.choose(question, choices, hint=hint)


def student_answer(student, question, choices, hint="") -> int:
    """The reward channel, for students that may predate `answer()`.

    `interfaces.Student` only promises `choose()`, and a teammate's student is
    allowed to implement just that - so the mode switch degrades to the log-prob
    channel instead of crashing the run.
    """
    fn = getattr(student, "answer", None)
    if fn is None:
        return student.choose(question, choices, hint=hint)
    return fn(question, choices, hint=hint)


_ANSWER_LABEL = re.compile(r"answer\s*(?:is|:|=)?\s*[\(\[]?([A-Za-z])([^A-Za-z]|$)",
                           re.IGNORECASE)
_BARE_LETTER = re.compile(r"^\s*[\(\[]?([A-Za-z])[\)\].:]?\s*$")
# a lower-case letter followed by a SPACE is a word, not a label: "the answer is
# a bird" must not resolve to option A
_LABEL_ENDS = {")", "]", ".", ",", ":", ";", "\n", ""}


def _index_of_letter(letter: str, n: int) -> int | None:
    i = ord(letter.lower()) - ord("a")
    return i if 0 <= i < n else None


def map_reply_to_choice(reply: str, choices) -> int | None:
    """Map a free-text answer back to an option index, or None if it commits to none.

    None rather than a guess: the caller falls back to the log-prob channel, so an
    unparseable reply costs measurement coverage instead of injecting a random
    reward. Same reason a reply naming SEVERAL options is refused - that is the
    stance `_names_gold` already takes for the free-text screen.
    """
    if not reply or not choices:
        return None
    n = len(choices)

    for m in _ANSWER_LABEL.finditer(reply):
        letter, nxt = m.group(1), m.group(2)
        if letter.isupper() or nxt in _LABEL_ENDS:
            i = _index_of_letter(letter, n)
            if i is not None:
                return i
    for line in reversed(reply.splitlines()):     # a bare "C." on its own line
        m = _BARE_LETTER.match(line)
        if m:
            i = _index_of_letter(m.group(1), n)
            if i is not None:
                return i
    low = reply.lower()
    hits = [i for i, c in enumerate(choices)
            if str(c).strip().lower() and str(c).strip().lower() in low]
    return hits[0] if len(hits) == 1 else None


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


# --source name -> the benchmarks.py registry entry that supplies the pool.
# Routing through benchmarks.py keeps ONE loader per corpus: the eval sets and the
# curation pool then agree on the schema, the oracle-hint field and the answer-key
# handling by construction rather than by two copies staying in sync.
SOURCES = {"openbookqa_honest": "obqa_train_honest", "openbookqa": "obqa_train",
           "qasc_honest": "qasc_train_honest", "qasc": "qasc_train",
           "race_middle": "race_middle_train"}

# THE SCREEN SELECTS FOR LEAKY HINTS, AND THAT IS THE BUG
#
# "Fails alone AND solves with help" has an easy degenerate solution: the hint
# contains the answer. So this filter preferentially keeps exactly the items
# whose oracle ceiling is unreachable by an honest tutor. Measured with
# `hint_audit.py` on the 549-item set run v0 trained on, against the OpenBookQA
# pool it was drawn from:
#
#                      hints tripping the leak rule   single-word answers
#   obqa_train pool              10.1%                      31%
#   the curated 549              28.2%                      47%
#
# A 2.8x enrichment in leaky hints. Run v0 read specificity ~0 as "OpenBookQA is
# the wrong corpus"; this says the screen manufactured a worse corpus than the
# one it was given. The `*_honest` pools drop the leaky items BEFORE screening,
# which is why they are the defaults here.
#
# Why OpenBookQA and not QASC: QASC's answers are the shortest of any candidate
# (60% a single word vs OpenBookQA's 31%), which is the structure run v0 blamed
# for specificity ~0, and its `combinedfact` states the gold verbatim in 88.5% of
# items - so its headline 0.253 -> 0.893 ceiling is mostly the student copying.
# Measured on the 0.5B over 150 sampled items, all four conditions, log-prob:
#
#   pool                  baseline  +hint   gain    hint-only minus choices-only
#   qasc combinedfact       0.160   0.893  +0.733   +0.647  <- the hint IS the answer
#   qasc fact1 (all)        0.160   0.487  +0.327   +0.253
#   qasc fact1 honest-only  0.180   0.387  +0.207   +0.147
#
# `race_middle` is the fallback if OpenBookQA's honest headroom proves too thin:
# 24,587 items, 4-word median answers, and a hint that points at a passage the
# student can already see. See docs/dataset_choice.md.
DEFAULT_SOURCE = "openbookqa_honest"


def load_source(source: str, limit: int, seed: int = 0):
    """Curation pool for `source`, minus items with no oracle hint.

    An item without one cannot be screened at all: "solves it WITH help" is
    undefined, so it would be dropped by the ZPD test anyway - as an unsolvable
    item rather than as missing data.
    """
    import benchmarks

    items = benchmarks.load_benchmark(SOURCES[source], limit=limit, seed=seed)
    kept = [it for it in items if (it.get("hint") or "").strip()]
    if len(kept) < len(items):
        print(f"[zpd] {len(items) - len(kept)} items dropped: no oracle hint", flush=True)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--source", default=DEFAULT_SOURCE, choices=sorted(SOURCES),
                    help=f"corpus to curate the training set from "
                         f"(default: {DEFAULT_SOURCE})")
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling of the pool + the student's free-text screen")
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

    seeding.seed_everything(args.seed)
    if args.stub:
        items = [{"question": f"toy question {i}?", "choices": ["a", "b", "c", "d"],
                  "gold_idx": i % 4, "hint": f"fact {i}", "source": "stub"}
                 for i in range(args.limit)]
        student = StubStudent()
    else:
        items = load_source(args.source, args.limit, seed=args.seed)
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
    print(f"source               : {'stub' if args.stub else args.source} "
          f"({'-' if args.stub else SOURCES[args.source]})")
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
