"""Held-out evaluation for the teacher policy.

Training reward tells you the teacher got better at the problems it was trained
on. This suite asks the harder question: does the hint actually teach a frozen
student problems it has never been tutored on? Every metric here exists to close
one specific hole in that claim:

  baseline_acc    student alone. Without it "teacher_acc = 60%" is meaningless -
                  the held-out set could just be easy.
  teacher_acc     student + the teacher's generated hint. The headline number.
  oracle_acc      student + the dataset's oracle fact. The CEILING: no hint can
                  do better than handing the student the exact fact it lacked,
                  so this bounds what "good teaching" can possibly buy.
  teaching_gain   teacher_acc - baseline_acc. The thing we are actually training.
  pct_of_ceiling  teaching_gain / (oracle_acc - baseline_acc). Normalizes the
                  gain by how much headroom this split even has, so numbers stay
                  comparable across different problem sets.
  leak_rate       reused from rewards.leaked_answer. The dominant reward hack is
                  to state the answer instead of teaching; a high teaching_gain
                  next to a high leak_rate is a hacked policy, not a tutor.
  mean_hint_len   the second hack is verbosity. Cheap to log, easy to correlate.
  transfer_acc    the subtle hack: hints that help regardless of the question
                  ("read carefully, eliminate wrong options"). We feed the
                  student the hint written for problem A while asking problem B.
                  A NEAR-ZERO transfer_gain is EXPECTED AND HEALTHY - it means
                  the hint is problem-specific teaching. A LARGE transfer_gain
                  means the "hint" is generic filler / a prompt-format effect
                  that would score well on the training reward without teaching
                  anything, so treat it as a red flag on teacher_acc.

The teacher is any callable `teacher_fn(question, choices) -> hint`. `StubTeacher`
plus `StubStudent` make the whole suite runnable on a laptop with no GPU and no
model download (`--stub`), which is how you smoke the plumbing before burning
cluster time.

    python evals.py --stub --limit 20
    python evals.py --problems data/zpd_problems.jsonl --teacher hf \
        --teacher-model Qwen/Qwen2.5-3B-Instruct --out runs/eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
from typing import Callable, Protocol, Sequence

import paths
from rewards import leaked_answer

# teacher_fn(question, choices) -> hint text
TeacherFn = Callable[[str, Sequence[str]], str]

DEFAULT_PROBLEMS = str(paths.DATA / "zpd_problems.jsonl")
SYNTHETIC_N = 200


class Chooser(Protocol):
    """The slice of the Student contract multiple choice scoring needs."""

    def choose(self, question: str, choices: Sequence[str], hint: str = "") -> int:
        ...


# ------------------------------------------------------------------ teachers

class StubTeacher:
    """Model-free teacher: one fixed, question-independent string.

    Deliberately generic, so a stub run also demonstrates what the transfer check
    is for: this "hint" carries no problem-specific information, so any gain it
    produces must show up in transfer_gain too.
    """

    DEFAULT = ("Recall the underlying science fact this question depends on, then "
               "rule out the choices that contradict it.")

    def __init__(self, text: str = DEFAULT):
        self.text = text

    def __call__(self, question: str, choices: Sequence[str]) -> str:
        return self.text


class OracleTeacher:
    """Upper bound as a teacher_fn: replays the dataset hint for each question.

    Useful as a control - running with `--teacher oracle` must reproduce
    oracle_acc exactly, which is a cheap self-test of the harness.
    """

    def __init__(self, items: Sequence[dict]):
        self._by_question = {it["question"]: it.get("hint", "") for it in items}

    def __call__(self, question: str, choices: Sequence[str]) -> str:
        return self._by_question.get(question, "")


class HFTeacher:
    """Real policy: a causal LM prompted to emit one short hint. Needs torch."""

    PROMPT = ("You are a tutor. A student is about to answer this multiple-choice "
              "science question. Give ONE short hint (max 25 words) that supplies the "
              "fact they are missing. Do NOT name or reveal the correct choice.\n\n"
              "Question: {question}\nChoices: {choices}\n\nHint:")

    def __init__(self, name: str, device: str = "auto", max_new_tokens: int = 48,
                 adapter: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def __call__(self, question: str, choices: Sequence[str]) -> str:
        prompt = self.PROMPT.format(question=question, choices=" | ".join(choices))
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens,
                                      do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        text = self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip().split("\n")[0].strip()


# ------------------------------------------------------------------ data

def synthetic_problems(n: int = SYNTHETIC_N) -> list[dict]:
    """Fallback set so the suite is testable with no data file and no models.

    gold_idx is aligned with StubStudent's internal notion of the answer, which
    makes these behave like real ZPD items (student fails alone, the hint can
    rescue it) instead of pure noise. Absolute stub numbers therefore depend on
    PYTHONHASHSEED; the plumbing they exercise does not.
    """
    items = []
    for i in range(n):
        q = f"synthetic science question {i}?"
        choices = [f"choice {i}-{c}" for c in "abcd"]
        items.append({
            "question": q,
            "choices": choices,
            "gold_idx": hash(q) % len(choices),
            "hint": f"the relevant fact for synthetic question {i} is fact-{i}",
        })
    return items


def load_problems(path: str = DEFAULT_PROBLEMS) -> list[dict]:
    """Read the ZPD jsonl; fall back to a synthetic set if it is not there.

    The real file lives on the cluster (731 rows) and is often absent locally, so
    a missing file is a warning, not a crash.
    """
    if not os.path.exists(path):
        print(f"[evals] {path} not found - falling back to {SYNTHETIC_N} synthetic problems")
        return synthetic_problems()
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("choices") and row.get("gold_idx") is not None:
                items.append(row)
    print(f"[evals] loaded {len(items)} problems from {path}")
    return items


def split_problems(items: Sequence[dict], test_frac: float = 0.15,
                   seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Deterministic train/test split.

    Fixed seed on a fixed-order shuffle: the held-out set is the same set on
    every run and every machine, so eval numbers from different checkpoints are
    actually comparable.
    """
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n_test = min(len(items), max(1, round(len(items) * test_frac))) if items else 0
    test = [items[i] for i in idx[:n_test]]
    train = [items[i] for i in idx[n_test:]]
    return train, test


# ------------------------------------------------------------------ eval

def _gold_text(item: dict) -> str:
    return str(item["choices"][int(item["gold_idx"])])


def evaluate_teacher(items: Sequence[dict], student: Chooser, teacher_fn: TeacherFn,
                     verbose: bool = False) -> dict:
    """Score one teacher on `items`. Returns the metrics dict (JSON-safe)."""
    n = len(items)
    if n == 0:
        raise ValueError("no problems to evaluate")

    hints = [str(teacher_fn(it["question"], it["choices"]) or "") for it in items]

    baseline, teacher, oracle = [], [], []
    leaks, lengths, records = [], [], []
    for it, hint in zip(items, hints):
        gold = int(it["gold_idx"])
        b = int(student.choose(it["question"], it["choices"]) == gold)
        t = int(student.choose(it["question"], it["choices"], hint=hint) == gold)
        o = int(student.choose(it["question"], it["choices"],
                               hint=str(it.get("hint", ""))) == gold)
        leak = float(leaked_answer(hint, [_gold_text(it)]))
        baseline.append(b)
        teacher.append(t)
        oracle.append(o)
        leaks.append(leak)
        lengths.append(len(hint.split()))
        records.append({"question": it["question"], "gold": _gold_text(it),
                        "hint": hint, "baseline_correct": b, "teacher_correct": t,
                        "oracle_correct": o, "leak": leak})
        if verbose:
            print(f"  [{'ok ' if t else 'x  '}] {it['question'][:60]!r} -> {hint[:80]!r}")

    # TRANSFER: ask problem i while showing the hint written for a DIFFERENT
    # problem. Pairing i -> i+1 (mod n) is a fixed derangement, so no item ever
    # sees its own hint. Near-zero transfer_gain is the healthy outcome; a large
    # one means the hints are generic and teacher_acc is inflated by format, not
    # teaching.
    transfer = None
    if n > 1:
        transfer = []
        for i, it in enumerate(items):
            other = hints[(i + 1) % n]
            transfer.append(int(student.choose(it["question"], it["choices"],
                                               hint=other) == int(it["gold_idx"])))
            records[i]["transfer_correct"] = transfer[-1]

    baseline_acc = st.fmean(baseline)
    teacher_acc = st.fmean(teacher)
    oracle_acc = st.fmean(oracle)
    headroom = oracle_acc - baseline_acc
    teaching_gain = teacher_acc - baseline_acc

    return {
        "n_items": n,
        "baseline_acc": baseline_acc,
        "teacher_acc": teacher_acc,
        "oracle_acc": oracle_acc,
        "teaching_gain": teaching_gain,
        "oracle_gain": headroom,
        # guard: on a split with no headroom the ratio is undefined, not 0 or inf
        "pct_of_ceiling": (teaching_gain / headroom) if abs(headroom) > 1e-9 else None,
        "leak_rate": st.fmean(leaks),
        "mean_hint_len": st.fmean(lengths),
        "empty_hint_rate": st.fmean(1.0 if not h.strip() else 0.0 for h in hints),
        "transfer_acc": st.fmean(transfer) if transfer else None,
        "transfer_gain": (st.fmean(transfer) - baseline_acc) if transfer else None,
        "records": records,
    }


# ------------------------------------------------------------------ reporting

def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.2%}"


def _signed_pct(x) -> str:
    return "n/a" if x is None else f"{x:+.2%}"


def print_report(m: dict) -> None:
    rows = [
        ("items", str(m["n_items"]), ""),
        ("baseline_acc", _pct(m["baseline_acc"]), "student alone"),
        ("teacher_acc", _pct(m["teacher_acc"]), "student + teacher hint"),
        ("oracle_acc", _pct(m["oracle_acc"]), "student + dataset hint (ceiling)"),
        ("teaching_gain", _signed_pct(m["teaching_gain"]), "teacher_acc - baseline_acc"),
        ("pct_of_ceiling", _pct(m["pct_of_ceiling"]), "gain / available headroom"),
        ("leak_rate", f"{m['leak_rate']:.3f}", "gold text stated in the hint"),
        ("mean_hint_len", f"{m['mean_hint_len']:.1f}", "words"),
        ("empty_hint_rate", _pct(m["empty_hint_rate"]), "teacher said nothing"),
        ("transfer_acc", _pct(m["transfer_acc"]), "other problem's hint"),
        ("transfer_gain", _signed_pct(m["transfer_gain"]), "near zero is healthy"),
    ]
    print("\n=== teacher eval (held-out) ===")
    for name, value, note in rows:
        note = f"  ({note})" if note else ""
        print(f"{name:<16} : {value:>9}{note}")

    gain, transfer_gain = m["teaching_gain"], m["transfer_gain"]
    if m["leak_rate"] > 0.05:
        print("\n[WARNING] the teacher is stating gold answer text - teaching_gain is "
              "leakage, not teaching.")
    if transfer_gain is not None and gain > 0.01 and transfer_gain > 0.5 * gain:
        print("\n[WARNING] most of the gain survives swapping hints between problems - "
              "the hints look like generic filler rather than problem-specific teaching.")
    if m["pct_of_ceiling"] is None:
        print("\n[WARNING] oracle_acc == baseline_acc on this split: no headroom, so "
              "pct_of_ceiling is undefined and teaching_gain is hard to interpret.")


def write_results(path: str, metrics: dict, meta: dict, max_records: int = 25) -> None:
    payload = {**meta, **{k: v for k, v in metrics.items() if k != "records"},
               "samples": metrics["records"][:max_records]}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[evals] results -> {path}")


# ------------------------------------------------------------------ cli

def build_student(stub: bool, model: str, device: str):
    # imported lazily: zpd_filter pulls in torch, and --stub must never need it
    if stub:
        from zpd_filter import StubStudent

        return StubStudent()
    import torch
    from zpd_filter import HFStudent

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    return HFStudent(model, device=device, dtype=dtype)


def build_teacher(kind: str, items: Sequence[dict], model: str, device: str,
                  adapter: str | None) -> TeacherFn:
    if kind == "stub":
        return StubTeacher()
    if kind == "oracle":
        return OracleTeacher(items)
    return HFTeacher(model, device=device, adapter=adapter)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a teacher policy on held-out problems.")
    ap.add_argument("--problems", default=DEFAULT_PROBLEMS,
                    help="ZPD jsonl; synthetic problems are used if it is missing")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap on held-out items evaluated (0 = all); applied after the split")
    ap.add_argument("--stub", action="store_true",
                    help="StubStudent + StubTeacher: no models, no GPU, no downloads")
    ap.add_argument("--out", default=str(paths.RUNS / "eval_results.json"))
    ap.add_argument("--teacher", choices=["stub", "hf", "oracle"], default=None,
                    help="teacher under test; defaults to 'stub' with --stub, else 'hf'")
    ap.add_argument("--teacher-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--teacher-adapter", default=None, help="LoRA adapter dir for the teacher")
    ap.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0, help="split seed; keep fixed across runs")
    ap.add_argument("--split", choices=["test", "train", "all"], default="test")
    ap.add_argument("--verbose", action="store_true", help="print every hint")
    args = ap.parse_args()

    items = load_problems(args.problems)
    if not items:
        raise SystemExit(f"[evals] no usable problems in {args.problems}")

    train, test = split_problems(items, test_frac=args.test_frac, seed=args.seed)
    chosen = {"test": test, "train": train, "all": list(items)}[args.split]
    if args.limit and args.limit > 0:
        chosen = chosen[: args.limit]
    print(f"[evals] split={args.split} evaluating {len(chosen)} of "
          f"{len(items)} problems (test set = {len(test)}, seed={args.seed})")

    student = build_student(args.stub, args.student_model, args.device)
    teacher_kind = args.teacher or ("stub" if args.stub else "hf")
    teacher_fn = build_teacher(teacher_kind, items, args.teacher_model, args.device,
                               args.teacher_adapter)

    metrics = evaluate_teacher(chosen, student, teacher_fn, verbose=args.verbose)
    print_report(metrics)
    write_results(args.out, metrics, {
        "problems": args.problems,
        "split": args.split,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "stub": args.stub,
        "teacher": teacher_kind,
        "teacher_model": None if teacher_kind != "hf" else args.teacher_model,
        "teacher_adapter": args.teacher_adapter,
        "student_model": "stub" if args.stub else args.student_model,
    })


if __name__ == "__main__":
    main()
