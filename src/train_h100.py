"""GRPO training: teacher LLM learns to tutor a frozen student.

Two reward modes:
  --reward fake   gameable keyword reward. A PIPELINE TEST: mean reward SHOULD
                  climb. If it doesn't, generation/advantage/loss/optimizer/sync
                  is broken. Nothing to do with tutoring.
  --reward real   the actual task. Pick a ZPD problem the student cannot solve
                  alone -> teacher writes a short hint (never sees the gold
                  answer) -> student answers WITH the hint -> reward = solved,
                  slammed to -1 if the teacher leaked the answer.

Colocated on one GPU: the engine sleeps while the trainer runs, then wakes.
SSH-friendly: no GUI, line-buffered logs, periodic sample dumps.

    python train_h100.py --backend stub --reward fake --steps 6   # no GPU smoke
    python train_h100.py --reward real --steps 200 --wandb        # the real run
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import signal
import time

import torch

import grpo
import paths
import seeding
import tasks
from config import Config
from engine import build_engine
from fake_reward import FakeReward
from interfaces import Trajectory, Turn
from monitor import Monitor
from rewards import build_real_rewarder
from zpd_filter import student_answer

def load_fake_topics(path=None):
    """Varied teaching prompts (math/science/english/social studies/logic) for the
    fake-reward pipeline test. Falls back if the file is absent."""
    path = path or str(paths.DATA / "topics.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            return [json.loads(l)["topic"] for l in f if l.strip()]
    return ["Explain to a 5th grader how to compare two fractions."]


def run_meta(save_dir):
    """Stable (wandb run id, local run dir), persisted so a requeued job continues
    the SAME wandb run and the same log dir - otherwise preemption splits the loss
    curve across two runs with a discontinuous step axis.

    Reuse is conditional on a resumable checkpoint existing. A requeue resumes from
    ckpt.pt and must keep the old identity; a fresh experiment starts from step 0 and
    must NOT, or it silently appends to the previous run's traces and metrics.
    """
    path = os.path.join(save_dir, "run_meta.json")
    resuming = os.path.exists(os.path.join(save_dir, "ckpt.pt"))
    if os.path.exists(path) and resuming:
        with open(path) as f:
            return json.load(f)
    import uuid

    meta = {"run_id": os.environ.get("SLURM_JOB_ID") or uuid.uuid4().hex[:8],
            "run_dir": str(paths.RUNS / time.strftime("%Y%m%d-%H%M%S"))}
    os.makedirs(save_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f)
    return meta


def save_ckpt(path, teacher, optimizer, step):
    """Atomic resumable checkpoint: LoRA weights + optimizer state + step.

    Written to a .tmp then os.replace'd, so a preemption mid-write can never
    leave a corrupt checkpoint behind.
    """
    from peft import get_peft_model_state_dict

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"step": step,
                "lora": get_peft_model_state_dict(teacher),
                "optim": optimizer.state_dict()}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_ckpt(path, teacher, optimizer):
    """Resume after a preemption/requeue. Returns the step to start from."""
    if not os.path.exists(path):
        return 0
    from peft import set_peft_model_state_dict

    ck = torch.load(path, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(teacher, ck["lora"])
    optimizer.load_state_dict(ck["optim"])
    start = int(ck["step"]) + 1
    print(f"[resume] loaded {path} - continuing from step {start}", flush=True)
    return start


class Timer:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.store[self.name] = self.store.get(self.name, 0.0) + (time.time() - self.t)


def load_teacher(cfg):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.teacher_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.teacher_model, dtype=cfg.resolve_dtype())
    base.config.use_cache = False
    model = get_peft_model(
        base,
        LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                   target_modules="all-linear", task_type="CAUSAL_LM"),
    ).to(cfg.resolve_device())
    model.print_trainable_parameters()
    return model, tok


def rollout_fake(cfg, engine, rewarder, rng):
    """Gameable keyword reward on generic teaching prompts."""
    topics = load_fake_topics()
    prompts = [rng.choice(topics) for _ in range(cfg.batch_prompts)]
    groups = engine.generate(prompts, n=cfg.group_size,
                             max_new_tokens=cfg.teacher_max_new_tokens,
                             temperature=cfg.temperature)
    samples, texts, traces, group_rewards = [], [], [], []
    for prompt, comps in zip(prompts, groups):
        grp = []
        for c in comps:
            traj = Trajectory(turns=[Turn(prompt, c)], transcript=c.text)
            r = rewarder.score(traj)["reward"]
            grp.append(r)
            texts.append(c.text)
            traces.append({"prompt": prompt, "completion": c.text, "reward": r})
        for c, a in zip(comps, grpo.group_normalized_advantages(grp, cfg.group_size)):
            samples.append({"prompt": prompt, "gen_ids": c.token_ids,
                            "old_logprobs": c.logprobs, "advantage": a})
        group_rewards.append(grp)
    return samples, texts, traces, group_rewards


class EpochSampler:
    """Draw prompts without replacement, reshuffling once the deck runs out.

    Independent draws with replacement leave coverage uneven: 300 steps x 4
    prompts is 1200 draws from 622 problems, so some are trained on repeatedly
    while others are never seen. A reshuffled deck visits each problem once per
    pass, which matters for a long run over a small curated set.
    """

    def __init__(self, items, rng):
        self._items = items
        self._rng = rng
        self._deck: list[int] = []

    def take(self, k):
        out = []
        while len(out) < k:
            if not self._deck:
                self._deck = list(range(len(self._items)))
                self._rng.shuffle(self._deck)
            out.append(self._items[self._deck.pop()])
        return out


def rollout_real(cfg, engine, rewarder, student, sampler, rng, tok=None):
    """Real task: hint -> student re-answers -> solved? (leak => -1)."""
    picks = sampler.take(cfg.batch_prompts)
    prompts = [tasks.teacher_prompt(p, tokenizer=tok) for p in picks]
    groups = engine.generate(prompts, n=cfg.group_size,
                             max_new_tokens=cfg.teacher_max_new_tokens,
                             temperature=cfg.temperature)

    use_spec = getattr(cfg, "specificity", "off") not in (None, "off")
    samples, texts, traces, group_rewards = [], [], [], []
    for i, (problem, prompt, comps) in enumerate(zip(picks, prompts, groups)):
        gold = tasks.gold_text(problem)
        # ONE foreign problem for the whole group, so the only thing that varies
        # across the K members is their own hint - see SpecificityGuard
        other = picks[(i + 1) % len(picks)] if len(picks) > 1 else None
        grp = []
        for c in comps:
            hint = c.text
            idx = student_answer(student, problem["question"], problem["choices"],
                                 hint=hint)
            pred = problem["choices"][idx]
            traj = Trajectory(turns=[Turn(prompt, c)], transcript=hint)
            traj.info = {"student_answer": pred, "gold": gold,
                         "question": problem["question"],
                         "distractors": [c for j, c in enumerate(problem["choices"])
                                         if j != problem["gold_idx"]]}
            if use_spec and other is not None:
                # does MY hint also work on a question it was not written for?
                off = student_answer(student, other["question"], other["choices"],
                                     hint=hint)
                traj.info["off_problem_solved"] = float(off == other["gold_idx"])
            scored = rewarder.score(traj)
            grp.append(scored["reward"])
            texts.append(hint)
            row = {"prompt": problem["question"], "completion": hint,
                   "choices": problem["choices"], "gold_idx": problem["gold_idx"],
                   "reward": scored["reward"], "solved": scored.get("solved", 0.0),
                   "leaked": scored.get("leaked", 0.0),
                   "specificity": scored.get("specificity"),
                   "off_problem_solved": scored.get("off_problem_solved"),
                   "student_answer": pred, "gold": gold}
            if cfg.hint_probe:
                from rewards import hint_only_leak
                row["hint_only_leak"] = hint_only_leak(
                    student, hint, problem["choices"], problem["gold_idx"])
            traces.append(row)
        for c, a in zip(comps, grpo.group_normalized_advantages(grp, cfg.group_size)):
            samples.append({"prompt": prompt, "gen_ids": c.token_ids,
                            "old_logprobs": c.logprobs, "advantage": a})
        group_rewards.append(grp)
    return samples, texts, traces, group_rewards


def run_dialogues(cfg, engine, student, problem_of, tok, turns, temperature,
                  keep_turns: bool = False, stop_when_solved: bool | None = None):
    """Run `turns` rounds of tutor<->student dialogue, one conversation per entry
    of problem_of. Shared by training rollouts and the held-out eval so that the
    eval measures the SAME behaviour that is being trained - evaluating a
    dialogue-trained teacher on one-shot hints would grade the wrong thing.

    `stop_when_solved` overrides cfg for one call. The eval passes False: the
    oracle stop consults gold, so leaving it on gives the own-problem condition a
    solve check at every turn while the swapped condition gets one shot at a
    fixed transcript, and specificity = teacher_acc - swapped_acc then measures
    the stop rather than the hint.

    Returns (transcripts, turns_of, tutor_texts, self_stopped). `tutor_texts`
    holds ONLY the teacher's turns: leak attribution must not charge the teacher
    for the answer when it was the student who said it. turns_of is only populated
    when keep_turns, since the eval has no use for per-turn token ids.
    """
    n = len(problem_of)
    transcripts = ["" for _ in range(n)]
    tutor_only = ["" for _ in range(n)]
    turns_of = [[] for _ in range(n)]
    done = [False] * n            # dialogues that ended early (solved, or teacher stopped)
    self_stopped = [0.0] * n      # ...of those, the ones the TEACHER ended
    self_stop = getattr(cfg, "self_stop", False)

    # --- the student speaks first ---
    openers = student.reply([tasks.student_opening_view(problem_of[d]) for d in range(n)])
    for d in range(n):
        transcripts[d] += f"Student: {openers[d].strip()}\n"

    for t in range(turns):
        live = [d for d in range(n) if not done[d]]
        if not live:
            break

        # --- teacher turn: one generation per ongoing dialogue ---
        t_prompts = [tasks.dialogue_prompt(problem_of[d], transcripts[d], tokenizer=tok,
                                           self_stop=self_stop)
                     for d in live]
        gens = engine.generate(t_prompts, n=1,
                               max_new_tokens=cfg.teacher_max_new_tokens,
                               temperature=temperature)
        for prompt, d, g in zip(t_prompts, live, gens):
            c = g[0]
            if keep_turns:
                # gen_ids stay RAW, marker included: the decision to stop is an
                # action the teacher took, and masking those tokens out of the loss
                # would leave the one behaviour we are trying to teach untrained
                turns_of[d].append({"prompt": prompt, "gen_ids": c.token_ids,
                                    "old_logprobs": c.logprobs})
            text, wants_stop = (tasks.strip_self_stop(c.text) if self_stop
                                else (c.text.strip(), False))
            if text:      # a turn that was ONLY the marker adds no tutoring to read
                transcripts[d] += f"Tutor: {text}\n"
                tutor_only[d] += text + "\n"
            if wants_stop:
                done[d] = True
                self_stopped[d] = 1.0

        if t == turns - 1:
            break

        live = [d for d in live if not done[d]]   # a teacher-ended dialogue gets no reply
        if not live:
            break

        # --- stop dialogues the student can now answer ---
        # NB: this consults gold, so it is an ORACLE stop - a deployed tutor would
        # have to judge readiness itself. It is here because every extra turn is
        # another chance to leak, and talking past understanding earns nothing.
        if (getattr(cfg, "stop_when_solved", True) if stop_when_solved is None
                else stop_when_solved):
            for d in live:
                p = problem_of[d]
                if student_answer(student, p["question"], p["choices"],
                                  hint=transcripts[d]) == p["gold_idx"]:
                    done[d] = True
            live = [d for d in live if not done[d]]
            if not live:
                break

        # --- student turn ---
        views = [tasks.student_dialogue_view(problem_of[d], transcripts[d]) for d in live]
        replies = student.reply(views)
        for d, rep in zip(live, replies):
            transcripts[d] += f"Student: {rep.strip()}\n"

    return transcripts, turns_of, tutor_only, self_stopped


def rollout_multiturn(cfg, engine, rewarder, student, sampler, rng, tok=None):
    """Teacher and student DISCUSS the problem, then the student answers.

    Each of the K group members runs its own conversation, so the K dialogues
    diverge. Every teacher turn becomes a training sample; the terminal reward
    (did the student finally solve it?) is shared by all turns in that trajectory.
    Student turns are environment text - they are never trained on, which is the
    teacher-token masking (we only ever store the teacher's gen_ids).
    """
    picks = sampler.take(cfg.batch_prompts)
    n_dialogues = cfg.batch_prompts * cfg.group_size
    # flat index d = prompt_i * K + k
    problem_of = [picks[d // cfg.group_size] for d in range(n_dialogues)]
    transcripts, turns_of, tutor_texts, self_stopped = run_dialogues(
        cfg, engine, student, problem_of, tok, cfg.turns, cfg.temperature,
        keep_turns=True)

    # --- terminal: student answers with the whole conversation as context ---
    use_spec = getattr(cfg, "specificity", "off") not in (None, "off")
    samples, texts, traces, group_rewards = [], [], [], []
    for i in range(cfg.batch_prompts):
        problem = picks[i]
        gold = tasks.gold_text(problem)
        distractors = [c for j, c in enumerate(problem["choices"]) if j != problem["gold_idx"]]
        # one foreign problem for the whole group; only the hint varies across k
        other = picks[(i + 1) % len(picks)] if len(picks) > 1 else None
        grp, grp_turns = [], []
        for k in range(cfg.group_size):
            d = i * cfg.group_size + k
            convo = transcripts[d]
            idx = student_answer(student, problem["question"], problem["choices"],
                                 hint=convo)
            pred = problem["choices"][idx]
            traj = Trajectory(turns=[], transcript=convo)
            # the student answers from the WHOLE conversation (it saw all of it),
            # but only the tutor's own words are used to judge leaking
            traj.info = {"student_answer": pred, "gold": gold, "distractors": distractors,
                         "question": problem["question"],
                         "teacher_text": tutor_texts[d]}
            if use_spec and other is not None:
                off = student_answer(student, other["question"], other["choices"],
                                     hint=tutor_texts[d])
                traj.info["off_problem_solved"] = float(off == other["gold_idx"])
            scored = rewarder.score(traj)
            grp.append(scored["reward"])
            grp_turns.append(turns_of[d])
            texts.append(convo)
            row = {"prompt": problem["question"], "completion": convo,
                   "choices": problem["choices"], "gold_idx": problem["gold_idx"],
                   "reward": scored["reward"], "solved": scored.get("solved", 0.0),
                   "leaked": scored.get("leaked", 0.0),
                   "specificity": scored.get("specificity"),
                   "off_problem_solved": scored.get("off_problem_solved"),
                   "student_answer": pred, "gold": gold, "turns": len(turns_of[d]),
                   "self_stopped": self_stopped[d]}
            if cfg.hint_probe:
                # tutor turns only, for the same attribution reason; leakage can
                # still be spread across several of them (elimination)
                from rewards import hint_only_leak
                row["hint_only_leak"] = hint_only_leak(
                    student, tutor_texts[d], problem["choices"], problem["gold_idx"])
            traces.append(row)
        advs = grpo.group_normalized_advantages(grp, cfg.group_size)
        for turns, a in zip(grp_turns, advs):
            for turn in turns:              # terminal reward shared by every teacher turn
                samples.append({**turn, "advantage": a})
        group_rewards.append(grp)
    return samples, texts, traces, group_rewards


def heldout_eval(cfg, engine, student, held_out, tok, n: int = 30):
    """Benchmark on problems the teacher NEVER trains on.

    This is the honest measure. Training reward can rise because the teacher got
    better OR because it found a hack - those look identical on training prompts.
    Held-out gold accuracy separates them, and divergence between the two is the
    hacking alarm. Greedy decoding so the number is comparable across steps.
    """
    from rewards import choices_only_baseline, hint_only_leak, leaked_answer

    items = held_out[:n]
    if not items:
        return {}
    if cfg.turns > 1:
        # match training: grade the dialogue, not a one-shot hint. The student is
        # shown the full transcript; leak checks see only the tutor's turns.
        # The one deliberate divergence is the oracle stop, disabled here so the
        # own-problem and swapped conditions are scored on equal terms.
        hints, _, tutor_texts, self_stopped = run_dialogues(cfg, engine, student, items,
                                                            tok, cfg.turns,
                                                            temperature=0.0,
                                                            stop_when_solved=False)
    else:
        prompts = [tasks.teacher_prompt(p, tokenizer=tok) for p in items]
        gens = engine.generate(prompts, n=1,
                               max_new_tokens=cfg.teacher_max_new_tokens,
                               temperature=0.0)
        hints = [g[0].text for g in gens]
        tutor_texts = hints
        self_stopped = [0.0] * len(items)   # nothing to stop: one turn, then answer
    # each item gets ANOTHER item's hint; rotating by one keeps the pairing
    # deterministic and guarantees no item ever receives its own hint
    swapped = hints[1:] + hints[:1]

    base_ok = help_ok = leak = probe = floor = swap_ok = clean_ok = 0.0
    hint_words = 0
    for p, hint, tutor_txt, other in zip(items, hints, tutor_texts, swapped):
        gold_idx = p["gold_idx"]
        gold = p["choices"][gold_idx]
        distractors = [c for j, c in enumerate(p["choices"]) if j != gold_idx]
        # the same channel the reward uses, so eval and training cannot drift apart
        base_ok += float(student_answer(student, p["question"], p["choices"]) == gold_idx)
        solved = float(student_answer(student, p["question"], p["choices"],
                                      hint=hint) == gold_idx)
        help_ok += solved
        # a hint from a DIFFERENT problem: whatever accuracy survives this is
        # generic encouragement, not teaching about this question
        swap_ok += float(student_answer(student, p["question"], p["choices"],
                                        hint=other) == gold_idx)
        # leak checks use the tutor's words only, never the student's
        leaked = leaked_answer(tutor_txt, gold, distractors, question=p["question"])
        leak += leaked
        # solved WITHOUT being told: the only solve that is evidence of teaching.
        # v0's headline solve rate fell purely because leaked solves went away, so
        # tracking this separately is what keeps a leak fix from reading as a
        # regression - and it is the honest number to compare across runs.
        clean_ok += solved * (1.0 - leaked)
        hint_words += len(tutor_txt.split())
        if cfg.hint_probe:
            probe += hint_only_leak(student, tutor_txt, p["choices"], gold_idx)
            floor += choices_only_baseline(student, p["choices"], gold_idx)

    m = len(items)
    out = {"eval/baseline_acc": base_ok / m, "eval/teacher_acc": help_ok / m,
           "eval/teaching_gain": (help_ok - base_ok) / m, "eval/leak_rate": leak / m,
           "eval/hint_words": hint_words / m, "eval/n": float(m),
           "eval/swapped_acc": swap_ok / m,
           "eval/clean_solved": clean_ok / m,
           # gain over the unhelped baseline, counting only un-leaked solves
           "eval/clean_gain": (clean_ok - base_ok) / m,
           # gain that does NOT survive swapping = the question-specific part
           "eval/specificity": (help_ok - swap_ok) / m}
    if getattr(cfg, "self_stop", False):
        # greedy decoding here, sampled in training: if this is far below the
        # training rate, stopping is a sampling accident rather than a policy
        out["eval/self_stop_rate"] = sum(self_stopped) / m
    if cfg.hint_probe:
        out["eval/hint_only_leak"] = probe / m
        out["eval/choices_only"] = floor / m
        # the only interpretable form of the probe
        out["eval/leak_above_floor"] = (probe - floor) / m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, choices=["stub", "hf", "vllm"])
    ap.add_argument("--reward", default="fake", choices=["fake", "real"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--reward-mode", default=None, choices=["keyword", "length", "random"])
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--student", default=None)
    ap.add_argument("--problems", default=tasks.DEFAULT_PATH)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None, help="checkpoint every N steps")
    ap.add_argument("--hint-probe", action="store_true", help="log the hint-only leak probe")
    ap.add_argument("--specificity", default=None, choices=["difference", "gated", "off"],
                    help="pay only for question-specific help: "
                         "solved(own problem) - solved(other problem | same hint)")
    ap.add_argument("--persona-adapter", default=None,
                    help="LoRA adapter making the student TALK like a kid; "
                         "applies to reply() only, choose() disables it")
    ap.add_argument("--kl-coef", type=float, default=None,
                    help="weight on KL-to-reference (default 0.05)")
    ap.add_argument("--eval-every", type=int, default=None, help="held-out benchmark every N steps (0=off)")
    ap.add_argument("--eval-n", type=int, default=None, help="held-out problems per benchmark")
    ap.add_argument("--eval-benchmark", default=None,
                    help="external eval set instead of the ZPD held-out split: a "
                         "registry name (python src/benchmarks.py --list) or a "
                         "path to a .jsonl")
    ap.add_argument("--no-sleep", action="store_true", help="keep the engine resident (no sleep/wake)")
    ap.add_argument("--turns", type=int, default=None, help="teacher turns per dialogue (1 = single-turn hint)")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="keep tutoring for the full turn budget even once the "
                         "student answers correctly")
    ap.add_argument("--student-answer-mode", default=None, choices=["logprob", "free"],
                    help="how the student commits to an answer, i.e. the reward "
                         "channel; 'free' invalidates the ZPD curation and the "
                         "published baselines (python src/check_answer_modes.py)")
    ap.add_argument("--self-stop", action="store_true",
                    help="let the teacher end the dialogue with [DONE] (multi-turn "
                         "only; disables the oracle --stop-when-solved)")
    ap.add_argument("--sync-every", type=int, default=None, help="push LoRA into the engine every N steps")
    ap.add_argument("--gpu-mem-util", type=float, default=None, help="vLLM share of GPU memory (rest for trainer)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seeds the data split, the sampler, torch/numpy and the "
                         "engine's per-call sampling seed (see src/seeding.py for "
                         "what is still nondeterministic)")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    for src, dst in [(args.backend, "backend"), (args.steps, "total_steps"),
                     (args.group_size, "group_size"), (args.reward_mode, "reward_mode"),
                     (args.teacher, "teacher_model"), (args.student, "student_model"),
                     (args.micro_batch, "micro_batch_size"), (args.save_every, "save_every"),
                     (args.sync_every, "sync_every"), (args.gpu_mem_util, "gpu_mem_util"),
                     (args.turns, "turns"),
                     (args.student_answer_mode, "student_answer_mode")]:
        if src:
            setattr(cfg, dst, src)
    cfg.use_wandb = args.wandb
    cfg.no_sleep = args.no_sleep
    cfg.hint_probe = args.hint_probe
    if args.specificity is not None:
        cfg.specificity = args.specificity
    if args.kl_coef is not None:
        cfg.kl_coef = args.kl_coef
    if args.persona_adapter:
        cfg.persona_adapter = args.persona_adapter
    cfg.eval_benchmark = args.eval_benchmark
    cfg.stop_when_solved = not args.no_early_stop
    cfg.self_stop = args.self_stop
    if cfg.self_stop:
        # see Config.self_stop: with the oracle stop cutting the dialogue off first,
        # over-talking costs the teacher nothing and [DONE] cannot be learned
        cfg.stop_when_solved = False
        if cfg.turns <= 1:
            print("[cfg] --self-stop has no effect at --turns 1: the single-turn "
                  "path answers straight after one hint, so there is no dialogue "
                  "to end", flush=True)
        else:
            print("[cfg] self-stop on - oracle stop_when_solved disabled", flush=True)
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every
    if args.eval_n:
        cfg.eval_n = args.eval_n
    if args.seed is not None:
        cfg.seed = args.seed
    # before load_teacher: the LoRA A matrices are randomly initialised, so a run
    # seeded after the model is built starts from different weights every time
    seeding.seed_everything(cfg.seed)
    rng = random.Random(cfg.seed)
    stub = cfg.resolve_backend() == "stub"
    device = cfg.resolve_device()
    meta = run_meta(cfg.save_dir)
    monitor = Monitor(use_wandb=cfg.use_wandb, wandb_project=cfg.wandb_project,
                      wandb_entity=cfg.wandb_entity, config=vars(cfg),
                      run_id=meta["run_id"], dir_override=meta["run_dir"])

    # ---- reward + (for real mode) the student and the ZPD problem set ----
    problems = student = sampler = None
    if args.reward == "real":
        rewarder = build_real_rewarder(specificity=cfg.specificity, leak_penalty=-1.0)
        print(f"[reward] LeakGuard o SpecificityGuard({cfg.specificity!r}) o SolveReward",
              flush=True)
        all_problems = tasks.load_zpd(args.problems)
        # An external eval set replaces the held-out split a few lines below, so
        # carving one out here would strand those problems: trained on by nobody,
        # evaluated by nobody.
        test_frac = 0.0 if cfg.eval_benchmark else 0.15
        problems, held_out = tasks.split_problems(all_problems, test_frac=test_frac,
                                                  seed=cfg.seed)
        from zpd_filter import HFStudent, StubStudent

        student = (StubStudent() if stub else
                   HFStudent(cfg.student_model, device=device,
                             persona_adapter=cfg.persona_adapter))
        if cfg.student_answer_mode != "logprob":
            student.set_answer_mode(cfg.student_answer_mode)
            print(f"[student] reward channel = {cfg.student_answer_mode}: the ZPD "
                  "curation and the reported baselines were measured on 'logprob' "
                  "and do not describe this run", flush=True)
        sampler = EpochSampler(problems, rng)
        print(f"[data] {len(problems)} train / {len(held_out)} held-out ZPD problems", flush=True)
        if cfg.eval_benchmark:
            # The ZPD held-out split has baseline_acc == 0 by construction, which
            # makes teaching_gain identical to teacher_acc. An unfiltered external
            # set gives the baseline real headroom AND tests transfer off the
            # training corpus.
            import benchmarks

            held_out = benchmarks.load_benchmark(cfg.eval_benchmark, limit=cfg.eval_n,
                                                 seed=cfg.seed)
            print(f"[data] held-out eval switched to '{cfg.eval_benchmark}' "
                  f"({len(held_out)} items, unfiltered)", flush=True)
    else:
        rewarder = FakeReward(mode=cfg.reward_mode)

    print(f"[cfg] backend={cfg.resolve_backend()} device={device} reward={args.reward} "
          f"K={cfg.group_size} micro_batch={cfg.micro_batch_size}", flush=True)

    if stub:
        teacher = tok = optimizer = scheduler = None
        engine = build_engine(cfg)
        print("[stub] no teacher weights - orchestration/monitoring only", flush=True)
    else:
        teacher, tok = load_teacher(cfg)
        engine = build_engine(cfg, model=teacher, tokenizer=tok)
        optimizer = torch.optim.AdamW(
            [p for p in teacher.parameters() if p.requires_grad], lr=cfg.lr)
        # linear warmup then CONSTANT (no cosine): RL gradients are noisy and an
        # early spike can derail the run; after warmup the lr stays flat.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps)))

    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, "ckpt.pt")

    # resume across Slurm preemption/requeue
    start_step = 0 if stub else load_ckpt(ckpt_path, teacher, optimizer)

    # Slurm sends SIGTERM before killing a preempted job - checkpoint and exit
    # cleanly so --requeue picks up exactly where we left off.
    preempted = {"flag": False}

    def _on_term(signum, frame):
        preempted["flag"] = True
        print(f"[signal {signum}] preemption - will checkpoint and exit", flush=True)

    signal.signal(signal.SIGTERM, _on_term)

    @contextlib.contextmanager
    def awake():
        """Every engine.generate() must happen inside this.

        vLLM sleep(level=1) offloads weights and drops the KV cache. Generating
        while asleep does not raise a clean error - it fails deep inside LoRA
        activation with `CUDA error: invalid argument`, which reads like a shape
        bug rather than a lifecycle bug.
        """
        if not cfg.no_sleep:
            engine.wake()
        try:
            yield
        finally:
            if not cfg.no_sleep:
                engine.sleep()

    def do_rollout():
        with awake():
            if args.reward == "real":
                roll = rollout_multiturn if cfg.turns > 1 else rollout_real
                return roll(cfg, engine, rewarder, student, sampler, rng, tok=tok)
            return rollout_fake(cfg, engine, rewarder, rng)

    for step in range(start_step, cfg.total_steps):
        t: dict = {}
        # engines that seed per request key off the step, so a requeued run picks
        # the stream back up instead of replaying the beginning (see set_seed_epoch)
        if hasattr(engine, "set_seed_epoch"):
            engine.set_seed_epoch(step)
        with Timer(t, "generate_and_reward"):
            samples, texts, traces, group_rewards = do_rollout()

        # ---- update ----
        metrics = {}
        with Timer(t, "update"):
            if not stub:
                batch = grpo.prepare_batch(tok, samples, device, max_len=cfg.max_seq_len)
                ref = grpo.reference_logprobs(teacher, batch, micro_batch=cfg.micro_batch_size)
                epoch_metrics = []
                for _ in range(cfg.update_epochs):
                    optimizer.zero_grad(set_to_none=True)
                    _, m = grpo.grpo_loss(teacher, batch, ref, cfg,
                                          micro_batch=cfg.micro_batch_size)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in teacher.parameters() if p.requires_grad], 1.0)
                    optimizer.step()
                    epoch_metrics.append(m)
                # once per TRAINING step, not per epoch - otherwise warmup finishes
                # update_epochs times too early
                scheduler.step()
                # average over epochs so the logged loss represents the STEP, not
                # just whichever epoch happened to run last
                metrics = {k: sum(d[k] for d in epoch_metrics) / len(epoch_metrics)
                           for k in epoch_metrics[0]}

        with Timer(t, "sync"):
            if (step + 1) % cfg.sync_every == 0:
                engine.sync_weights(teacher)

        # ---- metrics ----
        rewards = [r for grp in group_rewards for r in grp]
        if not stub:
            metrics["lr"] = optimizer.param_groups[0]["lr"]
        mean_r = sum(rewards) / max(1, len(rewards))
        # groups with no reward variance contribute EXACTLY ZERO gradient in GRPO
        # (advantage = (r - mean)/std = 0). This is the real "is my data working"
        # signal - if it approaches 1.0, most compute is wasted.
        zero_adv = sum(1 for g in group_rewards if max(g) == min(g)) / max(1, len(group_rewards))
        extra = {}
        if args.reward == "real":
            if cfg.hint_probe and traces and "hint_only_leak" in traces[0]:
                extra["hint_only_leak"] = sum(tr["hint_only_leak"] for tr in traces) / len(traces)
            extra = {**extra, "solved_rate": sum(tr.get("solved", 0.0) for tr in traces) / max(1, len(traces)),
                     "leak_rate": sum(tr.get("leaked", 0.0) for tr in traces) / max(1, len(traces))}
            used = [tr["turns"] for tr in traces if tr.get("turns")]
            if used:
                # falls below cfg.turns when the student gets there early
                extra["mean_turns"] = sum(used) / len(used)
            if cfg.self_stop:
                # mean_turns alone cannot say WHY a dialogue was short; this is the
                # share the teacher chose to end
                ended = [tr["self_stopped"] for tr in traces if "self_stopped" in tr]
                if ended:
                    extra["self_stop_rate"] = sum(ended) / len(ended)

        monitor.log_metrics(step, {"reward": mean_r, "zero_adv_frac": zero_adv, **extra,
                                   **metrics, **{f"time/{k}": v for k, v in t.items()}})
        monitor.log_traces(step, traces)
        monitor.check_hacking(step, texts, rewards)
        msg = (f"step {step}/{cfg.total_steps}  reward={mean_r:.3f}  zero_adv={zero_adv:.2f}  "
               f"loss={metrics.get('loss', float('nan')):.4f}  kl={metrics.get('kl', 0.0):.4f}  "
               f"clip={metrics.get('clip_frac', 0.0):.2f}")
        if extra:
            msg += f"  solved={extra['solved_rate']:.2f}  leak={extra['leak_rate']:.2f}"
        print(msg + "  " + "  ".join(f"{k}={v:.1f}s" for k, v in t.items()), flush=True)

        if (not stub and cfg.eval_every and step > 0 and step % cfg.eval_every == 0):
            with Timer(t, "heldout_eval"), awake():
                ev = heldout_eval(cfg, engine, student, held_out, tok, n=cfg.eval_n)
            if ev:
                monitor.log_metrics(step, ev)
                print(f"  [held-out] teacher_acc={ev['eval/teacher_acc']:.3f} "
                      f"clean_solved={ev['eval/clean_solved']:.3f} "
                      f"gain={ev['eval/teaching_gain']:+.3f} leak={ev['eval/leak_rate']:.3f}"
                      + (f" hint_only_leak={ev['eval/hint_only_leak']:.3f}"
                         if 'eval/hint_only_leak' in ev else ""), flush=True)

        if cfg.print_samples_every and step % cfg.print_samples_every == 0:
            monitor.print_samples(k=1)          # stdout (tail -f the job log)
            monitor.append_samples_md(step, traces)  # samples.md (tail -f over SSH)
            monitor.log_sample_table(step, traces)   # wandb Table (live web UI)

        if not stub and ((step + 1) % cfg.save_every == 0 or preempted["flag"]):
            save_ckpt(ckpt_path, teacher, optimizer, step)
            teacher.save_pretrained(os.path.join(cfg.save_dir, "adapter-latest"))
            monitor.plot()
            monitor.write_html()

        if preempted["flag"]:
            print("[exit] checkpointed at step "
                  f"{step}; Slurm --requeue will resume from here", flush=True)
            monitor.close()
            return

    if not stub:
        save_ckpt(ckpt_path, teacher, optimizer, cfg.total_steps - 1)
        teacher.save_pretrained(os.path.join(cfg.save_dir, "teacher-final"))
    monitor.close()


if __name__ == "__main__":
    main()
