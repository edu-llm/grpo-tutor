"""Monitoring: traces you can actually read + scalars + reward-hacking detectors.

Design choice: wandb is fine for scalar curves (loss/reward/KL) but poor for
reading generations. So:
  - scalars  -> metrics.jsonl (+ wandb if enabled) + a PNG curve
  - traces   -> traces.jsonl AND a self-contained traces.html you open in a
                browser: every sample, its reward, and any hack flags, sorted so
                the best/worst are easy to eyeball.

Everything is local-first and dependency-light; wandb is optional.
"""

from __future__ import annotations

import html
import json
import os
import statistics as st
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------- detectors

def repetition_ratio(text: str, n: int = 3) -> float:
    """1 - (distinct n-grams / total n-grams). High = degenerate repetition."""
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def answer_leakage(text: str, answers) -> float:
    """Fraction of held-out answers that appear verbatim in the teacher's text.

    THE key hack for a tutoring reward: leaking answers instead of teaching.
    """
    if not answers:
        return 0.0
    low = text.lower()
    hits = sum(1 for a in answers if str(a).strip().lower() in low)
    return hits / len(answers)


def group_collapse(texts) -> float:
    """Fraction of duplicate completions in a group. High = mode collapse."""
    if not texts:
        return 0.0
    return 1.0 - len(set(texts)) / len(texts)


def corr(xs, ys) -> float:
    """Pearson correlation; used for reward-vs-length (length hacking)."""
    if len(xs) < 3:
        return 0.0
    try:
        return st.correlation(xs, ys)
    except Exception:
        return 0.0


@dataclass
class Alert:
    step: int
    kind: str
    value: float
    message: str


DEFAULT_THRESHOLDS = {
    "leakage": 0.15,        # >15% of answers appearing verbatim
    "repetition": 0.5,      # >50% repeated 3-grams
    "collapse": 0.5,        # >50% duplicate completions in a group
    "reward_len_corr": 0.6,  # reward strongly tracking length
    "length_growth": 2.0,   # mean length 2x the first-step baseline
}


# ---------------------------------------------------------------- monitor

@dataclass
class Monitor:
    run_dir: str = "runs"
    use_wandb: bool = False
    wandb_project: str = "grpo_tutor"
    config: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    run_id: str | None = None      # stable id so a requeued job RESUMES the same wandb run
    dir_override: str | None = None  # stable local dir so traces/metrics stay in one place

    def __post_init__(self):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = self.dir_override or os.path.join(self.run_dir, stamp)
        os.makedirs(self.dir, exist_ok=True)
        self.traces_path = os.path.join(self.dir, "traces.jsonl")
        self.metrics_path = os.path.join(self.dir, "metrics.jsonl")
        self.history: list[dict] = []
        self.alerts: list[Alert] = []
        self._recent_traces: list[dict] = []
        self._baseline_len = None
        self._wandb = None
        if self.use_wandb:
            try:
                import wandb

                self._wandb = wandb
                # id + resume="allow" so a Slurm requeue continues the SAME run and
                # the step axis stays continuous instead of restarting in a new run
                # env var wins if set (conventional wandb behaviour); passing
                # project= explicitly would otherwise silently ignore WANDB_PROJECT
                project = os.environ.get("WANDB_PROJECT") or self.wandb_project
                wandb.init(project=project, config=self.config, dir=self.dir,
                           id=self.run_id, resume="allow" if self.run_id else None)
                print(f"[monitor] wandb project={project}")
                if self.run_id:
                    print(f"[monitor] wandb run id={self.run_id} (resumable)")
                mode = os.environ.get("WANDB_MODE", "online")
                print(f"[monitor] wandb mode={mode}")
                if mode == "offline":
                    # compute nodes have no internet; upload later from a login node
                    print(f"[monitor] after the job:  wandb sync {self.dir}/wandb/latest-run")
            except Exception as e:  # never let logging kill a run
                print(f"[monitor] wandb disabled ({e})")
        print(f"[monitor] run dir: {self.dir}")

    # ---- scalars ----
    def log_metrics(self, step: int, metrics: dict):
        row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        self.history.append(row)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if self._wandb:
            self._wandb.log(row, step=step)

    # ---- traces ----
    def log_traces(self, step: int, records: list[dict]):
        """records: [{prompt, completion, reward, ...}] - written verbatim to jsonl."""
        with open(self.traces_path, "a") as f:
            for r in records:
                f.write(json.dumps({"step": step, **r}) + "\n")
        self._recent_traces = [{"step": step, **r} for r in records]

    # ---- hacking detection ----
    def check_hacking(self, step: int, completions: list[str], rewards: list[float],
                      answers=None) -> list[Alert]:
        found: list[Alert] = []
        if not completions:
            return found

        leak = st.fmean(answer_leakage(c, answers or []) for c in completions)
        rep = st.fmean(repetition_ratio(c) for c in completions)
        coll = group_collapse(completions)
        lengths = [len(c.split()) for c in completions]
        mean_len = st.fmean(lengths)
        rl = corr(lengths, rewards) if len(set(rewards)) > 1 else 0.0

        if self._baseline_len is None:
            self._baseline_len = max(1.0, mean_len)
        growth = mean_len / self._baseline_len

        checks = [
            ("leakage", leak, "teacher may be LEAKING answers instead of teaching"),
            ("repetition", rep, "degenerate/repetitive generations"),
            ("collapse", coll, "group mode-collapse (duplicate completions)"),
            ("reward_len_corr", rl, "reward tracking LENGTH (length hacking)"),
            ("length_growth", growth, "completion length ballooning vs baseline"),
        ]
        for kind, value, msg in checks:
            if value > self.thresholds[kind]:
                a = Alert(step, kind, float(value), msg)
                found.append(a)
                print(f"[HACK? step {step}] {kind}={value:.2f} - {msg}")

        self.alerts.extend(found)
        self.log_metrics(step, {
            "hack/leakage": leak, "hack/repetition": rep, "hack/collapse": coll,
            "hack/reward_len_corr": rl, "hack/length_growth": growth,
            "gen/mean_len": mean_len,
        })
        return found

    # ---- outputs ----
    def plot(self, keys=("reward", "loss"), path: str | None = None):
        if not self.history:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = [k for k in keys if any(k in h for h in self.history)]
        if not keys:
            return
        fig, axes = plt.subplots(len(keys), 1, figsize=(7, 3 * len(keys)), squeeze=False)
        for ax, k in zip(axes[:, 0], keys):
            xs = [h["step"] for h in self.history if k in h]
            ys = [h[k] for h in self.history if k in h]
            ax.plot(xs, ys)
            ax.set_ylabel(k)
            ax.set_xlabel("step")
        fig.tight_layout()
        fig.savefig(path or os.path.join(self.dir, "curves.png"), dpi=120)
        plt.close(fig)

    def write_html(self, max_samples: int = 60):
        """Self-contained browsable trace viewer (this is the wandb-is-bad-at-this part)."""
        rows = []
        try:
            with open(self.traces_path) as f:
                all_rows = [json.loads(l) for l in f]
            rows = all_rows[-max_samples:]
        except FileNotFoundError:
            pass
        rows.sort(key=lambda r: r.get("reward", 0.0), reverse=True)

        alert_html = "".join(
            f"<li><b>step {a.step}</b> {html.escape(a.kind)}={a.value:.2f} - {html.escape(a.message)}</li>"
            for a in self.alerts[-40:]
        ) or "<li>none</li>"

        blocks = []
        for r in rows:
            blocks.append(
                "<div class='c'>"
                f"<div class='h'>step {r.get('step')} &middot; reward "
                f"<b>{r.get('reward', 0.0):.3f}</b></div>"
                f"<pre class='p'>{html.escape(str(r.get('prompt', ''))[-800:])}</pre>"
                f"<pre class='g'>{html.escape(str(r.get('completion', '')))}</pre>"
                "</div>"
            )

        doc = f"""<!doctype html><meta charset="utf-8"><title>GRPO traces</title>
<style>
body{{font:14px/1.45 ui-monospace,Menlo,monospace;margin:24px;background:#111;color:#ddd}}
h1{{font-size:18px}} .c{{border:1px solid #333;border-radius:8px;margin:12px 0;overflow:hidden}}
.h{{background:#1b1b1b;padding:8px 12px;border-bottom:1px solid #333}}
pre{{margin:0;padding:10px 12px;white-space:pre-wrap;word-break:break-word}}
.p{{color:#8aa;background:#151515;max-height:160px;overflow:auto}}
.g{{color:#cfe}} ul{{color:#f9a}}
</style>
<h1>GRPO traces - {html.escape(self.dir)}</h1>
<p>{len(rows)} samples (sorted by reward). Alerts:</p><ul>{alert_html}</ul>
{''.join(blocks)}"""
        path = os.path.join(self.dir, "traces.html")
        with open(path, "w") as f:
            f.write(doc)
        return path

    def log_sample_table(self, step: int, rows: list[dict], k: int = 8):
        """Log responses to wandb as a Table so they're readable live in the web UI.

        This is the "responses somewhere accessible" path: same dashboard as the
        loss curves, no port-forwarding, viewable from anywhere.
        """
        if not self._wandb or not rows:
            return
        try:
            ranked = sorted(rows, key=lambda r: r.get("reward", 0.0))
            picks = ranked[: max(1, k // 2)] + ranked[-max(1, k // 2):]
            cols = ["step", "reward", "solved", "leaked", "prompt", "completion"]
            table = self._wandb.Table(columns=cols)
            for r in picks:
                table.add_data(
                    step,
                    float(r.get("reward", 0.0)),
                    float(r.get("solved", 0.0) or 0.0),
                    float(r.get("leaked", 0.0) or 0.0),
                    str(r.get("prompt", ""))[:600],
                    str(r.get("completion", ""))[:2000],
                )
            self._wandb.log({"samples": table}, step=step)
        except Exception as e:
            print(f"[monitor] sample table skipped ({e})", flush=True)

    def append_samples_md(self, step: int, rows: list[dict], k: int = 2):
        """Append readable best/worst samples to samples.md - trivial to `tail -f` over SSH."""
        if not rows:
            return
        ranked = sorted(rows, key=lambda r: r.get("reward", 0.0))
        picks = [("worst", r) for r in ranked[:k]] + [("best", r) for r in ranked[-k:]]
        with open(os.path.join(self.dir, "samples.md"), "a") as f:
            f.write(f"\n## step {step}\n")
            for label, r in picks:
                f.write(f"\n**[{label}] reward={r.get('reward', 0.0):.3f}"
                        f" solved={r.get('solved', '-')} leaked={r.get('leaked', '-')}**\n\n")
                f.write(f"- prompt: {str(r.get('prompt', ''))[:300]}\n")
                f.write(f"- hint: {str(r.get('completion', '')).strip()[:800]}\n")
                if "student_answer" in r:
                    f.write(f"- student answered: {r['student_answer']}  (gold: {r.get('gold')})\n")

    def print_samples(self, k: int = 2, width: int = 400):
        """Dump best/worst completions to stdout - the SSH-friendly view of traces."""
        rows = sorted(self._recent_traces, key=lambda r: r.get("reward", 0.0))
        if not rows:
            return
        picks = [("worst", r) for r in rows[:k]] + [("best", r) for r in rows[-k:]]
        for label, r in picks:
            txt = str(r.get("completion", "")).replace("\n", " ")[:width]
            print(f"  [{label} r={r.get('reward', 0.0):.3f}] {txt}", flush=True)

    def close(self):
        self.plot()
        path = self.write_html()
        print(f"[monitor] traces: {path}")
        print(f"[monitor] metrics: {self.metrics_path}")
        if self._wandb:
            self._wandb.finish()
