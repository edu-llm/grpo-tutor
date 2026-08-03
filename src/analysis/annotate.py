"""Local UI for hand-writing student turns.

    python src/annotate.py        then open http://localhost:8765

Generated persona data failed in ways metrics missed: 30% of replies opened with
"Like,", 20% ended "Confused", and only half referenced the question at all. A
model asked to imitate a 7th grader writes its stereotype of one. So the anchors
have to be written by a person.

Contexts are REAL - question plus the actual tutor turn from a training run - so
what you write is directly usable as a training row, not just a style sample.
Each answer appends to data/student_seeds_manual.jsonl immediately; closing the
tab loses nothing.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import random
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

import paths

OUT = paths.DATA / "student_seeds_manual.jsonl"

KINDS = [
    ("opening", "First thing you say - you have just read the question"),
    ("reply", "Replying to what the tutor just said"),
]

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>student seeds</title><style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0b1020;color:#f0f4fa;
  font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:19px;margin:0 0 4px;font-weight:700}
.sub{color:#9fb0cc;font-size:14px;margin-bottom:22px}
.card{background:#141b30;border:1px solid #263354;border-radius:14px;padding:20px;margin-bottom:18px}
.kind{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.06em;
  text-transform:uppercase;color:#0b1020;background:#7cc4ff;border-radius:999px;padding:3px 10px;margin-bottom:12px}
.q{font-size:18px;font-weight:600;margin-bottom:14px}
.turn{background:#0e1526;border-left:3px solid #7cc4ff;border-radius:0 8px 8px 0;
  padding:12px 14px;margin:12px 0;color:#dce6f7;font-size:15px}
.turn b{color:#7cc4ff;font-weight:700}
.choices{background:#0e1526;border:1px dashed #4a5f8f;border-radius:10px;
  padding:11px 14px;margin-bottom:6px;font-size:14px;color:#dce6f7}
.choices .warn{display:block;color:#ffd479;font-weight:700;font-size:12px;
  letter-spacing:.04em;text-transform:uppercase;margin-bottom:7px}
.choices span.opt{display:inline-block;margin-right:16px;white-space:nowrap}
label{display:block;font-size:14px;font-weight:600;color:#c7d5ee;margin:16px 0 6px}
textarea{width:100%;min-height:88px;background:#0e1526;color:#f0f4fa;font-size:17px;
  border:2px solid #33436e;border-radius:10px;padding:12px 14px;resize:vertical;
  font-family:inherit;line-height:1.5}
textarea:focus{outline:none;border-color:#7cc4ff;box-shadow:0 0 0 3px rgba(124,196,255,.25)}
.row{display:flex;gap:10px;align-items:center;margin-top:14px;flex-wrap:wrap}
button{font:inherit;font-weight:700;font-size:15px;border-radius:10px;padding:11px 20px;
  border:2px solid transparent;cursor:pointer}
.primary{background:#7cc4ff;color:#08111f}
.primary:hover{background:#a3d5ff}
.ghost{background:transparent;color:#c7d5ee;border-color:#33436e}
.ghost:hover{border-color:#7cc4ff;color:#fff}
button:focus-visible{outline:3px solid #ffd479;outline-offset:2px}
.hint{color:#9fb0cc;font-size:13px}
.count{font-weight:700;color:#8ef0b8}
.tips{background:#141b30;border:1px solid #263354;border-radius:14px;padding:16px 20px}
.tips ul{margin:8px 0 0;padding-left:20px;color:#c7d5ee;font-size:14px}
.tips li{margin:5px 0}
.done{margin-top:10px;font-size:14px;color:#8ef0b8;min-height:20px;font-weight:600}
kbd{background:#0e1526;border:1px solid #33436e;border-radius:5px;padding:1px 6px;font-size:12px}
</style></head><body><div class="wrap">
<h1>Student seed collector</h1>
<div class="sub">Written so far: <span class="count" id="n">0</span> &nbsp;·&nbsp; saves to
  <code>data/student_seeds_manual.jsonl</code> as you go</div>

<div class="card">
  <div class="kind" id="kind">...</div>
  <div class="q" id="q">loading…</div>
  <div class="choices" id="choices"></div>
  <div id="convo"></div>
  <label for="a" id="lab">What does the student say?</label>
  <textarea id="a" autofocus placeholder="type it the way a real kid would…"></textarea>
  <div class="row">
    <button class="primary" id="save">Save &amp; next</button>
    <button class="ghost" id="skip">Skip</button>
    <span class="hint"><kbd>⌘</kbd>+<kbd>Enter</kbd> to save</span>
  </div>
  <div class="done" id="msg"></div>
</div>

<div class="tips"><b>Keep them varied</b>
<ul>
  <li>Not all questions — include wrong guesses, half-understanding, frustration</li>
  <li>Write how a kid types: lowercase, typos, no full stops are all fine</li>
  <li>Never say the answer, even if you know it</li>
  <li>Don't name options by letter — the student can't see the list, only the question</li>
  <li>Short. Under ~15 words</li>
</ul></div>
</div><script>
let cur=null;
async function load(){
  const r=await fetch('/api/next'); cur=await r.json();
  document.getElementById('n').textContent=cur.saved;
  document.getElementById('kind').textContent=cur.kind_label;
  document.getElementById('q').textContent=cur.question;
  document.getElementById('choices').innerHTML='<span class="warn">'
    +'context only \u2014 the student cannot see these when it speaks</span>'
    +cur.choices.map(c=>`<span class="opt">\u2022 ${c}</span>`).join('');
  document.getElementById('convo').innerHTML=cur.turns.map(
    t=>`<div class="turn"><b>Tutor</b><br>${t}</div>`).join('');
  document.getElementById('lab').textContent=cur.kind==='opening'
    ? "First thing you say about this question:" : "What do you say back?";
  const a=document.getElementById('a'); a.value=''; a.focus();
}
async function send(skip){
  const a=document.getElementById('a'); const text=a.value.trim();
  if(!skip && !text){a.focus();return;}
  await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({...cur, assistant:text, skipped:!!skip})});
  document.getElementById('msg').textContent = skip? 'skipped' : '✓ saved';
  setTimeout(()=>document.getElementById('msg').textContent='',1200);
  load();
}
document.getElementById('save').onclick=()=>send(false);
document.getElementById('skip').onclick=()=>send(true);
document.addEventListener('keydown',e=>{
  if((e.metaKey||e.ctrlKey)&&e.key==='Enter')send(false);});
load();
</script></body></html>"""


def load_contexts(limit=400):
    """Real (question, tutor turns) pairs from training traces."""
    out = []
    for f in sorted(glob.glob(str(paths.RUNS / "*" / "traces.jsonl")), reverse=True):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("turns") or not r.get("choices"):
                continue
            tutor = [ln[len("Tutor:"):].strip()
                     for ln in r.get("completion", "").split("\n") if ln.startswith("Tutor:")]
            if not tutor:
                continue
            q = r.get("prompt", "")
            ch = list(r["choices"])
            out.append({"question": q, "choices": ch, "turns": [], "kind": "opening"})
            for i in range(1, len(tutor) + 1):
                out.append({"question": q, "choices": ch, "turns": tutor[:i],
                            "kind": "reply"})
            if len(out) >= limit:
                return out
    return out


class Handler(BaseHTTPRequestHandler):
    contexts: list = []
    rng = random.Random(0)

    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/next"):
            c = dict(self.rng.choice(self.contexts))
            c["turns"] = [html.escape(t) for t in c["turns"]]
            c["kind_label"] = dict(KINDS)[c["kind"]]
            c["saved"] = sum(1 for _ in open(OUT)) if OUT.exists() else 0
            return self._json(c)
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        row = json.loads(self.rfile.read(n) or b"{}")
        if not row.get("skipped") and row.get("assistant", "").strip():
            with open(OUT, "a") as f:
                f.write(json.dumps({"question": row["question"],
                                    "tutor_turns": [re.sub(r"&#x27;|&quot;|&amp;", "'", t)
                                                    for t in row.get("turns", [])],
                                    "kind": row.get("kind"),
                                    "text": row["assistant"].strip()}) + "\n")
        self._json({"ok": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    Handler.contexts = load_contexts()
    if not Handler.contexts:
        raise SystemExit("no usable traces found - run training first, or copy a "
                         "runs/*/traces.jsonl over from the cluster")
    have = sum(1 for _ in open(OUT)) if OUT.exists() else 0
    print(f"[annotate] {len(Handler.contexts)} contexts loaded, {have} already written")
    print(f"[annotate] open http://localhost:{args.port}   (ctrl-c to stop)")
    print(f"[annotate] appending to {OUT}")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
