"""Collect the user's own (human-typed) inputs from Cursor agent transcripts.

Extracts only the text inside <user_query>...</user_query> - i.e. what YOU
actually typed - dropping AI replies, tool results, and the auto-injected
context (open files, timestamps, attached files). Writes a jsonl dataset plus a
plain-text view, and prints basic stats.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

TRANSCRIPTS = Path.home() / ".cursor/projects/Users-sophiaz-RL/agent-transcripts"
OUT_JSONL = Path(__file__).parent / "my_inputs.jsonl"
OUT_TXT = Path(__file__).parent / "my_inputs.txt"

QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL)


def main_transcripts():
    # main chat files are <uuid>/<uuid>.jsonl; skip anything under subagents/
    for f in TRANSCRIPTS.rglob("*.jsonl"):
        if "subagents" in f.parts:
            continue
        if f.stem == f.parent.name:
            yield f


def extract_from(path: Path):
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("role") != "user":
            continue
        content = rec.get("message", {}).get("content", [])
        parts = content if isinstance(content, list) else [{"text": content}]
        for part in parts:
            text = part.get("text", "") if isinstance(part, dict) else str(part)
            for m in QUERY_RE.findall(text):
                q = m.strip()
                if q:
                    out.append(q)
    return out


def main():
    inputs = []
    for f in main_transcripts():
        got = extract_from(f)
        inputs.extend(got)
        print(f"{f.parent.name[:8]}: {len(got)} inputs")

    with OUT_JSONL.open("w") as fh:
        for q in inputs:
            fh.write(json.dumps({"text": q}) + "\n")
    OUT_TXT.write_text("\n---\n".join(inputs))

    lengths = [len(q) for q in inputs]
    print("\n=== stats ===")
    print(f"total inputs: {len(inputs)}")
    if lengths:
        print(f"avg chars: {sum(lengths) / len(lengths):.1f}  min: {min(lengths)}  max: {max(lengths)}")
        words = [len(q.split()) for q in inputs]
        print(f"avg words: {sum(words) / len(words):.1f}")
    print(f"wrote {OUT_JSONL} and {OUT_TXT}")


if __name__ == "__main__":
    main()
