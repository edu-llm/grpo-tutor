#!/bin/bash
# One-shot status for a run. Usage, from anywhere:
#   ssh orcd-login 'cd ~/orcd/scratch/grpo_tutor && bash scripts/status.sh'
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1

echo "=== jobs ==="
squeue -u "$USER" -h -o "  %.10i %.14j %.9T %.8M %.22E" 2>/dev/null | grep . \
  || echo "  (nothing queued or running)"

echo
echo "=== training set ==="
if [ -f data/zpd_problems.jsonl ]; then
  echo "  zpd_problems.jsonl: $(wc -l < data/zpd_problems.jsonl) problems"
fi
if [ -f data/zpd_problems_oldcriterion.jsonl ]; then
  echo "  (old criterion had: $(wc -l < data/zpd_problems_oldcriterion.jsonl))"
fi

LATEST_ZPD=$(ls -t logs/zpdrecur_*.out 2>/dev/null | head -1)
if [ -n "$LATEST_ZPD" ]; then
  echo
  echo "=== last curation ($LATEST_ZPD) ==="
  grep -E "accuracy|gain|free-text|kept|WARNING" "$LATEST_ZPD" 2>/dev/null | sed 's/^/  /'
fi

LATEST=$(ls -t logs/mt_*.out logs/full_*.out 2>/dev/null | head -1)
echo
echo "=== training ($([ -n "$LATEST" ] && echo "$LATEST" || echo none)) ==="
if [ -n "$LATEST" ]; then
  echo "  last steps:"
  grep -E "^step" "$LATEST" 2>/dev/null | tail -3 | sed 's/^/    /'
  echo "  held-out evals:"
  grep -E "held-out" "$LATEST" 2>/dev/null | tail -4 | sed 's/^/    /' || echo "    (none yet)"
  grep -qE "Traceback|CUDA error" "$LATEST" 2>/dev/null && echo "  !! errors present - tail the log"
fi
