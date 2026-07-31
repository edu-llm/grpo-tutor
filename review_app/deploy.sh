#!/usr/bin/env bash
# Publish review_app/ to GitHub Pages at https://edu-llm.github.io/tutor-review/
#
# The site is deployed from a SEPARATE repo (edu-llm/tutor-review) so that
# publishing does not touch the research repo's git history. Run from anywhere.
#
#   bash review_app/deploy.sh
#
# Requires: gh, authenticated with write access to the edu-llm org.
set -euo pipefail

ORG=edu-llm
REPO=tutor-review
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [[ ! -f "$SRC/data/items.json" ]]; then
  echo "data/items.json missing - run: python src/build_review_set.py" >&2
  exit 1
fi

echo "==> staging site"
mkdir -p "$WORK/site/data"
cp "$SRC"/index.html "$SRC"/styles.css "$SRC"/app.js "$WORK/site/"
cp "$SRC"/data/items.json "$SRC"/data/analysis_key.json "$WORK/site/data/"
touch "$WORK/site/.nojekyll"
cat > "$WORK/site/README.md" <<'MD'
# tutor-review

Static review app for AI tutor dialogues from `grpo_tutor` training run v0.
Live at <https://edu-llm.github.io/tutor-review/>.

Reviewers reply as the student with the correct answer hidden, then judge whether
the tutor gave the answer away and whether the hint would help anyone learn.
Answers stay in the browser and are exported as JSON; there is no backend.

Questions and options come from OpenBookQA (openly licensed); the tutor and student
turns are our own models' output. Deployed from `review_app/` in the research repo —
edit there, not here.
MD

# Belt and braces: the build script already refuses to emit state-assessment
# content, but never publish without checking the actual bytes being pushed.
# (Checks the staged bundle, not the source tree, whose docs name the terms.)
if grep -rilE 'staar|pssa|mcas|caaspp|njsla' "$WORK/site" >/dev/null 2>&1; then
  echo "REFUSING TO DEPLOY: state-assessment content found in the bundle" >&2
  grep -rilE 'staar|pssa|mcas|caaspp|njsla' "$WORK/site" >&2
  exit 1
fi

if ! gh repo view "$ORG/$REPO" >/dev/null 2>&1; then
  echo "==> creating $ORG/$REPO"
  gh repo create "$ORG/$REPO" --public \
    --description "Human review of AI tutor dialogues (grpo_tutor run v0)"
fi

echo "==> pushing"
cd "$WORK/site"
git init -q -b main
git add -A
git -c user.name="grpo_tutor deploy" -c user.email="deploy@edu-llm.invalid" \
    commit -qm "Deploy tutor dialogue review app"
git remote add origin "https://github.com/$ORG/$REPO.git"
git push -q --force origin main

echo "==> enabling Pages"
gh api -X POST "repos/$ORG/$REPO/pages" \
  -f 'source[branch]=main' -f 'source[path]=/' >/dev/null 2>&1 || \
gh api -X PUT "repos/$ORG/$REPO/pages" \
  -f 'source[branch]=main' -f 'source[path]=/' >/dev/null 2>&1 || true

URL="https://$ORG.github.io/$REPO/"
echo "==> waiting for $URL"
for _ in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$URL" || true)
  [[ "$code" == "200" ]] && { echo "live: $URL"; exit 0; }
  sleep 10
done
echo "not live yet - check: gh api repos/$ORG/$REPO/pages" >&2
exit 1
