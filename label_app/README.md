# label_app

Collects human labels on tutor turns from training run v2, to train a reward
model. Live at <https://grpo-tutor-label.web.app>.

Separate from `review_app/`, which reviews run v0 and is **OpenBookQA-only** by
policy — its build script and deploy script both refuse state-assessment
content. This app deliberately carries state-assessment content, so it must
never be merged with that one.

## What labellers do

Two item types, interleaved one pair after every four turns:

- **turn** — one tutor message, in the context of everything said before it.
  Rated on leak (`no` / `hints_at_it` / `names_it`) and help (`helps` /
  `too_vague` / `just_tells`).
- **pair** — two dialogues for the same question, "which taught better"
  (`a` / `b` / `tie`).

Turns are what the leak head trains on and what calibrates the rule-based
detector. Pairs are what the usefulness head trains on, and they are drawn from
within a single GRPO group because a within-group comparison is exactly what the
advantage uses — cross-problem pairs teach an easier question the algorithm never
has to answer.

## Rebuilding the bundle

```bash
python src/build_label_set.py --run runs/20260731-212530
```

Writes `data/label_items.json` (published) and `data/label_key.json` (**not**
published — it holds the rule's verdict, and showing it would anchor the
labeller on the label we are trying to validate). The 1,200 turns are stratified
on the rule's own decision, over-weighting the boundary so both precision and
recall are estimable.

## Deploying

```bash
cd label_app
npx -y firebase-tools@latest deploy --only firestore,hosting
```

## Getting the labels back

Labels land in the Firestore `labels` collection. Clients cannot read it, so:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/keys/grpo-tutor-label.json
python src/fetch_labels.py --out data/labels.jsonl
```

The service account key comes from the Firebase console under Project settings →
Service accounts. Keep it out of the repo.

## Security posture

Rules are in `firestore.rules`. The `labels` collection is **write-once**: create
only, with a strict schema, enum checks, length caps on every string, and a
freshness check on the timestamp. No client can read, update or delete.

Anonymous sign-in is **off**, so writes are currently unauthenticated — a
deliberate choice, since volunteers cannot be asked to register. The client
already calls `signInAnonymously()`, so enabling the provider in the console
immediately starts attaching a `uid`, after which `allow create` can be tightened
to require `request.auth != null` with no other change.

These rules are a prototype that has been attack-tested against schema
pollution, enum tampering, oversized payloads, stale timestamps, uid spoofing and
collection reads — all rejected. Review them before opening this up widely.

## Licensing

Questions and answer options are released items from public state assessments
(California, Texas, Massachusetts, New Jersey). Rights remain with the issuing
state agencies; tutor and student turns are model output. This app is served with
`X-Robots-Tag: noindex, nofollow` and carries attribution in the footer.
Publishing it was an explicit decision — see the repo's `.gitignore`, which keeps
the same content out of git.
