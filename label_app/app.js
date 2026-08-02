/* Reward-model labelling — static, works with or without a backend.
 *
 * Two item types, interleaved: TURN (rate one tutor message on leak + help) and
 * PAIR (two dialogues for the same question, which taught better). Turns train
 * the leak head and calibrate the rule-based detector; pairs train the usefulness
 * head, and they are drawn from within a single GRPO group because that is the
 * comparison the advantage actually uses.
 *
 * Labels live in localStorage from the moment they are made. Submission is
 * best-effort on top of that, never instead of it: a failed POST loses nothing,
 * and the download button is always the fallback.
 */
'use strict';

/* Submission goes through firebase-submit.js, which sets window.submitLabel.
 * If that module fails to load the app degrades to download-only rather than
 * blocking anyone from labelling. */
const canSubmit = () => typeof window.submitLabel === 'function';

const SCHEMA = 'grpo-tutor-label/v1';
const STORE = 'grpo-tutor-label:v1:';
const LAST_KEY = STORE + '_last';
const PAIR_EVERY = 4;          // one A/B after every four turn ratings

/* Ratings, not comparisons. Every point is anchored with a description: an
 * unanchored 1-5 drifts across a session and between people, and two labellers
 * can agree completely about a hint and still differ by two points. */
/* Ratings, not comparisons. Kept to a word each: the scale is answered a
 * thousand times, so anything longer gets skimmed after the first dozen and the
 * reading cost is paid on every single item. */
const LEAK_OPTS = [
  { v: 1, k: '1', t: 'none' },
  { v: 2, k: '2', t: 'hints' },
  { v: 3, k: '3', t: 'gives it' },
];
const HELP_OPTS = [
  { v: 1, k: '4', t: 'useless' },
  { v: 2, k: '5', t: 'filler' },
  { v: 3, k: '6', t: 'some' },
  { v: 4, k: '7', t: 'good' },
  { v: 5, k: '8', t: 'excellent' },
];
const PAIR_OPTS = [
  { v: 'a',    k: '1', t: 'A taught better',  s: '' },
  { v: 'b',    k: '2', t: 'B taught better',  s: '' },
  { v: 'tie',  k: '3', t: 'About the same',   s: 'including "both are bad"' },
];

const $ = (id) => document.getElementById(id);
const state = {
  who: '', bundle: null, queue: [], at: 0,
  labels: {}, skipped: [], pending: { leak: null, help: null, pair: null },
};

/* ── storage ─────────────────────────────────────────────────────────────── */
const key = () => STORE + state.who.toLowerCase().replace(/\s+/g, '_');

function save() {
  try {
    localStorage.setItem(key(), JSON.stringify({
      who: state.who, labels: state.labels, skipped: state.skipped, at: state.at,
    }));
    localStorage.setItem(LAST_KEY, state.who);
  } catch (e) { banner('Could not save locally — download before you close the tab.'); }
}

function load(who) {
  try {
    const raw = localStorage.getItem(STORE + who.toLowerCase().replace(/\s+/g, '_'));
    return raw ? JSON.parse(raw) : null;
  } catch (e) { return null; }
}

/* ── deterministic per-person shuffle, so two people see different orders ─── */
function hash(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function shuffled(arr, seedStr) {
  const a = arr.slice();
  let s = hash(seedStr) || 1;
  const rnd = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/* ── rendering helpers ───────────────────────────────────────────────────── */
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const gold = (item) => { try { return atob(item.gold_b64); } catch (e) { return ''; } };

function renderChoices(el, item) {
  const g = gold(item);
  el.innerHTML = item.choices.map((c, i) => {
    const isGold = String(c) === g;
    return `<li class="${isGold ? 'is-gold' : ''}">
      <span class="ch-letter">${String.fromCharCode(65 + i)}</span>
      <span>${esc(c)}</span>
      ${isGold ? '<span class="ch-tag">correct answer</span>' : ''}</li>`;
  }).join('');
}

function renderDialogue(el, lines) {
  el.innerHTML = lines.map((l) =>
    `<p class="turn turn-${l.who}"><span class="turn-who">${l.who === 'tutor' ? 'Tutor' : 'Student'}:</span> ${esc(l.text)}</p>`
  ).join('');
}

function parseTranscript(text) {
  const out = [];
  String(text).split('\n').forEach((line) => {
    const m = /^(Tutor|Student):\s?(.*)$/.exec(line);
    if (m) out.push({ who: m[1].toLowerCase(), text: m[2] });
    else if (out.length) out[out.length - 1].text += '\n' + line;
  });
  return out.filter((t) => t.text.trim());
}

function renderOpts(el, opts, onPick, selected) {
  el.innerHTML = opts.map((o) => `
    <button type="button" class="opt${selected === o.v ? ' is-on' : ''}" role="radio"
            aria-checked="${selected === o.v}" data-v="${o.v}"
            title="${esc(o.t)} (key ${o.k})">
      <span class="opt-n">${o.v}</span>
      <span class="opt-t">${esc(o.t)}</span>
    </button>`).join('');
  el.querySelectorAll('.opt').forEach((b) => {
    // data-* is always a string; ratings are numbers, so map back through the
    // option list rather than writing '3' into Firestore where 3 is expected
    b.addEventListener('click', () => {
      const hit = opts.find((o) => String(o.v) === b.dataset.v);
      onPick(hit ? hit.v : b.dataset.v);
    });
  });
}

/* ── screens ─────────────────────────────────────────────────────────────── */
function show(id) {
  ['screen-landing', 'screen-turn', 'screen-pair', 'screen-done']
    .forEach((s) => { $(s).hidden = (s !== id); });
  $('topbar').hidden = (id === 'screen-landing');
  window.scrollTo({ top: 0, behavior: 'instant' });
}

function banner(msg) {
  $('banner-text').textContent = msg;
  $('banner').hidden = false;
}

function tally() {
  const n = Object.keys(state.labels).length;
  $('tally').textContent = n === 0 ? 'Ready when you are'
    : `${n} labelled${state.skipped.length ? ` · ${state.skipped.length} skipped` : ''}`;
}

function current() { return state.queue[state.at]; }

function render() {
  tally();
  const item = current();
  if (!item) return done();
  state.pending = { leak: null, help: null, pair: null };

  if (item._kind === 'turn') {
    show('screen-turn');
    $('t-question').textContent = item.question;
    renderChoices($('t-choices'), item);
    renderDialogue($('t-context'), item.context);
    // the conversation before this turn is context, not the thing being judged,
    // so it starts folded away - two short turns are cheap enough to show
    const ctx = $('t-ctx-wrap');
    ctx.hidden = item.context.length === 0;
    ctx.open = item.context.length <= 2;
    $('t-ctx-summary').textContent =
      `Show the ${item.context.length} message${item.context.length === 1 ? '' : 's'} before this`;
    $('t-turn').textContent = item.tutor_turn;
    $('t-note').value = '';
    renderOpts($('opts-leak'), LEAK_OPTS, (v) => { state.pending.leak = v; render_marks(); }, null);
    renderOpts($('opts-help'), HELP_OPTS, (v) => { state.pending.help = v; render_marks(); }, null);
    $('t-next').disabled = true;
  } else {
    show('screen-pair');
    $('p-question').textContent = item.question;
    renderChoices($('p-choices'), item);
    renderDialogue($('p-a'), parseTranscript(item.a));
    renderDialogue($('p-b'), parseTranscript(item.b));
    $('p-note').value = '';
    renderOpts($('opts-pair'), PAIR_OPTS, (v) => { state.pending.pair = v; render_marks(); }, null);
    $('p-next').disabled = true;
  }
}

function render_marks() {
  const item = current();
  if (item._kind === 'turn') {
    renderOpts($('opts-leak'), LEAK_OPTS, (v) => { state.pending.leak = v; render_marks(); }, state.pending.leak);
    renderOpts($('opts-help'), HELP_OPTS, (v) => { state.pending.help = v; render_marks(); }, state.pending.help);
    $('t-next').disabled = !(state.pending.leak != null && state.pending.help != null);
  } else {
    renderOpts($('opts-pair'), PAIR_OPTS, (v) => { state.pending.pair = v; render_marks(); }, state.pending.pair);
    $('p-next').disabled = !state.pending.pair;
  }
}

function commit() {
  const item = current();
  const base = { id: item.id, kind: item._kind, who: state.who, at: new Date().toISOString() };
  if (item._kind === 'turn') {
    if (state.pending.leak == null || state.pending.help == null) return;
    state.labels[item.id] = { ...base, leak: state.pending.leak, goodness: state.pending.help,
                              note: $('t-note').value.trim() || undefined };
  } else {
    if (!state.pending.pair) return;
    state.labels[item.id] = { ...base, winner: state.pending.pair,
                              note: $('p-note').value.trim() || undefined };
  }
  state.at += 1;
  save();
  submitOne(state.labels[item.id]);
  render();
}

function skip() {
  const item = current();
  if (item && !state.skipped.includes(item.id)) state.skipped.push(item.id);
  state.at += 1;
  save();
  render();
}

function done() {
  show('screen-done');
  const n = Object.keys(state.labels).length;
  const turns = Object.values(state.labels).filter((l) => l.kind === 'turn').length;
  $('done-summary').textContent =
    `${n} labels — ${turns} tutor messages rated and ${n - turns} head-to-heads.`;
  $('submit-status').textContent = canSubmit()
    ? 'Your labels were sent as you went. The download is a backup.'
    : 'Sending is unavailable — please download the file and send it on.';
}

/* ── submission (best effort, never blocking) ────────────────────────────── */
const unsent = [];

function submitOne(label) {
  if (!canSubmit()) { unsent.push(label); return; }
  window.submitLabel(label, state.bundle.source).catch(() => {
    // already in localStorage; retried on the next label and downloadable anyway
    unsent.push(label);
  });
  while (unsent.length) {
    const queued = unsent.shift();
    window.submitLabel(queued, state.bundle.source).catch(() => {});
  }
}

function download() {
  const payload = {
    schema: SCHEMA, source: state.bundle.source, who: state.who,
    exported: new Date().toISOString(),
    labels: Object.values(state.labels), skipped: state.skipped,
  };
  const blob = new Blob([JSON.stringify(payload, null, 1)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `tutor-labels-${state.who.toLowerCase().replace(/\s+/g, '-')}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ── queue: interleave so nobody does 1,200 of the same thing ────────────── */
function buildQueue() {
  // ONE canonical order, the same for everybody, entered at a different point
  // per person. Two labellers only ever meet the same item once between them
  // have covered the whole set, so effort is spent on breadth instead of on
  // duplicating each other - and no coordination or server round-trip is needed
  // to arrange it, which a claim-an-item scheme would require.
  //
  // The trade is that agreement between labellers becomes unmeasurable: nobody
  // shares items by construction. Set --n-overlap-turns in build_label_set.py to
  // put a shared prefix back if that matters later.
  const turns = state.bundle.turns.map((t) => ({ ...t, _kind: 'turn' }));
  const pairs = state.bundle.pairs.map((p) => ({ ...p, _kind: 'pair' }));
  const canonical = [];
  let pi = 0;
  turns.forEach((t, i) => {
    canonical.push(t);
    if ((i + 1) % PAIR_EVERY === 0 && pi < pairs.length) canonical.push(pairs[pi++]);
  });
  while (pi < pairs.length) canonical.push(pairs[pi++]);

  const start = canonical.length ? hash(state.who) % canonical.length : 0;
  return canonical.slice(start).concat(canonical.slice(0, start));
}

function start(who, resume) {
  state.who = who;
  const prev = resume ? load(who) : null;
  if (prev) { state.labels = prev.labels || {}; state.skipped = prev.skipped || []; }
  state.queue = buildQueue();
  // resume past anything already answered or skipped
  state.at = 0;
  while (state.at < state.queue.length) {
    const id = state.queue[state.at].id;
    if (state.labels[id] || state.skipped.includes(id)) state.at += 1; else break;
  }
  render();
}

/* ── boot ────────────────────────────────────────────────────────────────── */
fetch('data/label_items.json')
  .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
  .then((b) => {
    state.bundle = b;
    $('loading').hidden = true;
    $('attribution').textContent = b.attribution || '';
    $('landing-count').textContent =
      `${b.turns.length} tutor messages and ${b.pairs.length} head-to-heads available.`;
    const last = localStorage.getItem(LAST_KEY);
    if (last && load(last)) {
      const n = Object.keys(load(last).labels || {}).length;
      $('name').value = last;
      $('resume-line').hidden = false;
      $('resume-line').textContent = `Welcome back, ${last} — ${n} labels saved. Enter the same name to carry on.`;
    }
  })
  .catch((e) => {
    $('loading').hidden = true;
    $('fatal').hidden = false;
    $('fatal').textContent = `Could not load the items (${e.message}). Reload, or tell Sophia.`;
  });

$('name-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const who = $('name').value.trim();
  if (!who) {
    $('name-error').hidden = false;
    $('name-error').textContent = 'Please enter a name so we can label your answers.';
    return;
  }
  $('name-error').hidden = true;
  start(who, true);
});

$('t-next').addEventListener('click', commit);
$('p-next').addEventListener('click', commit);
$('t-skip').addEventListener('click', skip);
$('p-skip').addEventListener('click', skip);
$('btn-save-top').addEventListener('click', download);
$('btn-download').addEventListener('click', download);
$('banner-close').addEventListener('click', () => { $('banner').hidden = true; });
$('btn-again').addEventListener('click', () => {
  if (!state.skipped.length) return;
  state.queue = state.queue.filter((i) => state.skipped.includes(i.id));
  state.skipped = [];
  state.at = 0;
  render();
});
$('btn-reset').addEventListener('click', () => {
  if (!confirm('Erase your labels on this device? This cannot be undone.')) return;
  localStorage.removeItem(key());
  state.labels = {}; state.skipped = []; state.at = 0;
  show('screen-landing');
});

document.addEventListener('keydown', (e) => {
  if (e.target.matches('textarea, input')) return;
  const item = current();
  if (!item || $('screen-landing').hidden === false) return;
  if (e.key === 'Enter') {
    const btn = item._kind === 'turn' ? $('t-next') : $('p-next');
    if (!btn.disabled) { e.preventDefault(); commit(); }
    return;
  }
  const opts = item._kind === 'turn' ? LEAK_OPTS.concat(HELP_OPTS) : PAIR_OPTS;
  const hit = opts.find((o) => o.k === e.key);
  if (!hit) return;
  e.preventDefault();
  if (item._kind === 'pair') state.pending.pair = hit.v;
  else if (LEAK_OPTS.includes(hit)) state.pending.leak = hit.v;
  else state.pending.help = hit.v;
  render_marks();
});
