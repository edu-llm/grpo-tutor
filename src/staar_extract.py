"""Turn released Texas STAAR test PDFs into self-contained multiple-choice items.

    !!! THE EXTRACTED CONTENT IS NOT REDISTRIBUTABLE. DO NOT COMMIT IT. !!!

Every STAAR PDF carries:

    Copyright (c) 2019, Texas Education Agency. All rights reserved.
    Reproduction of all or portions of this work is prohibited without
    express written permission from the Texas Education Agency.

This repository is PUBLIC. The test content is TEA's; only the method below is
ours. So the downloaded PDFs and everything derived from them are written under
`data/staar/`, which is in `.gitignore`, and this file plus the docs are the
only things that get committed. If you ever find `data/staar/` staged, the
answer is `git restore --staged`, not `git add -f`. Rebuild it instead:

    python src/staar_extract.py --download

WHY THIS EXISTS
---------------
`docs/dataset_choice.md` surveyed the HF corpora and found the binding
constraint is not volume but the oracle *hint*: the ZPD screen keeps items the
student "solves with help", which has a degenerate solution (a hint containing
the answer), so most corpora's headroom is the student copying. STAAR sidesteps
the survey entirely by being a different kind of object - real grade 3-8 exam
items, human-written distractors, four subjects. It has NO hint field at all,
which is a real limitation and is spelled out in the docs; see `staar` in
`benchmarks.py`.

WHAT THE SOURCE ACTUALLY LOOKS LIKE
-----------------------------------
Probed 2013-2021 x grades 3-8 x {math, reading, science, social-studies} x
{test, key, answer-key} = 972 URLs. Only 2018 and 2019 exist as a test/key pair
under the canonical name; 2013-2017 are gone from the site, 2020 was cancelled,
2021 sits on the tea2 mirror under a different naming scheme with NO answer key
(rationale PDFs only), and 2022+ never existed on paper because STAAR moved
online. So the usable universe is two years, and that is not a bug in the
prober.

    {test}  https://tea.texas.gov/.../{year}-staar-{grade}-{subject}-test.pdf
    {key}   ...-key.pdf, except 2019 social studies which is ...-answer-key.pdf

`pypdf` is the wrong tool here: on the grade 8 science form it drops every
intra-word space ("Atomsofwhichtwoelements..."). `pdfplumber` gets it right, so
that is the dependency, and it is deliberately imported lazily - it is not in
`requirements.txt` and nothing else in the repo needs it.

    python -m venv /tmp/pdfenv && /tmp/pdfenv/bin/python -m pip install pdfplumber

Test-booklet structure, which the parser leans on:

  * Everything before the line "DIRECTIONS" is front matter (periodic table,
    formula sheet) and extracts as noise. Reading booklets have no DIRECTIONS
    line; there the first passage header serves.
  * Items are numbered at line start. Reading passages ALSO number their
    paragraphs from 1, so a bare number is not enough to locate an item.
  * Option letters usually alternate with item parity: odd items use A B C D,
    even items use F G H J. NOT ALWAYS - the 2018 forms are the "Online"
    release and use A B C D throughout, and their keys say so. Assuming parity
    would have mislabeled every even-numbered item in half the corpus without
    raising anything, so the scheme is read off each key rather than assumed.
  * "Griddable" items are numeric entry and have no options at all.
  * Option text is not reliably one-option-per-line. Table-valued options lay
    out in two columns ("A Dog Beds C Dog Beds"), and some forms run all four
    onto a single line. Those are located so the item count stays aligned, then
    dropped, because their text cannot be recovered.

HOW THE LABELS ARE KEPT HONEST
------------------------------
A misaligned key silently mislabels every item in a form, which is worse than
extracting nothing. The parser therefore never assumes its own output is right:
it reads the key first (which also tells it which items are griddable, since
those have a numeric answer), then requires that the option sets it found match
the key item-for-item AND that each set's letters match the parity the item
number predicts. Any disagreement drops the whole form rather than the item.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import paths

STAAR_DIR = paths.DATA / "staar"
PDF_DIR = STAAR_DIR / "pdf"

BASES = [
    "https://tea.texas.gov/student-assessment/staar/released-test-questions",
    "https://tea2.tea.texas.gov/data-reports/staar/released-test-questions",
]
# Only 2018/2019 survive as test+key pairs; see the module docstring for the
# probe that established that. `--years 2013 ... 2021` re-runs it if TEA ever
# reposts the archive, at about four minutes for the full grid.
YEARS = (2018, 2019)
GRADES = (3, 4, 5, 6, 7, 8)
SUBJECTS = ("math", "reading", "science", "social-studies")
KEY_SUFFIXES = ("key", "answer-key")

ODD_LETTERS = "ABCD"
EVEN_LETTERS = "FGHJ"
ALL_LETTERS = ODD_LETTERS + EVEN_LETTERS


def letters_for(item_no: int, alternating: bool) -> str:
    return EVEN_LETTERS if alternating and item_no % 2 == 0 else ODD_LETTERS


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------

def fetch(url: str, timeout: int = 120) -> bytes | None:
    """Return the body only if it is really a PDF.

    TEA's 404 page is served as a 140KB HTML document, and on some paths with a
    200, so neither the status code nor the length can be trusted. The magic
    bytes can.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return None
    return body if body[:5] == b"%PDF-" else None


def download_all(years, grades, subjects, workers: int = 12) -> dict:
    """Fetch every {test, key} pair that exists. Returns a probe report."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    wanted = []
    for year in years:
        for grade in grades:
            for subject in subjects:
                wanted.append((year, grade, subject))

    def one(form):
        year, grade, subject = form
        stem = f"{year}-staar-{grade}-{subject}"
        got = {}
        for kind, suffixes in (("test", ("test",)), ("key", KEY_SUFFIXES)):
            for suffix in suffixes:
                name = f"{stem}-{suffix}.pdf"
                dest = PDF_DIR / name
                if dest.exists() and dest.stat().st_size > 1024:
                    got[kind] = dest
                    break
                for base in BASES:
                    body = fetch(f"{base}/{name}")
                    if body:
                        dest.write_bytes(body)
                        got[kind] = dest
                        break
                if kind in got:
                    break
        return form, got

    report = {"found": [], "test_only": [], "key_only": [], "missing": []}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for form, got in ex.map(one, wanted):
            if "test" in got and "key" in got:
                report["found"].append(form)
            elif "test" in got:
                report["test_only"].append(form)
            elif "key" in got:
                report["key_only"].append(form)
            else:
                report["missing"].append(form)
    return report


# --------------------------------------------------------------------------
# text extraction
# --------------------------------------------------------------------------

BOILERPLATE = re.compile(
    r"^\s*(?:"
    r"Page\s*\d+"
    r"|STAAR\b.*"
    r"|Science|Mathematics|Math|Reading|Social\s*Studies|READING"
    r"|Copyright\s*©.*"
    r"|written permission.*"
    r"|State of Texas|Assessments of|Academic Readiness"
    r"|GRADE\s*\d+|Administered\b.*|RELEASED"
    r"|Record your answer.*|correct place value\.?"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}"
    r"|\d{5,}"
    r"|-{20,}"
    r")\s*$",
    re.IGNORECASE,
)

# The back cover, which otherwise lands inside the last option of the form.
# Case-SENSITIVE on purpose: the printed marker is "STOP", and matching it
# loosely deletes any question whose stem happens to say "comes to a stop".
BACK_COVER = re.compile(
    r"^\s*(?:BE SURE YOU HAVE RECORDED.*|ON THE ANSWER DOCUMENT.*|[A-Za-z ]{0,20}STOP)\s*$")

# pdfplumber renders a few glyphs in ways that are wrong rather than merely
# ugly: STAAR's comma comes out as a low-9 quote, and its minus doubles.
TIDY = (("\u201a", ","), ("\u2212\u2212", "\u2212"), ("\u2013\u2013", "\u2013"))


def tidy(text: str) -> str:
    for bad, good in TIDY:
        text = text.replace(bad, good)
    # a lone "?" opening a stem is the placeholder box of a fill-in-the-title
    # figure; the list it belongs to is in the stem, the box itself is not
    text = re.sub(r"^\?\s+", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()


# The DIRECTIONS paragraph is prose, sits in front of item 1, and would
# otherwise be measured as a shared stimulus and condemn the whole booklet.
INSTRUCTIONS = re.compile(
    r"Read (?:each question|the selection|the next|these selections)"
    r"|determine the best answer|four answer choices|griddable question"
    r"|answer document|Then fill in|choose the best answer",
    re.IGNORECASE,
)


def page_lines(pdf_path) -> list[str]:
    import pdfplumber

    lines: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw in text.split("\n"):
                line = raw.strip()
                if (line and not BOILERPLATE.match(line) and not BACK_COVER.match(line)
                        and not INSTRUCTIONS.search(line)):
                    lines.append(line)
    return lines


# --------------------------------------------------------------------------
# answer key
# --------------------------------------------------------------------------

KEY_ROW = re.compile(r"^(\d{1,2})\s+(\S.*\S|\S)\s*$")


def parse_key(pdf_path) -> dict[int, str]:
    """item number -> correct answer, as printed.

    Key rows look like `3 1 Readiness 8.5(B) A`: item number first, correct
    answer last, reporting-category noise in between. Take the last column
    rather than pattern-matching it - griddable answers are numeric and come in
    shapes a character class keeps missing ("1200", "535.25", "\u20131"), and a row
    that fails to match truncates the key, which silently shortens the form.
    """
    answers: dict[int, str] = {}
    for line in page_lines(pdf_path):
        m = KEY_ROW.match(line)
        if not m:
            continue
        item_no, answer = int(m.group(1)), m.group(2).split()[-1]
        # rows arrive in order, so the next row is the next item; anything else
        # is table noise (repeated headers, footnotes)
        if item_no == len(answers) + 1:
            answers[item_no] = answer
    return answers


# --------------------------------------------------------------------------
# test booklet
# --------------------------------------------------------------------------

DIRECTIONS = re.compile(r"^DIRECTIONS\s*$", re.IGNORECASE)


def body_lines(lines: list[str]) -> list[str]:
    """Drop the front matter: reference tables extract as pure garbage.

    Reading booklets have no DIRECTIONS line and no front matter to speak of,
    so there is nothing to trim and the whole thing is the body - which is what
    we want, since their first passage precedes their first item and has to be
    visible for the shared-stimulus check to see it.
    """
    for i, line in enumerate(lines):
        if DIRECTIONS.match(line):
            return lines[i + 1:]
    return lines


def key_scheme(key: dict[int, str]) -> tuple[bool | None, str]:
    """Is this an alternating (paper) or a uniform-ABCD (online) form?

    Read off the key, never assumed. If any even item answers F/G/H/J the form
    alternates; if every letter answer is in A-D it is uniform. A form that is
    neither - an odd item answering H, say - is not a form we understand, and
    guessing would mislabel it silently, so it gets refused.
    """
    letters = {n: a for n, a in key.items() if a in ALL_LETTERS}
    if not letters:
        return None, "key has no letter answers at all"
    even_high = any(a in EVEN_LETTERS for n, a in letters.items() if n % 2 == 0)
    odd_high = any(a in EVEN_LETTERS for n, a in letters.items() if n % 2 == 1)
    if odd_high:
        return None, "an odd-numbered item answers F/G/H/J - unknown lettering"
    return even_high, ""


def _starts(line: str, letter: str) -> str | None:
    m = re.match(rf"^{letter}\s+(\S.*)$", line)
    return m.group(1) if m else None


def _option_end(lines: list[str], start: int, hi: int) -> int:
    """Where the last option's text stops: at the next item number or option."""
    j = start
    while j < hi:
        if re.match(r"^\d{1,2}(\s|$)", lines[j]):
            break
        if any(_starts(lines[j], c) for c in ALL_LETTERS):
            break
        j += 1
    return j


def find_block(lines: list[str], lo: int, hi: int,
               letters: str) -> tuple[int, int, list[str] | None] | None:
    """Locate one item's four options inside the line range [lo, hi).

    Anchored on the SECOND letter, not the first. "A" opens plenty of ordinary
    sentences ("A student uses this diagram..."), so anchoring on it and
    reading forward happily swallows a stem line as option A; anchoring on B
    and then taking the LAST "A ..." line before it cannot.

    Returns (start, end, texts). `texts` is None when the options are
    demonstrably there but unreadable - two-column tables print A and C on one
    line, so C never opens a line - which the caller needs to tell apart from
    "this item is not here at all".
    """
    l0, l1, l2, l3 = letters
    for j in range(lo, hi):
        if _starts(lines[j], l1) is None:
            continue
        k = next((x for x in range(j + 1, hi) if _starts(lines[x], l2)), None)
        if k is None:
            continue
        m = next((x for x in range(k + 1, hi) if _starts(lines[x], l3)), None)
        if m is None:
            continue
        i = next((x for x in range(j - 1, lo - 1, -1) if _starts(lines[x], l0)), None)
        if i is None:
            continue
        end = _option_end(lines, m + 1, hi)
        cuts = ((i, j), (j, k), (k, m), (m, end))
        texts = [" ".join([_starts(lines[a], letters[p])] + lines[a + 1:b]).strip()
                 for p, (a, b) in enumerate(cuts)]
        return i, end, texts

    for i in range(lo, hi):                     # all four run onto one line
        run = split_run_together(lines[i], letters)
        if run:
            return i, i + 1, run

    # Present but unreadable. `hi` as the end is deliberate: the text after an
    # unreadable block is figure wreckage, and calling it a stimulus would
    # wrongly condemn every item that follows.
    where = {c: next((x for x in range(lo, hi) if _starts(lines[x], c)), None) for c in letters}
    if all(v is not None for v in where.values()):
        return min(where.values()), hi, None
    two_col = next((x for x in range(lo, hi) if _starts(lines[x], l0)
                    and re.search(rf"(?:^|\s){l2}\s+\S", lines[x])), None)
    if two_col is not None:
        return two_col, hi, None
    return None


RUN_TOGETHER = re.compile(r"(?:^|\s)([A-DFGHJ])\s+")


def split_run_together(line: str, letters: str) -> list[str] | None:
    """Recover options when all four are printed on one line.

    "A Sodium, Na, and magnesium, Mg B Boron, B, and carbon, C C Copper..." -
    the letters are delimiters, not text, so splitting on newlines finds one
    option and splitting on the letters finds four. Only accepted when the four
    appear exactly once each and in order, since option text itself contains
    bare capitals ("Boron, B").
    """
    hits = [(m.start(1), m.group(1)) for m in RUN_TOGETHER.finditer(line)]
    picked, want = [], list(letters)
    for at, letter in hits:
        if want and letter == want[0]:
            picked.append(at)
            want.pop(0)
    if want or len(picked) != 4:
        return None
    bounds = picked + [len(line)]
    return [line[bounds[p] + 1:bounds[p + 1]].strip() for p in range(4)]


LOOKAHEAD = 150       # how far past an item's number its options may sit
STIMULUS_WORDS = 40   # this much prose between items is a shared passage
MAX_CANDIDATES = 120
PROSE_LINE_WORDS = 7  # shorter lines than this are table cells, not prose

# An item opens with its number alone on a line, or with its number and the
# start of the stem - which is always capitalised. Requiring that throws out
# the table cells ("3 36 3 36") and inline figures that otherwise flood the
# candidate list in the maths booklets.
def _opener(item_no: int) -> re.Pattern:
    return re.compile(rf"^{item_no}(?:$|\s+(?=[A-Z\u201c\u2018\"'(]))")


def opener_lines(lines: list[str], item_no: int) -> list[int]:
    pattern = _opener(item_no)
    out = []
    for i, line in enumerate(lines):
        if not pattern.match(line):
            continue
        # A bare number inside a consecutive run is a graph axis, not an item:
        # "8 / 7 / 6 / 5 / 4 / 3 / 2 / 1 / 0" down the side of a chart. It has
        # to be the CONSECUTIVE run, not just adjacent digits - social studies
        # numbers the parts of a figure, so item 26 is a bare "26" followed by
        # a bare "3", and a looser rule throws the item away.
        if line.isdigit():
            near = [lines[x] for x in (i - 1, i + 1) if 0 <= x < len(lines)]
            if any(n.isdigit() and abs(int(n) - item_no) == 1 for n in near):
                continue
        out.append(i)
    return out


def prose_words(lines: list[str]) -> int:
    """Words in the lines long enough to be sentences rather than table cells.

    Counting raw words here reads a wide maths table as a reading passage: the
    cells are words too. Passage lines run 10-14 words; table rows run 2-5.
    """
    return sum(len(w) for w in (line.split() for line in lines)
               if len(w) >= PROSE_LINE_WORDS)


def item_starts(lines: list[str], key: dict[int, str],
                alternating: bool) -> tuple[dict[int, int] | None, str]:
    """Find the line that opens each item.

    Alignment hangs on this, so it anchors on the ITEM NUMBER rather than on
    the options. Anchoring on options was the first design and it is wrong: an
    item whose options are unreadable (a two-column table) gets skipped, the
    scan then matches the NEXT item's options, and every label from there on is
    off by one - silently, because each individual item still looks fine.
    Numbers are complete and monotone even when an item's body is garbage.

    Numbers are also ambiguous, in two ways that pull in opposite directions.
    Reading booklets number their passage paragraphs from 1, so "9" at line
    start is often paragraph 9; and maths booklets are full of stray small
    integers in tables. Taking the first plausible candidate for each item in
    turn fails on both - one bad early pick runs the cursor past the items that
    follow, and then nothing matches at all.

    So instead of committing item by item, choose the whole assignment at once:
    over all strictly increasing candidate sequences, take the one that gets
    the most items to actually have their options where they should be. A
    passage paragraph does not, a real item does, and one unreadable item costs
    a point instead of derailing everything after it.
    """
    n_items = max(key)
    cands: dict[int, list[int]] = {}
    good: dict[int, set[int]] = {}
    for item_no in range(1, n_items + 1):
        cands[item_no] = opener_lines(lines, item_no)[:MAX_CANDIDATES]
        good[item_no] = set()
        if key.get(item_no) in ALL_LETTERS:
            letters = letters_for(item_no, alternating)
            for c in cands[item_no]:
                block = find_block(lines, c, min(c + LOOKAHEAD, len(lines)), letters)
                # a later line with the same number between here and the
                # options means THAT one opens the item and this one is prose
                if block and not any(c < other < block[0] for other in cands[item_no]):
                    good[item_no].add(c)
        if not cands[item_no]:
            return None, f"no line opens item {item_no} of {n_items}"

    # best[i] = (items placed correctly from here on, next choice), scanning back
    nxt: dict[int, dict[int, int]] = {}
    best: dict[int, dict[int, int]] = {n_items + 1: {len(lines): 0}}
    for item_no in range(n_items, 0, -1):
        best[item_no], nxt[item_no] = {}, {}
        later = best[item_no + 1]
        for c in cands[item_no]:
            options = [(later[j] + (c in good[item_no]), -j) for j in later if j > c]
            if not options:
                continue
            score, pick = max(options)          # ties go to the EARLIEST line:
            best[item_no][c] = score            # axis labels and table cells
            nxt[item_no][c] = -pick             # sit after the item they follow
    if not best[1]:
        return None, f"no run of {n_items} numbered lines is in order"

    start = max(best[1], key=lambda c: (best[1][c], -c))
    placed, item_no = {}, 1
    while item_no <= n_items:
        placed[item_no] = start
        start = nxt[item_no][start]
        item_no += 1
    return placed, ""


def parse_form(lines: list[str], key: dict[int, str]) -> tuple[list[dict] | None, str]:
    """Split a booklet into items, or refuse the whole form.

    The key is the authority on how many items there are, which are griddable
    (those answer with a number) and how the options are lettered; this only
    has to agree with it. Disagreeing is the failure mode worth protecting
    against, so a mismatch returns None for the whole form rather than a best
    effort - one relabelled form is worse than one missing form.
    """
    alternating, why = key_scheme(key)
    if alternating is None:
        return None, why
    starts, why = item_starts(lines, key, alternating)
    if starts is None:
        return None, why

    bounds = {n: (starts[n], starts.get(n + 1, len(lines))) for n in starts}
    items = []
    # a reading booklet opens with its first passage, so the very first item is
    # already under a stimulus before any item has been read
    stimulus_seen = prose_words(lines[:starts[min(starts)]]) >= STIMULUS_WORDS
    for item_no in sorted(key):
        lo, hi = bounds[item_no]
        answer = key[item_no]
        if answer not in ALL_LETTERS:
            items.append({"item_no": item_no, "answer": answer, "griddable": True})
            continue
        letters = letters_for(item_no, alternating)
        if answer not in letters:
            return None, f"item {item_no} answers {answer} but its options are {letters}"
        found = find_block(lines, lo, hi, letters)
        start, tail, texts = found if found else (hi, hi, None)

        stem_lines = [s for s in lines[lo:start] if s]
        if stem_lines:
            stem_lines[0] = re.sub(rf"^{item_no}\s*", "", stem_lines[0]).strip()
            stem_lines = [s for s in stem_lines if s]
        # anything left in the region after the options is a stimulus for
        # whatever comes NEXT - a reading passage, most of the time
        items.append({
            "item_no": item_no,
            "stem_lines": stem_lines,
            "letters": letters,
            "options": texts,
            "answer": answer,
            "griddable": False,
            "under_stimulus": stimulus_seen,
        })
        stimulus_seen = stimulus_seen or prose_words(lines[tail:hi]) >= STIMULUS_WORDS
    return items, ""


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------

VISUAL = re.compile(
    r"\b(?:chart|diagram|figure|graph|graphs|map|maps|picture|image|images|photo"
    r"|photograph|illustration|drawing|model|models|table|tables|grid|number line"
    r"|timeline|cartoon|poster|sketch|schematic|shown|show[ns]?\s+(?:above|below|here)"
    r"|pictured|depicted|labeled|label(?:s|ed)?\s+(?:above|below)"
    r"|above|below|following\s+(?:diagram|chart|table|graph|map|figure|model|picture)"
    r"|this\s+(?:diagram|chart|table|graph|map|figure|model|picture|drawing|image)"
    r"|these\s+(?:diagrams|charts|tables|graphs|maps|figures|models|pictures)"
    r"|arrow|arrows|shaded|shading|dotted line|x-axis|y-axis|axis|axes"
    r"|coordinate (?:grid|plane)|scale drawing|net of|cross section"
    r"|ruler|protractor|dot plot|histogram|venn|stem-and-leaf|box plot"
    r"|scatterplot|spinner|tally|bar graph|line graph|pie chart|circle graph"
    # a demonstrative with nothing to point at: the antecedent was a picture
    r"|these\s+(?:polygons|shapes|solids|objects|figures|angles|lines|points"
    r"|triangles|rectangles|drawings|nets|prisms|graphs))\b",
    re.IGNORECASE,
)

# Unicode the booklets only use inside figures: arrows, geometry marks, boxes.
FIGURE_GLYPHS = re.compile(r"[\u2190-\u21FF\u2500-\u27BF\u2B00-\u2BFF\uFFFD\u25A0-\u25FF]")
BAD_ENCODING = re.compile(r"\(cid:|\uFFFD")
NO_VOWEL = re.compile(r"\b(?![A-Z]{2,}\b)[A-Za-z]{4,}\b")

# Fractions, exponents and subscripts are two-dimensional, and flattening them
# into a line of text does not fail loudly - it produces a plausible-looking
# string that means something else. "V = pi(6)^2 h" becomes "V = pi(6)2h",
# "1/2 pt" becomes "1 pt 2 1", and a balanced equation's subscripts pile up at
# the end as "2AgNO + K SO o Ag SO + 2KNO 3 2 4 2 4 3". None of these can be
# repaired from the text alone, so they are detected and dropped.
FLATTENED_MATH = re.compile(
    r"\)\s*\d"                       # (6)2h - a lost exponent
    r"|\b(?:cm|mm|km|in|ft|yd|m|s)\.?\d\b"   # cm2, in.3
    r"|(?:\b\d\b ){3,}"              # a tail of loose subscripts
    r"|[\u00d7\u00f7+\u2212]\s*$"     # an operator with nothing after it
    r"|\u22c5"                       # the dot operator only ever survives as
                                     # wreckage: "9.0 (dot) 3" for the answer 9
    r"|\S {2,}\S"                    # a glyph dropped out of an expression
)

SHARED = re.compile(
    r"\b(?:the\s+)?(?:passage|selection|selections|article|articles|poem|poems|story|stories"
    r"|excerpt|essay|interview|memoir|play|speech\s+above|text\s+box"
    r"|paragraph|paragraphs|stanza|stanzas|line\s+\d+|lines\s+\d+"
    r"|the\s+author|the\s+narrator|the\s+speaker|the\s+poet|the\s+writer"
    r"|both\s+selections|these\s+selections)\b",
    re.IGNORECASE,
)

# STAAR stems are questions or sentence-completion prompts; anything else is
# a fragment the extractor tore out of a figure.
STEM_END = re.compile(r"[?\u2014\u2013:-]\s*$")

DROP_REASONS = (
    "griddable", "parse", "shared_stimulus", "image", "debris", "degenerate",
)


def looks_like_caption(line: str) -> bool:
    """A figure caption or an axis/part label, not prose.

    Captions are short, Title Case, and unpunctuated ("Model of Sun and
    Earth"); labels are shorter still ("Hand = Sun", "100 kg", "Plate").
    Wrapped prose is neither: it is long and mostly lowercase.
    """
    if line.startswith(("•", "-", "\u2022")):
        return False
    if line.endswith((".", "?", "!", ",", ";", ":")):
        return False
    words = line.split()
    if not words:
        return True
    if len(words) <= 5:
        return True
    capitalized = sum(1 for w in words if w[:1].isupper())
    return len(words) <= 9 and capitalized / len(words) >= 0.6


def classify(item: dict) -> str:
    """Return "" to keep, otherwise the reason to drop."""
    stem_lines = item["stem_lines"]
    stem = " ".join(stem_lines).strip()
    if item["options"] is None:          # located, but the layout is unreadable
        return "parse"
    options = [o.strip() for o in item["options"]]
    blob = stem + " \n " + " \n ".join(options)

    if not stem or len(options) != 4 or not all(options):
        return "parse"
    if not STEM_END.search(stem):
        return "parse"
    if len(stem.split()) < 5:
        return "parse"

    if item["under_stimulus"] or SHARED.search(blob):
        return "shared_stimulus"

    if VISUAL.search(blob) or FIGURE_GLYPHS.search(blob):
        return "image"

    if BAD_ENCODING.search(blob) or FLATTENED_MATH.search(blob):
        return "debris"
    if any(not re.search(r"[aeiouyAEIOUY]", w) for w in NO_VOWEL.findall(blob)):
        return "debris"
    if any(looks_like_caption(line) for line in stem_lines[:-1]):
        return "debris"

    if len(set(o.lower() for o in options)) != 4:
        return "degenerate"
    # an option of one or two characters is a fragment ("pt") or a bare figure
    # reference ("A 1", "B 2"), never a real answer
    if any(len(re.sub(r"\W", "", o)) <= 2 for o in options):
        return "degenerate"
    return ""


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def extract_form(year: int, grade: int, subject: str, stats: Counter) -> list[dict]:
    stem = f"{year}-staar-{grade}-{subject}"
    test_pdf = PDF_DIR / f"{stem}-test.pdf"
    key_pdf = next((PDF_DIR / f"{stem}-{s}.pdf" for s in KEY_SUFFIXES
                    if (PDF_DIR / f"{stem}-{s}.pdf").exists()), None)
    if not test_pdf.exists() or key_pdf is None:
        if test_pdf.exists():
            stats["form_no_key"] += 1
            print(f"  {stem}: DROPPED, no answer key")
        return []

    key = parse_key(key_pdf)
    if not key:
        stats["form_no_key"] += 1
        print(f"  {stem}: DROPPED, answer key did not parse")
        return []

    items, why = parse_form(body_lines(page_lines(test_pdf)), key)
    if items is None:
        stats["form_misaligned"] += 1
        stats["dropped_by_misaligned_form"] += len(key)
        print(f"  {stem}: DROPPED whole form, {why}")
        return []

    n_grid = sum(1 for it in items if it["griddable"])
    stats["raw"] += len(key)
    stats["griddable"] += n_grid
    kept = []
    reasons = Counter()
    for item in items:
        if item["griddable"]:
            continue
        reason = classify(item)
        if reason:
            reasons[reason] += 1
            stats[reason] += 1
            continue
        letters = item["letters"]
        kept.append({
            "question": tidy(" ".join(item["stem_lines"])),
            "choices": [tidy(o) for o in item["options"]],
            "gold_idx": letters.index(item["answer"]),
            "hint": None,
            "source": "staar",
            "subject": subject.replace("-", "_"),
            "grade": grade,
            "year": year,
            "item_no": item["item_no"],
        })
    stats["kept"] += len(kept)
    detail = " ".join(f"{r}={reasons[r]}" for r in DROP_REASONS if reasons[r])
    print(f"  {stem}: {len(key)} items, griddable={n_grid}, {detail or 'no drops'} -> kept {len(kept)}")
    return kept


def main():
    ap = argparse.ArgumentParser(
        description="Extract self-contained MC items from released STAAR PDFs. "
                    "OUTPUT IS TEA-COPYRIGHTED AND GITIGNORED - do not commit it.")
    ap.add_argument("--download", action="store_true", help="fetch the PDFs first")
    ap.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    ap.add_argument("--grades", type=int, nargs="+", default=list(GRADES))
    ap.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    ap.add_argument("--out", default=str(STAAR_DIR / "staar_items.jsonl"))
    ap.add_argument("--show", type=int, default=0, help="print N kept items in full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    STAAR_DIR.mkdir(parents=True, exist_ok=True)
    if args.download:
        print("downloading (only 2018/2019 exist as test+key pairs) ...")
        rep = download_all(args.years, args.grades, args.subjects)
        print(f"  test+key: {len(rep['found'])}   test only: {len(rep['test_only'])}   "
              f"key only: {len(rep['key_only'])}   neither: {len(rep['missing'])}")
        for label in ("test_only", "key_only"):
            for form in sorted(rep[label]):
                print(f"    {label}: {form[0]}-staar-{form[1]}-{form[2]}")

    stats = Counter()
    items = []
    print("\nextracting ...")
    for year in sorted(args.years):
        for subject in args.subjects:
            for grade in sorted(args.grades):
                items += extract_form(year, grade, subject, stats)

    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    print(f"\nwrote {len(items)} items -> {args.out}   (COPYRIGHTED, GITIGNORED)")
    print(f"\n{'raw items in the parsed forms':<34} {stats['raw']}")
    for reason in DROP_REASONS:
        if stats[reason]:
            print(f"{'  dropped: ' + reason:<34} {stats[reason]}")
    if stats["form_no_key"]:
        print(f"{'  forms dropped, no key':<34} {stats['form_no_key']}")
    if stats["form_misaligned"]:
        print(f"{'  forms dropped, key misaligned':<34} {stats['form_misaligned']} "
              f"({stats['dropped_by_misaligned_form']} items)")
    print(f"{'kept':<34} {stats['kept']}")

    by_subject = Counter(it["subject"] for it in items)
    print("\nkept per subject")
    for subject, n in by_subject.most_common():
        print(f"  {subject:<16} {n}")
    by_grade = Counter(it["grade"] for it in items)
    print("kept per grade: " + "  ".join(f"g{g}={by_grade[g]}" for g in sorted(by_grade)))

    if args.show:
        import random

        sample = list(items)
        random.Random(args.seed).shuffle(sample)
        seen = Counter()
        picked = []
        for it in sample:                      # spread the sample over subjects
            if seen[it["subject"]] < max(1, args.show // max(1, len(by_subject))) + 2:
                seen[it["subject"]] += 1
                picked.append(it)
            if len(picked) == args.show:
                break
        for n, it in enumerate(picked, 1):
            print(f"\n--- {n}. {it['subject']} grade {it['grade']} {it['year']} "
                  f"item {it['item_no']} ---")
            print(it["question"])
            for i, choice in enumerate(it["choices"]):
                print(f"  {'ABCD'[i]}{' <-- GOLD' if i == it['gold_idx'] else '       '} {choice}")


if __name__ == "__main__":
    main()
