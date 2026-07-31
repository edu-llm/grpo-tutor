"""Turn released Pennsylvania PSSA Item and Scoring Samplers into self-contained
multiple-choice items.

    !!! THE EXTRACTED CONTENT IS NOT REDISTRIBUTABLE. DO NOT COMMIT IT. !!!

Every sampler carries the Commonwealth's notice permitting duplication "by
Pennsylvania educators for local classroom use". This repository is PUBLIC, so
the PDFs and everything derived from them go under `data/state_tests/`, which
is in `.gitignore`; this file is the only thing that gets committed. Rebuild
the data instead of committing it:

    python src/extract_pa.py --download

WHY THIS EXISTS
---------------
Same reason as `staar_extract.py` (read its docstring first - this module
reuses its filter vocabulary almost verbatim): `docs/dataset_choice.md` found
the binding constraint is the oracle hint, not volume, and real grade 3-8 exam
items with human-written distractors sidestep the survey. PSSA is the richest
of the released-item sources because its answer key is not a bare letter.

WHAT THE SOURCE ACTUALLY LOOKS LIKE
-----------------------------------
Probed 12 years x 6 grades x 3 subjects x 2 filename conventions = 432 URLs;
109 exist. Two conventions coexist and neither is a prefix of the other:

    2015-2023  {base}/{year} pssa iss {math|ela|science} grade {n}.pdf
    2024+      {base}/{year}-pssa-{mathematics|ela|science}-grade-{n}-item-sampler.pdf

Note `math`/`ela` in the old scheme against `mathematics`/`ela` in the new one:
guessing `mathematics` or `english language arts` for the old years 404s on all
72 of them, which is how this looked like a math-and-ELA-free source at first.
Science only exists for grades 4 and 8 (that is the real PSSA design, not a
gap), 2015 has no science, 2024 has only grade 8, and 2017/2020/2025 have no
samplers at all.

THE ANSWER KEY IS INLINE, AND SO IS MUCH MORE THAN THE KEY
----------------------------------------------------------
There is no separate key file. Each item is followed by an information table:

    Item Information
    Alignment              A-T.1.1.1
    Answer Key             B
    Depth of Knowledge     1
    p-value A              21%
    p-value B              46% (correct answer)
    ...
    Option Annotations     A. uses the tens place instead of the tenths place
                           B. correct
                           ...

The p-values are the percentage of *real Pennsylvania children* who chose each
option, and the annotations are the item writer's account of the misconception
each distractor is built from. Nothing else in this project has either. They
are carried through as optional `p_values` and `option_rationales` keys so that
"in the zone" can later be defined by how hard an item is for a nine-year-old
rather than by whether a 0.5B model happened to fail it.

`option_rationales` IS NOT A HINT AND MUST NEVER BE SHOWN TO A STUDENT MODEL.
The gold option's entry reads "correct" or "Key: ...", so the field states the
answer outright - it is annotation about the item, in the same category as
`gold_idx`. Read `docs/dataset_choice.md` on what happens to a measurement
when the hint names the answer before using this for anything.

The table is also a second, independent copy of the key: the correct option's
p-value row is annotated "(correct answer)". Where both are present and they
disagree the item is dropped, which is a check `staar_extract.py` could not
make because TEA prints the key once.

THREE LAYOUTS, NOT ONE
----------------------
  * 2018-2023 "stacked": the table header is `Item Information` and each field
    is its own line. This is the clean case and it is most of the corpus.
  * 2016 "side-by-side": the header is `Item Information Option Annotations`
    and the table is two columns that `extract_text` interleaves by vertical
    position, so `Alignment A-T.2.1.3 A. correct` is one line of left column
    followed by one line of right column. Stripping the known left-column
    prefixes recovers the right column in order.
  * 2015 "inline star": no table at all. The option and the rationale for
    choosing it sit side by side on the item itself and the correct one is
    marked with a trailing `*`. This year is DOWNLOADED AND DELIBERATELY NOT
    EXTRACTED; the reasoning is below, because it cost a day and is not
    obvious from the outside.
  * 2024 is stacked, but the files are ADA-tagged and the tagger spells
    abbreviations out: `0.007 cm` extracts as `0.007 c m` and the six-digit
    item code doubles every digit. That is unrepairable from the text, so
    adjacent single-letter tokens are a drop reason (which also catches the
    flattened subscripts in `(C H OH)` in the older science forms).

WHY 2015 IS FETCHED AND THEN THROWN AWAY
----------------------------------------
2015 is twelve forms of roughly fifty items each, more raw items than every
other year combined, and it is still not used. The layout puts the option and
its distractor rationale in two columns of one table row, and `extract_text`
concatenates them:

    A. 1,002     does not do any subtraction, just divides 4 into each value
    A. 0.6  rounds down or truncates
    A. 126,450 miles least number of miles that rounds to 126,500
    A. 6 + 8 = 8 + 6 uses addition instead of multiplication

The separator is five spaces, then two, then one, then one. `layout=True` does
not help: the gap is a real single space in the last two, because the option
text happens to run right up to the rationale column. Splitting on a
whitespace run therefore cannot work, and splitting on a column inferred from
the other rows picks the wrong column often enough to matter - the four
options of the first example align at three different plausible columns and
the majority one truncates `1,554 R2` to `1,554`.

What survives an inline split is not a merely untidy option. It is an option
carrying the examiner's explanation of *why that option is wrong*, attached to
the wrong-answer choices and absent from the right one. A model does not need
to do the arithmetic to answer "which of these four strings has no argument
against it appended". That is the leak `docs/dataset_choice.md` was written
about, manufactured by the parser instead of by the ZPD screen, and it is
worth more than fifty items to not have it. The 2016-2024 layouts print the
rationale in a separate table below the item, where it can be picked up
deliberately as `option_rationales` for all four options or not at all.

HOW THE LABELS ARE KEPT HONEST
------------------------------
Same discipline as Texas, for the same reason - a misaligned key silently
mislabels a whole form, which is worse than extracting nothing.

  * The option lettering is READ OFF each form, never assumed, from the letters
    the p-value rows use. Texas alternates A-D / F-G-H-J by item parity on some
    forms and not others; PSSA happens to letter everything A-D, but a form
    whose scheme cannot be established is refused rather than guessed.
  * Parsing is anchored on ITEM NUMBERS. Every information table is matched to
    the nearest item number above it and those numbers must come out strictly
    increasing, so an item whose body is unreadable costs that item instead of
    shifting every label after it by one.
  * A form whose tables do not parse, or whose item numbers are not in order,
    is dropped whole.

`pdfplumber` is the dependency, imported lazily and deliberately absent from
`requirements.txt`, because `pypdf` drops intra-word spaces on these files:

    python -m venv /tmp/pdfenv && /tmp/pdfenv/bin/python -m pip install pdfplumber
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import paths

STATE_DIR = paths.DATA / "state_tests"
PDF_DIR = STATE_DIR / "pdf_pa"

BASE = ("https://www.pa.gov/content/dam/copapwp-pagov/en/education/documents/"
        "instruction/assessment-and-accountability/pssa/item-and-scoring-samples/")

# The full probe grid. 2017, 2020 and 2025 are in it because they are plausible
# and absent; leaving them in keeps the report honest about what was looked for.
YEARS = (2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
GRADES = (3, 4, 5, 6, 7, 8)
SUBJECTS = ("math", "ela", "science")
NEW_STYLE_FROM = 2024
# Probed and downloaded like everything else, then not parsed. The module
# docstring explains at length; the short version is that its option text
# cannot be separated from its distractor rationale.
UNUSABLE_YEARS = (2015,)

LETTER_SCHEMES = ("ABCD", "FGHJ")
BEYOND = {"ABCD": "E", "FGHJ": "K"}     # the letter a fifth option would use

DROP_REASONS = (
    "open_ended", "multi_part", "no_key", "key_conflict", "parse",
    "shared_stimulus", "image", "flattened_math", "debris", "split_letters",
    "degenerate",
)


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------

def url_for(year: int, grade: int, subject: str) -> str:
    if year >= NEW_STYLE_FROM:
        name = {"math": "mathematics"}.get(subject, subject)
        return BASE + f"{year}-pssa-{name}-grade-{grade}-item-sampler.pdf"
    return BASE + urllib.parse.quote(f"{year} pssa iss {subject} grade {grade}.pdf")


def pdf_path(year: int, grade: int, subject: str):
    return PDF_DIR / f"{year}-pssa-{subject}-grade-{grade}.pdf"


def fetch(url: str, timeout: int = 180, tries: int = 3) -> bytes | None:
    """Return the body only if it is really a PDF.

    pa.gov answers a missing sampler with a 404, but it also drops connections
    under concurrency, and a dropped connection is not evidence of absence. So
    a transport error retries and only a 404 is taken at face value - the first
    version of this probe reported 2015 and half of 2018 as missing purely
    because it believed its own timeouts.
    """
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            return body if body[:5] == b"%PDF-" else None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except (urllib.error.URLError, OSError):
            pass
    return None


def download_all(years, grades, subjects, workers: int = 8) -> dict:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    wanted = [(y, g, s) for y in years for g in grades for s in subjects]

    def one(form):
        year, grade, subject = form
        dest = pdf_path(*form)
        if dest.exists() and dest.stat().st_size > 10_000:
            return form, "cached"
        body = fetch(url_for(*form))
        if body is None:
            return form, "absent"
        dest.write_bytes(body)
        return form, "downloaded"

    report = defaultdict(list)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for form, state in ex.map(one, wanted):
            report[state].append(form)
    report["probed"] = wanted
    return report


# --------------------------------------------------------------------------
# text extraction
# --------------------------------------------------------------------------

# Running heads, footers, page furniture and the six-digit internal item code.
# The item code is NOT matched as a bare number: dropping every short numeric
# line would also delete the stacked numerators and denominators of a flattened
# fraction, and those are the evidence that the fraction was flattened.
BOILERPLATE = re.compile(
    r"^\s*(?:"
    r"PSSA\b.*"
    r"|P\s+S\s+S\s+A\b.*"
    r"|Pennsylvania Department of Education.*"
    r"|Page\s*\d+"
    r"|\d{6}"                       # internal item code
    r"|GO\s*ON|STOP|CONTINUE"
    r"|Go\s+(?:on\s+)?to\s+the\s+next\s+page.*"
    r"|Continued\..*"
    r"|THIS PAGE IS|INTENTIONALLY BLANK\.?"
    r"|After you have checked your work.*"
    r"|and test booklet.*"
    r"|Question\s+\d+\s+in\s+this.*"
    r"|A\s+calculator\s+is\s+permitted.*"
    r"|Questions?\s+[\d\u2013-]+\s+in\s+this.*"
    r"|-{20,}"
    r")\s*$",
    re.IGNORECASE,
)

# The 2024 ADA tagger renders the six-digit item code with every digit doubled
# ("1137159" -> "11113377115599"), which no other rule would recognise.
DOUBLED_CODE = re.compile(r"^\s*(\d)\1(\d)\2(\d)\3(\d)\4(\d)\5(\d)\6\s*$")

SECTION = re.compile(
    r"^\s*(?:MULTIPLE[- ]CHOICE\s+(?:ITEMS?|QUESTIONS?)"
    r"|OPEN[- ]ENDED\s+(?:ITEMS?|QUESTIONS?)"
    r"|EVIDENCE[- ]BASED\s+SELECTED[- ]RESPONSE\s+QUESTIONS?"
    r"|TEXT[- ]DEPENDENT\s+ANALYSIS(?:\s+(?:PROMPT|QUESTION))?"
    r"|Item-Specific Scoring Guideline"
    r"|Scoring Guide"
    r"|Top-Scoring (?:Student )?Response.*"
    r"|STUDENT RESPONSE.*"
    r")\s*$",
    re.IGNORECASE,
)

# Where the booklet stops describing itself and starts asking questions. The
# front matter is not merely noise: the "Item and Scoring Sampler Format" page
# prints a blank EXAMPLE information table, complete with its own
# `Item Information` header and four `p-value` rows, and a parser that counts
# tables without trimming this finds one more table than there are items and
# refuses every form in the corpus.
BODY_START = re.compile(
    r"^\s*(?:MULTIPLE[- ]CHOICE\s+(?:ITEMS?|QUESTIONS?)"
    r"|PASSAGES?\s+\d+(?:\s+AND\s+\d+)?)\s*$",
    re.IGNORECASE,
)

# A passage has started. Once one has, every later item in the booklet is under
# a shared stimulus until the end of the form - PSSA never returns to
# stand-alone items after the reading passages begin.
PASSAGE_MARKER = re.compile(
    r"^\s*(?:PASSAGES?\s+\d+(?:\s+AND\s+\d+)?"
    r"|Read the (?:passage|poem|drama|selection|article|story|text|following)\b.*"
    r"|Use the (?:passage|passages|poem|article|selection)\b.*"
    r"|The next \w+ passages\b.*"
    r")\s*$",
    re.IGNORECASE,
)

# "Use the drawing below to answer question 12." Usually this is the opening
# sentence of item 12 itself, where the VISUAL filter sees it and the item is
# dropped. Sometimes it is a heading above the picture instead, several lines
# ABOVE item 12's number, and then item 12's own stem says only "Which
# statement best explains why the ice cubes begin to melt?" - which reads as a
# perfectly self-contained question and is not one. So the directive is parsed
# for the question numbers it names and those items are condemned by number.
STIMULUS_DIRECTIVE = re.compile(
    r"^(?:Directions:\s*)?U\s?se the .{0,80}?to answer (?:the )?questions?"
    r"([\d\s,\u2013\u2014-]*(?:through\s*\d+)?)\.?\s*$",
    re.IGNORECASE,
)

TIDY = (("\u2010", "-"), ("\u2011", "-"), ("\u2212\u2212", "\u2212"))

# 2015, 2022 and 2023 are typeset with a thin space in front of every period,
# which pdfplumber widens into a real one: "1 . Multiply: 372 × 108", "0 .305",
# "A-T .2 .1 .1". English never puts whitespace before a period, so collapsing
# it is unambiguous - and it has to happen before anything else looks at the
# text, because until it does no line in those 36 forms opens with an item
# number and every one of them is refused for having no items in it.
SPACED_PERIOD = re.compile(r"(?<=\S)[ \t]+\.")


def repair(line: str) -> str:
    for bad, good in TIDY:
        line = line.replace(bad, good)
    return SPACED_PERIOD.sub(".", line)


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tidy(text: str) -> str:
    return squash(repair(text))


def load_pages(path) -> list[str]:
    """Page texts, cached beside the PDF.

    The cache lives in `pdf_pa/` rather than anywhere else on purpose: that
    directory is already gitignored as copyrighted content, and the cache is
    copyrighted content. Extraction is a couple of minutes for the corpus and
    the filters want more iterations than that.
    """
    cache = path.with_suffix(".text.txt")
    if cache.exists() and cache.stat().st_size > 100:
        return cache.read_text().split("\f")

    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        pages = [(p.extract_text() or "") for p in pdf.pages]
    cache.write_text("\f".join(pages))
    return pages


def clean(pages: list[str]) -> tuple[list[str], list[int]]:
    """Non-boilerplate lines, and the page each came from."""
    lines: list[str] = []
    page_of: list[int] = []
    for p, text in enumerate(pages):
        for raw in text.split("\n"):
            line = raw.strip()
            if not line or BOILERPLATE.match(line) or DOUBLED_CODE.match(line):
                continue
            line = repair(line)
            lines.append(line)
            page_of.append(p)
    start = next((i for i, l in enumerate(lines) if BODY_START.match(l)), 0)
    return lines[start:], page_of[start:]


# --------------------------------------------------------------------------
# the item information table
# --------------------------------------------------------------------------

TABLE_HEADER = re.compile(
    r"^(?:Item Information(?:\s+Option Annotations)?"
    r"|Category\s+Item-Specific Information)\s*$", re.IGNORECASE)
# The open-ended scoring guides repeat the header as "#17 Item Information".
# They are not MC tables and the anchor `^` is what keeps them out.

OPENER = re.compile(r"^(\d{1,2})\.(?:\s|$)")
OPTION = re.compile(r"^([A-Z])\s*[.)]\s*(.*)$")

ANSWER_KEY = re.compile(r"^Answer\s+Key\s+([A-Z])(?:\s|$)", re.IGNORECASE)
EBSR_KEY = re.compile(r"^Answer\s+Key:\s*Part\s+(?:One|Two)", re.IGNORECASE)
MEAN_SCORE = re.compile(r"^(?:Mean\s+(?:Student\s+)?Score|Mean\s+Score)\b", re.IGNORECASE)
PVALUE_ROW = re.compile(r"^p-values?\s+([A-Z])\s+(N/A|\d+)%?(\s*\(correct answer\))?",
                        re.IGNORECASE)
# Matched as a PREFIX, not as a whole line. 2016 prints the p-value grid in the
# left column of the same table whose right column is the annotations, and
# pdfplumber interleaves them by height, so the grid's header row arrives as
# "A B C D the objects shown." with somebody else's sentence welded to it.
LETTER_ROW = re.compile(r"^([A-Z])\s+([A-Z])\s+([A-Z])\s+([A-Z])(?:\s|$)")
PERCENT_ROW = re.compile(r"^(\d{1,3})%\s+(\d{1,3})%\s+(\d{1,3})%\s+(\d{1,3})%(?:\s|$)")
ANNOTATIONS = re.compile(r"^Option\s+Annotations\s*(.*)$", re.IGNORECASE)

# 2016 packs two columns onto one line. These are the left-column fields; what
# is left after they are removed is the right column, and pdfplumber emits the
# lines in vertical order, so the right column reassembles in order.
LEFT_COLUMN = re.compile(
    r"^(?:Alignment(?:\s+\S+)?|Answer\s+Key\s+[A-Z]|Depth\s+of\s+Knowledge\s+\d+"
    r"|p-values|Mean\s+Score\s+[\d.]+|Item Information|Option Annotations)\s*",
    re.IGNORECASE)


def parse_table(lines: list[str]) -> dict:
    """Read one item information table.

    Returns what was found rather than raising: the caller decides whether a
    missing key means "open-ended item" or "broken form".
    """
    out: dict = {"answer": None, "p_values": {}, "correct_marked": None,
                 "open_ended": False, "multi_part": False, "rationales": {}}
    annotation_from = None
    for i, line in enumerate(lines):
        if EBSR_KEY.match(line):
            out["multi_part"] = True
        m = ANSWER_KEY.match(line)
        if m and out["answer"] is None:
            out["answer"] = m.group(1).upper()
        if MEAN_SCORE.match(line):
            out["open_ended"] = True
        m = PVALUE_ROW.match(line)
        if m:
            letter, value, correct = m.group(1).upper(), m.group(2), m.group(3)
            out["p_values"][letter] = None if value == "N/A" else int(value) / 100
            if correct:
                out["correct_marked"] = letter
        if annotation_from is None and ANNOTATIONS.match(line):
            annotation_from = i

    if not out["p_values"]:                             # the 2016 grid
        letters = next((LETTER_ROW.match(l) for l in lines if LETTER_ROW.match(l)), None)
        percents = next((PERCENT_ROW.match(l) for l in lines if PERCENT_ROW.match(l)), None)
        if letters and percents:
            for letter, pct in zip(letters.groups(), percents.groups()):
                out["p_values"][letter.upper()] = int(pct) / 100

    if annotation_from is None:
        # 2016 interleaves the annotation column with the table fields rather
        # than heading it; stripping the fields leaves the column, in order.
        body = [LEFT_COLUMN.sub("", l) for l in lines]
        body = [PERCENT_ROW.sub("", LETTER_ROW.sub("", l)).strip() for l in body]
    else:
        body = list(lines[annotation_from:])
        body[0] = ANNOTATIONS.match(body[0]).group(1)
    out["rationales"] = parse_rationales(body)
    return out


def parse_rationales(lines: list[str]) -> dict[str, str]:
    """Per-option annotations, or nothing.

    Only accepted when all four letters open a line, exactly once each and in
    order. The ELA annotations are a single paragraph about the item as a whole
    rather than four per-option notes, and a paragraph that happens to begin
    "A. " somewhere in the middle would otherwise be cut into nonsense.
    """
    starts: dict[str, int] = {}
    for i, line in enumerate(lines):
        m = re.match(r"^([A-Z])[.)]\s+(\S.*)$", line.strip())
        if m and m.group(1) not in starts:
            starts[m.group(1)] = i
    for scheme in LETTER_SCHEMES:
        want = list(scheme)
        if set(starts) >= set(want) and [starts[c] for c in want] == sorted(starts[c] for c in want):
            bounds = [starts[c] for c in want] + [len(lines)]
            out = {}
            for k, letter in enumerate(want):
                body = " ".join(lines[bounds[k]:bounds[k + 1]]).strip()
                out[letter] = tidy(re.sub(rf"^{letter}[.)]\s*", "", body))
            if all(out.values()):
                return out
    return {}


# --------------------------------------------------------------------------
# form parsing: the 2016-2024 table layouts
# --------------------------------------------------------------------------

STIMULUS_WORDS = 40
PROSE_LINE_WORDS = 7


def prose_words(lines: list[str]) -> int:
    """Words on lines long enough to be sentences rather than table cells."""
    return sum(len(w) for w in (line.split() for line in lines)
               if len(w) >= PROSE_LINE_WORDS)


def scheme_of(tables: list[dict], option_letters: Counter) -> tuple[str | None, str]:
    """Which four letters this form uses, read off the form.

    The p-value rows name every option whether or not it was correct, so they
    are the authority; the answer keys alone would only ever show the letters
    that happened to be right. When a form has no p-values at all the letters
    that actually open option lines stand in, and a form that matches no known
    scheme is refused rather than guessed - see `staar_extract.key_scheme` for
    the Texas form this rule was written for, where the scheme really does vary
    between releases of the same test.
    """
    seen = set()
    for t in tables:
        seen |= set(t["p_values"])
    if not seen and option_letters:
        # No form in the corpus needs this - every one of them prints p-value
        # rows, even the 2024 science sampler whose values are all "N/A" - but
        # a form that did would have to fall back on the letters that open
        # option lines. A letter has to be common to count: "choose two" items
        # carry a fifth option and a stray capital-plus-period in a stem
        # contributes a letter of its own, and neither is part of the scheme.
        floor = max(option_letters.values()) * 0.2
        seen = {c for c, n in option_letters.items() if n >= floor}
    for scheme in LETTER_SCHEMES:
        if seen == set(scheme):
            return scheme, ""
    if not seen:
        return None, "no option letters anywhere in the form"
    return None, f"option letters {''.join(sorted(seen))} match no known scheme"


def split_options(region: list[str], scheme: str) -> tuple[list[str] | None, int, str]:
    """The four option texts inside one item's question region.

    Returns (texts, n_runs, why). `n_runs` counts how many times the scheme's
    first letter opens a line: an evidence-based selected-response item is one
    numbered question with two complete A-D sets under "Part One" and "Part
    Two", and it has to be recognised rather than silently truncated to its
    first half.
    """
    at: dict[str, list[int]] = defaultdict(list)
    for i, line in enumerate(region):
        m = OPTION.match(line)
        if m and m.group(1) in scheme + BEYOND[scheme]:
            at[m.group(1)].append(i)
    n_runs = len(at[scheme[0]])
    # A fifth option means a "choose two" item, and reading only the first four
    # would quietly hand back a four-way item whose answer is half an answer -
    # option D would also swallow option E's text, since D's text runs to the
    # next thing the parser recognises.
    if at[BEYOND[scheme]]:
        return None, 2, "a fifth option letter opens a line"
    if any(letter not in at for letter in scheme):
        return None, n_runs, "not all four option letters open a line"
    first = [at[letter][0] for letter in scheme]
    if first != sorted(first) or len(set(first)) != 4:
        return None, n_runs, "option letters are out of order"

    bounds = first + [len(region)]
    texts = []
    for k, letter in enumerate(scheme):
        body = " ".join(region[bounds[k]:bounds[k + 1]])
        texts.append(OPTION.match(body).group(2).strip() if OPTION.match(body) else "")
    return texts, n_runs, ""


def anchor_tables(tables: list[int], openers: dict[int, int]) -> tuple[list, int]:
    """Match each information table to the item number that introduces it.

    Anchoring on numbers rather than on option letters is the whole point (see
    `staar_extract.item_starts` for the version of this that got it wrong
    first): an item whose body is unreadable must cost that item, not shift
    every label after it by one.

    Numbers alone are not enough, though, because plenty of other lines open
    with one. A stem that says "Divide 12 by 3." is a candidate, a figure part
    numbered "0." is a candidate, and taking the nearest candidate above each
    table produces sequences like [1, 2, 3, 0, 5, ...] - the fourth item
    labelled with a figure's part number. So a candidate additionally has to
    continue the count and has to lie after the previous item's table; the
    latest candidate that does both is the opener. A table with no such
    candidate is left unresolved rather than mislabelled, and a form where too
    many are unresolved is refused whole.
    """
    placed: list[tuple[int, int, int]] = []
    unresolved = 0
    last_number, last_table = 0, -1
    for t in tables:
        cands = [i for i, n in openers.items() if last_table < i < t and n > last_number]
        if not cands:
            unresolved += 1
            continue
        opener = max(cands)
        placed.append((openers[opener], opener, t))
        last_number, last_table = openers[opener], t
    return placed, unresolved


def parse_form(pages: list[str]) -> tuple[list[dict] | None, str]:
    """Split one sampler into items, or refuse the whole form."""
    lines, page_of = clean(pages)
    tables = [i for i, l in enumerate(lines) if TABLE_HEADER.match(l)]
    if not tables:
        return None, "no item information tables (2015 layout, handled separately)"

    openers = {i: int(OPENER.match(l).group(1)) for i, l in enumerate(lines)
               if OPENER.match(l)}
    if not openers:
        return None, "no numbered item openers"

    placed, unresolved = anchor_tables(tables, openers)
    if not placed:
        return None, "no item information table could be matched to an item number"
    if unresolved > max(1, len(tables) // 4):
        return None, (f"{unresolved} of {len(tables)} information tables have no item "
                      f"number above them - this is not the layout the parser expects")

    # The table runs to the next item number, except that the next item's
    # picture and its heading sit in that gap too, and without this the last
    # option's annotation ends "...not lose chemical energy to the air. 83% 9%
    # 4% 4% Use the drawing below to answer question 13. Motions of a Metal
    # Ball strong magnet motion 1 metal ball motion 2".
    parsed_tables = []
    for item_no, opener, table in placed:
        end = min([i for i in openers if i > table] + [len(lines)])
        stop = next((i for i in range(table + 1, end)
                     if STIMULUS_DIRECTIVE.match(lines[i]) or PASSAGE_MARKER.match(lines[i])
                     or SECTION.match(lines[i])), end)
        parsed_tables.append(parse_table(lines[table + 1:stop]))

    option_letters = Counter(OPTION.match(l).group(1) for l in lines if OPTION.match(l))
    scheme, why = scheme_of(parsed_tables, option_letters)
    if scheme is None:
        return None, why

    # A page carrying neither an item number nor a table, but plenty of prose,
    # is a reading passage, and once one has appeared every later item leans on
    # it. Only pages at or after the first item count: the booklet opens with
    # several pages of scoring guidelines that are prose by any measure, and
    # counting those condemned all 16 items of the 2016 science forms - which
    # have no passages at all - as passage-dependent.
    structural = {page_of[i] for i in list(openers) + tables}
    by_page: dict[int, list[str]] = defaultdict(list)
    for i, line in enumerate(lines):
        by_page[page_of[i]].append(line)
    body_page = page_of[placed[0][1]]
    passage_pages = {p for p, page_lines in by_page.items()
                     if p >= body_page and p not in structural
                     and prose_words(page_lines) >= STIMULUS_WORDS}
    # Explicit passage headings still count wherever they are: in the reading
    # booklets PASSAGE 1 precedes item 1, so it is never on a body page.
    marker_pages = {page_of[i] for i, l in enumerate(lines) if PASSAGE_MARKER.match(l)}
    first_passage = min(passage_pages | marker_pages, default=None)
    directed = directed_items(lines, openers)

    items = []
    for (item_no, opener, table), info in zip(placed, parsed_tables):
        region = lines[opener:table]
        region = [l for l in region if not SECTION.match(l)]
        texts, n_runs, why_opt = split_options(region, scheme)
        stem_lines: list[str] = []
        if texts is not None:
            first_opt = next(i for i, l in enumerate(region)
                             if OPTION.match(l) and OPTION.match(l).group(1) == scheme[0])
            stem_lines = [l for l in region[:first_opt] if l]
            if stem_lines:
                stem_lines[0] = re.sub(rf"^{item_no}\.\s*", "", stem_lines[0]).strip()
                stem_lines = [l for l in stem_lines if l]
        items.append({
            "item_no": item_no,
            "stem_lines": stem_lines,
            "options": texts,
            "n_runs": n_runs,
            "scheme": scheme,
            "why_opt": why_opt,
            "under_stimulus": first_passage is not None and page_of[opener] >= first_passage,
            "directed": item_no in directed,
            **info,
        })
    return items, ""


def directed_items(lines: list[str], openers: dict[int, int]) -> set[int]:
    """Item numbers that a stimulus directive points a picture at.

    The directive normally names them ("...to answer question 12", or a range
    written either "5-8" or "5 through 8"). When it names none it is a heading
    for whatever comes next, so it takes the following item.
    """
    directed: set[int] = set()
    for i, line in enumerate(lines):
        m = STIMULUS_DIRECTIVE.match(line)
        if not m:
            continue
        numbers = [int(n) for n in re.findall(r"\d+", m.group(1) or "")]
        if numbers:
            directed |= set(range(min(numbers), max(numbers) + 1))
        else:
            following = min((j for j in openers if j > i), default=None)
            if following is not None:
                directed.add(openers[following])
    return directed


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------

# Straight from staar_extract.VISUAL - the two states write stems in the same
# idiom and the vocabulary transferred without a single addition needed beyond
# PSSA's own "Use the ... below to answer the question" stock phrase.
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
    r"|these\s+(?:polygons|shapes|solids|objects|figures|angles|lines|points"
    r"|triangles|rectangles|drawings|nets|prisms|graphs))\b",
    re.IGNORECASE,
)

SHARED = re.compile(
    r"\b(?:the\s+)?(?:passage|selection|selections|passages|article|articles|poem|poems"
    r"|story|stories|excerpt|essay|interview|memoir|play|drama|speech|text\s+box"
    r"|paragraph|paragraphs|stanza|stanzas|line\s+\d+|lines\s+\d+"
    r"|the\s+author|the\s+narrator|the\s+speaker|the\s+poet|the\s+writer"
    r"|both\s+selections|these\s+selections|Part\s+One|Part\s+Two)\b",
    re.IGNORECASE,
)

FIGURE_GLYPHS = re.compile(r"[\u2190-\u21FF\u2500-\u27BF\u2B00-\u2BFF\uFFFD\u25A0-\u25FF]")
# A private-use codepoint is a glyph the font never mapped to Unicode, so what
# it meant is not recoverable. PSSA uses them for the symbol-font operators:
# "3 x 0.5x" comes out as "3 \uf097 0.5x", which reads as a plausible expression
# with the multiplication signs simply missing.
# `\d{9,}` is the 2024 tagger's doubled item code ("1137159" -> "11113377115599")
# landing inside an option instead of on a line of its own, where the
# line-level rule would have caught it.
BAD_ENCODING = re.compile(r"\(cid:|\uFFFD|[\uE000-\uF8FF]|\b\d{9,}\b")
NO_VOWEL = re.compile(r"\b(?![A-Z]{2,}\b)[A-Za-z]{4,}\b")

# A stacked fraction, exponent or subscript is two-dimensional and flattening
# it produces a plausible-looking string that means something else. PSSA prints
# the fraction bar as a run of underscores or as `}`, and the numerator and
# denominator land on their own lines, so "Add: 7/8 + 11/5" arrives as
# "7 11 / 1. Add: + / 8 5". None of it can be repaired from the text.
FLATTENED_MATH = re.compile(
    r"_{2,}"                          # the printed fraction bar
    r"|\}"                            # ...and its other rendering
    r"|\)\s*\d"                       # (6)2h - a lost exponent
    r"|\b(?:cm|mm|km|in|ft|yd|m|s)\.?\d\b"
    r"|(?:\b\d\b ){3,}"
    r"|[\u00d7\u00f7+\u2212]\s*$"     # an operator with nothing after it
    r"|\u22c5"
    # A glyph dropped out of an expression and left its space behind. This is
    # how "3 . 0.5x + 4 . 0.8" arrives once the dot operator is gone, and the
    # result is an option that reads like arithmetic and is not.
    r"|\S {2,}\S"
    # A radical loses its index and its bar, which changes what it means
    # without changing how it reads: the cube root of 8 extracts as "\u221a 8", whose
    # answer is 2 in the key and 2.83 on the page. Some forms mis-decode the
    # sign itself, and it arrives as a lone accented Latin-1 letter.
    r"|[\u221a-\u221d]"
    r"|(?<![A-Za-z\u00c0-\u00ff])[\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff](?![A-Za-z\u00c0-\u00ff])"
    # An exponent merged with its base: "1 \u00d7 10\u2076" comes out "1 \u00d7 106", and the
    # accompanying "10^(6+6)" comes out "10(6 + 6)".
    r"|\u00d7\s*10\d"
    # The two halves of a stacked fraction, stranded at the end of the line
    # they were printed beside: "...divide 20 by 4 ... to get $5.50. 4 1"
    r"|(?:\s\b\d\b){2,}\s*$"
)

# Everything that is left over once the digits and separators are removed is
# nothing, so the option never had any words in it: "0 9" was a fraction.
LOOSE_NUMBERS = re.compile(r"^\d[\d\s.,]*\s\d[\d.,]*$")
# A continuation line made only of digits and separators is the other half of a
# stacked fraction. It is checked per line, before the lines are joined, since
# joining is exactly what makes it invisible.
NUMERIC_LINE = re.compile(r"^[\d.,\s]+$")

# The 2024 ADA tagger splits abbreviations into letters ("cm" -> "c m", "PSSA"
# -> "P S S A"); the older science forms do the same to chemical subscripts
# ("C H OH"). `a` and `I` are excluded because they are words.
SPLIT_LETTERS = re.compile(r"(?<![A-Za-z])[b-hj-zB-HJ-Z]\s+[b-hj-zB-HJ-Z](?![A-Za-z])")

MIN_STEM_WORDS = 3   # "Subtract: 124.8 - 9.34" is a real and complete item

RATIONALE_WRECKAGE = re.compile(r"\}|_{2,}|\(cid:|[\uE000-\uF8FF]|\uFFFD")


def classify(item: dict) -> str:
    """Return "" to keep, otherwise the reason to drop."""
    if item["multi_part"] or item["n_runs"] > 1:
        return "multi_part"
    if item["options"] is None:
        return "open_ended" if item["open_ended"] else "parse"
    if item["answer"] is None:
        return "open_ended" if item["open_ended"] else "no_key"
    if item["answer"] not in item["scheme"]:
        return "no_key"
    if item["correct_marked"] and item["correct_marked"] != item["answer"]:
        return "key_conflict"

    stem_lines = [l for l in item["stem_lines"] if l]
    stem = " ".join(stem_lines).strip()
    options = [o.strip() for o in item["options"]]
    blob = stem + " \n " + " \n ".join(options)

    if not stem or len(options) != 4 or not all(options):
        return "parse"
    if len(stem.split()) < MIN_STEM_WORDS:
        return "parse"

    if item["under_stimulus"] or SHARED.search(blob):
        return "shared_stimulus"
    if item.get("directed") or VISUAL.search(blob) or FIGURE_GLYPHS.search(blob):
        return "image"

    # Per option as well as over the whole item: the end-of-line rules below
    # only mean anything against a single option's own end.
    if FLATTENED_MATH.search(blob) or any(FLATTENED_MATH.search(o) for o in options):
        return "flattened_math"
    if any(NUMERIC_LINE.match(l) for l in stem_lines[1:]):
        return "flattened_math"
    if any(LOOSE_NUMBERS.match(o) for o in options):
        return "flattened_math"
    if SPLIT_LETTERS.search(blob):
        return "split_letters"
    if BAD_ENCODING.search(blob):
        return "debris"
    if any(not re.search(r"[aeiouyAEIOUY]", w) for w in NO_VOWEL.findall(blob)):
        return "debris"

    if len({o.lower() for o in options}) != 4:
        return "degenerate"
    if any(len(re.sub(r"\W", "", o)) < 1 for o in options):
        return "degenerate"
    return ""


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def extract_form(year: int, grade: int, subject: str, stats: Counter,
                 quiet: bool = False) -> list[dict]:
    path = pdf_path(year, grade, subject)
    stem_name = path.stem
    if not path.exists():
        return []

    if year in UNUSABLE_YEARS:
        stats["form_unusable"] += 1
        return []

    items, why = parse_form(load_pages(path))
    if items is None:
        stats["form_dropped"] += 1
        print(f"  {stem_name}: DROPPED whole form, {why}")
        return []

    stats["raw"] += len(items)
    reasons = Counter()
    kept = []
    for item in items:
        reason = classify(item)
        if reason:
            reasons[reason] += 1
            stats[reason] += 1
            continue
        scheme = item["scheme"]
        record = {
            "question": tidy(" ".join(item["stem_lines"])),
            "choices": [tidy(o) for o in item["options"]],
            "gold_idx": scheme.index(item["answer"]),
            "hint": None,
            "source": "pa",
            "subject": subject,
            "grade": grade,
            "year": year,
            "item_no": item["item_no"],
        }
        p_values = {c: item["p_values"][c] for c in scheme
                    if item["p_values"].get(c) is not None}
        if len(p_values) == 4:
            record["p_values"] = p_values
            stats["with_p_values"] += 1
        # The annotations quote the item back at you, so they carry the same
        # flattened fractions the item did ("is } of the value ... 10"). The
        # item itself is fine - it passed the filters - so this drops only the
        # note, not the question.
        rationales = {c: item["rationales"][c] for c in scheme if item["rationales"].get(c)}
        if any(RATIONALE_WRECKAGE.search(r) for r in rationales.values()):
            rationales = {}
        if len(rationales) == 4:
            record["option_rationales"] = rationales
            stats["with_rationales"] += 1
        kept.append(record)

    stats["kept"] += len(kept)
    if not quiet:
        detail = " ".join(f"{r}={reasons[r]}" for r in DROP_REASONS if reasons[r])
        print(f"  {stem_name}: {len(items)} items, {detail or 'no drops'} -> kept {len(kept)}")
    return kept


def gold_stats(items: list[dict]) -> tuple[float, float, int]:
    lengths = [len(it["choices"][it["gold_idx"]].split()) for it in items]
    if not lengths:
        return 0.0, 0.0, 0
    return (statistics.median(lengths),
            sum(1 for n in lengths if n == 1) / len(lengths),
            max(lengths))


def main():
    ap = argparse.ArgumentParser(
        description="Extract self-contained MC items from released PSSA Item and "
                    "Scoring Samplers. OUTPUT IS PENNSYLVANIA-COPYRIGHTED AND "
                    "GITIGNORED - do not commit it.")
    ap.add_argument("--download", action="store_true", help="probe and fetch the PDFs first")
    ap.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    ap.add_argument("--grades", type=int, nargs="+", default=list(GRADES))
    ap.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    ap.add_argument("--out", default=str(STATE_DIR / "pa_items.jsonl"))
    ap.add_argument("--show", type=int, default=0, help="print N kept items in full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    if args.download:
        print("probing pa.gov (both filename conventions) ...")
        rep = download_all(args.years, args.grades, args.subjects)
        have = len(rep["downloaded"]) + len(rep["cached"])
        print(f"  probed {len(rep['probed'])}   found {have}   404 {len(rep['absent'])}")
        by_year = Counter(y for y, _, _ in rep["downloaded"] + rep["cached"])
        for year in sorted(by_year):
            print(f"    {year}: {by_year[year]} forms")

    stats = Counter()
    items: list[dict] = []
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
    if stats["form_dropped"]:
        print(f"{'  forms dropped whole':<34} {stats['form_dropped']}")
    if stats["form_unusable"]:
        print(f"{'  forms skipped, 2015 layout':<34} {stats['form_unusable']}")
    print(f"{'kept':<34} {stats['kept']}")
    print(f"{'  with p_values':<34} {stats['with_p_values']}")
    print(f"{'  with option_rationales':<34} {stats['with_rationales']}")

    by_subject = Counter(it["subject"] for it in items)
    print("\nkept per subject")
    for subject, n in by_subject.most_common():
        print(f"  {subject:<10} {n}")
    by_grade = Counter(it["grade"] for it in items)
    print("kept per grade: " + "  ".join(f"g{g}={by_grade[g]}" for g in sorted(by_grade)))
    by_year = Counter(it["year"] for it in items)
    print("kept per year:  " + "  ".join(f"{y}={by_year[y]}" for y in sorted(by_year)))

    median, single, longest = gold_stats(items)
    print(f"\ngold answer: median {median:.0f} words, {single:.0%} single-word, "
          f"longest {longest}")

    if args.show:
        import random

        sample = list(items)
        random.Random(args.seed).shuffle(sample)
        seen = Counter()
        picked = []
        for it in sample:
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
                letter = "ABCD"[i]
                gold = " <-- GOLD" if i == it["gold_idx"] else "        "
                pv = it.get("p_values", {}).get(letter)
                share = f"  [{pv:.0%} of students]" if pv is not None else ""
                print(f"  {letter}{gold} {choice}{share}")
            for letter, why in it.get("option_rationales", {}).items():
                print(f"      {letter}: {why}")


if __name__ == "__main__":
    main()
