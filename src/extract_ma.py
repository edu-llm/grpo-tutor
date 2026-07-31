"""Turn released Massachusetts MCAS test PDFs into self-contained multiple-choice items.

    !!! THE EXTRACTED CONTENT IS NOT REDISTRIBUTABLE. DO NOT COMMIT IT. !!!

Every MCAS release carries the Commonwealth's copyright notice. This repository
is PUBLIC; the test content is DESE's and only the method below is ours. The
downloaded PDFs and everything derived from them are written under
`data/state_tests/`, which is in `.gitignore`. If you ever find that directory
staged, the answer is `git restore --staged`, not `git add -f`. Rebuild it:

    python src/extract_ma.py --download

WHY THIS EXISTS
---------------
Same reason as `staar_extract.py`: `docs/dataset_choice.md` found that the
binding constraint on the HF corpora is the oracle *hint*, not volume, and real
released exam items sidestep the survey by being a different kind of object.
Massachusetts adds two things Texas does not have: grade 8 Civics, which is the
only social-studies content in the project, and a Science and
Technology/Engineering strand at grades 5 and 8. Like STAAR, there is no hint
field; see `docs/dataset_choice.md`.

WHAT THE SOURCE ACTUALLY LOOKS LIKE
-----------------------------------
The index at https://www.doe.mass.edu/mcas/release.html is year-selectable but
the `?yr=` parameter does not actually serve old pages - every value returns the
current year's document. The real per-year pages are

    https://www.doe.mass.edu/mcas/{year}/release/

and their PDFs are linked with bare relative names whose spelling changed in
2023 (`gr8-math.pdf` became `g8-math.pdf`), so the links are scraped rather than
guessed. Only 2019 and 2021-2026 exist: `/mcas/2013..2018/release/` all 404 on
the live site. The pre-2019 paper-era booklets survive only on the Wayback
Machine, and are deliberately NOT used - see the note at the bottom of this
docstring.

Each booklet is one grade x one subject, with the answer key INLINE, as a
"Released Operational Items" table on the last content pages. `pypdf` mangles
these PDFs (it strips intra-word spaces); `pdfplumber` does not, so that is the
dependency, imported lazily - it is not in `requirements.txt`.

    python -m venv /tmp/pdfenv_ma && /tmp/pdfenv_ma/bin/pip install pdfplumber

HOW AN ITEM IS FOUND, AND WHY NOT THE OBVIOUS WAYS
--------------------------------------------------
Two anchors that work on STAAR are unavailable here, and both fail silently,
which is the dangerous kind:

  * OPTION LETTERS. MCAS letters everything A-D with no parity alternation, but
    the letters are not text: they are glyphs in a font called `CircleFrameMT`
    (a letter inside a drawn circle). They extract as bare "A"/"B"/"C"/"D" on
    their own lines, indistinguishable from a stem line that happens to begin
    with a capital. Worse, grade 8 Civics mixes three-option and four-option
    questions, so any rule that assumes a fourth is wrong on the one subject we
    most want.

  * ITEM NUMBERS. They are glyphs in `CombiNumerals-Bold`, a circled-number
    font, and they extract as junk: item 1 is "q", item 13 is "d", item 20 is
    the pair "2)". The encoding is recoverable (q-o are 1-9, a-; are 10-19, and
    two-glyph pairs compose a leading digit with a shifted digit where ")" is 0,
    "!" is 1, ... "(" is 9), but it is not reliable: on a multi-part Civics item
    only the FIRST sub-question carries a number, so the glyphs undercount the
    questions and every label after the first multi-part item would shift.

So both are used as *typography* - which font a character is drawn in tells us
whether it is an option letter or body text, and that is a fact about the
document rather than a guess - and NEITHER is used for alignment. Alignment
hangs on the key table's `Page No.` column instead, which pins each item to the
printed page it appears on. That is a per-item anchor: a systematic off-by-one
cannot hide, because it would have to move every item onto the wrong page at
once, and the printed page number is read straight off the page footer.

The check is therefore: for every printed page, the number of option groups
found on it must equal the number the key predicts, where a selected-response
item predicts one group, a constructed response none, and a multi-part item
(its key answer written "C;A") one per part. A page that disagrees is dropped
whole, and a booklet where a quarter of the pages disagree is refused entirely.

Pages are dropped rather than the whole form because one disagreement is
usually honest rather than wrong: a constructed-response item whose Part A is
multiple choice prints a group the key gives no answer for, since DESE
publishes answers for selected-response items only. Confining the damage to the
page keeps that from costing the other nineteen items in the booklet, and an
off-by-one still cannot travel, because the next page re-anchors on its own
printed number.

2019-2023 booklets carry a second, independent key: an inline marker printed
above each item, `SC265247 OP C`, giving the item's internal ID and its answer
right next to the item itself. Every item that has one is compared against it
and a single disagreement fails the whole form - a booklet that is wrong about
one answer cannot be trusted about the rest. This is not theoretical: it throws
out the 2023 high school Biology form, whose item 11 is keyed D in the table
and B in the margin, and 664 items across 46 forms pass it. DESE stopped
printing the markers in 2024, which is why the table is the primary key rather
than the convenient one.

WHAT ACTUALLY SURVIVES, AND WHAT DOES NOT
-----------------------------------------
Three whole categories come out empty, and none of them is a parser fault:

  * ELA is dropped on sight. Every question on the MCAS reading test hangs off
    a passage printed beside it - that is what the test is - so there is no
    such thing as a standalone MCAS reading item. Left to the phrase filters it
    leaks the ones that name their passage rather than referring to "the
    passage" ("Based on The View from Saturday, ...").
  * Grade 8 Civics yields nothing. It is built entirely from document-based
    performance tasks: all 31 released questions across 2025 and 2026 are
    either a sub-part of a multi-part question or turn on a source, map,
    timeline or political cartoon printed with them. This is the subject the
    project most wanted and it is genuinely unusable in this format.
  * 2024's grade 3-8 files are eight pages of front matter and a table of item
    descriptions. DESE published the information that year but not the items.
    The high school files for 2024 are real booklets, so the year is not
    entirely lost.

The high school end-of-course tests (Biology, Chemistry, Introductory Physics,
Technology/Engineering) are included even though they are named by subject
rather than by grade, because they are science and science is what this state
was worth reading for. They are recorded as grade 10.

Line text is rebuilt from characters rather than taken from pdfplumber's
`extract_text`, which puts the space in the wrong place on MCAS option text
("t he legislative branch"). Within a word MCAS's glyph advances are exactly
flush and a real word space is about a third of the font size, so a threshold
at 0.15 em separates them cleanly.

WHY NOT THE OLDER, BETTER-TYPESET YEARS
---------------------------------------
The pre-2019 paper-era booklets would very likely parse better. They are gone
from doe.mass.edu (all of 2013-2018 404) and exist only as Wayback Machine
snapshots. Pulling a training corpus out of a third-party archive of PDFs the
publisher has withdrawn is a licensing question and not a parsing one, so the
prober records that they are missing and stops there. `--wayback` is
deliberately not a flag.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import paths

STATE_DIR = paths.DATA / "state_tests"
PDF_DIR = STATE_DIR / "pdf_ma"

INDEX = "https://www.doe.mass.edu/mcas"
YEARS = (2019, 2021, 2022, 2023, 2024, 2025, 2026)

# The link text on the release pages, mapped to the subject we record. Spanish
# translations of the same forms are linked alongside the English ones and are
# skipped: they are the same items, so keeping them would duplicate the corpus
# in a language the rest of the project does not use.
SUBJECTS = {
    "math": "math",
    "ela": "reading",
    "ste": "science",
    "civics": "civics",
    # the high school end-of-course tests, which are science by another name
    "hs-bio": "science",
    "hs-biology": "science",
    "hs-chem": "science",
    "hs-physics": "science",
    "hs-techeng": "science",
}


# --------------------------------------------------------------------------
# discovery and download
# --------------------------------------------------------------------------

def fetch(url: str, timeout: int = 120) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return None


def fetch_pdf(url: str) -> bytes | None:
    """Return the body only if it really is a PDF.

    doe.mass.edu serves its 404 as a 40KB HTML page, sometimes under a 200, so
    neither the status nor the length can be trusted. The magic bytes can.
    """
    body = fetch(url)
    return body if body and body[:5] == b"%PDF-" else None


LINK = re.compile(r'href="([^"]+\.pdf)"', re.IGNORECASE)
# gr8-math.pdf (2019-2022) and g8-math.pdf (2023+); gr10/g10 for high school
NAME = re.compile(r"^gr?(\d{1,2})-(math|ela|ste|civics)\.pdf$", re.IGNORECASE)
# The high school end-of-course science tests are named by their subject
# rather than by a grade. They are not tied to one grade - Biology is usually
# sat at the end of grade 9 or 10 - and are recorded as grade 10, the grade the
# rest of the high school programme is labelled with.
HS_NAME = re.compile(r"^hs-(bio|biology|chem|physics|techeng)\.pdf$", re.IGNORECASE)
HS_GRADE = 10


def discover(years) -> tuple[list[tuple], dict]:
    """Scrape each year's release page for its grade/subject PDFs.

    The filenames are read off the page rather than constructed, because the
    naming changed in 2023 and guessing would silently lose three years.
    """
    forms, report = [], {"pages": [], "missing_pages": [], "skipped": Counter()}
    for year in years:
        url = f"{INDEX}/{year}/release/"
        body = fetch(url)
        if not body or b"Release of" not in body:
            report["missing_pages"].append(url)
            continue
        html = body.decode("utf-8", "replace")
        names = []
        for href in LINK.findall(html):
            name = href.rsplit("/", 1)[-1]
            m, hs = NAME.match(name), HS_NAME.match(name)
            if m:
                names.append((int(m.group(1)), m.group(2).lower(), name))
            elif hs:
                names.append((HS_GRADE, "hs-" + hs.group(1).lower(), name))
            elif "spanish" in name.lower() or re.search(r"-(es|ht|pt|vi|zh)", name):
                report["skipped"]["translation"] += 1
            else:
                report["skipped"][name] += 1
        report["pages"].append((url, len(names)))
        for grade, subject, name in sorted(set(names)):
            forms.append((year, grade, subject, f"{url}{name}"))
    return forms, report


def download(forms, workers: int = 8) -> dict:
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    def one(form):
        year, grade, subject, url = form
        dest = PDF_DIR / f"{year}-g{grade}-{subject}.pdf"
        if dest.exists() and dest.stat().st_size > 1024:
            return form, True
        body = fetch_pdf(url)
        if body:
            dest.write_bytes(body)
        return form, bool(body)

    report = {"got": [], "failed": []}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for form, ok in ex.map(one, forms):
            report["got" if ok else "failed"].append(form)
    return report


# --------------------------------------------------------------------------
# layout: characters -> lines that know their own typography
# --------------------------------------------------------------------------

OPTION_FONT = "CircleFrame"      # a letter drawn inside a circle: an option
NUMBER_FONT = "Combi"            # a circled number: an item number
WORD_GAP = 0.15                  # of the font size; see the module docstring
SMALL_TYPE = 0.80                # smaller than this much of the line: a super/subscript
BAR_WIDTH = 60                   # a horizontal rule this short is a fraction bar
BAR_REACH = 14                   # how far above or below the bar its digits sit


def pua(text: str) -> str:
    """Map private-use codepoints back to the byte the font was asked for.

    The circled-number font is subset with no usable ToUnicode, so its glyphs
    arrive either as ASCII or as U+F0xx depending on the year. Folding the
    high range down makes the two spellings comparable.
    """
    return "".join(chr(ord(c) & 0xFF) if 0xE000 <= ord(c) <= 0xF8FF else c
                   for c in text)


def line_text(chars) -> str:
    """Rebuild a line from its characters.

    pdfplumber's own text is wrong on MCAS option text - it renders "the
    legislative branch" as "t he legislative branch" - so the spacing is
    recomputed from the glyph advances, which are unambiguous.
    """
    out = []
    for prev, ch in zip([None] + list(chars), chars):
        if prev is not None:
            gap = ch["x0"] - prev["x1"]
            if gap > WORD_GAP * max(ch["size"], 1) and not out[-1].isspace():
                out.append(" ")
        out.append(ch["text"])
    return pua("".join(out)).strip()


FOOTER_TOP = 0.90        # page numbers sit in the bottom tenth of the page
FOOTER_NUM = re.compile(r"(\d{1,4})\s*$")


def read_pages(pdf_path) -> tuple[list | None, str]:
    """Every line of the document, tagged with the printed page it sits on.

    Each PDF page carries its printed number twice: once for real and once
    mirrored at a negative x, an artifact of the booklet imposition. Only the
    positive copy is the page you are looking at, so every character at x < 0
    is dropped before anything else happens.

    The printed number is then not trusted page by page. Some footers do not
    extract at all and some pick up a stray item ID ("2200 21"), and a page
    whose number is read wrong would move its items somewhere else in the
    alignment. Since MCAS booklets are numbered continuously, the whole mapping
    is one integer - printed = index + offset - so the offset is taken as the
    consensus of the footers that did read, and any footer that disagrees with
    the consensus fails the booklet.
    """
    import pdfplumber

    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for index, page in enumerate(pdf.pages):
            lines, height, footers = [], page.height or 1, []
            # A fraction is drawn, not written: a short horizontal rule with a
            # numerator above it and a denominator below. Flattened to text it
            # becomes "She needs 2 cup of sugar for 3 each loaf", which reads
            # like a typo rather than like the missing information it is. The
            # rule itself is the only reliable trace, so it is collected here.
            bars = [r for r in (page.rects + page.lines)
                    if abs(r["y1"] - r["y0"]) < 3 and 4 < (r["x1"] - r["x0"]) < BAR_WIDTH]
            for raw in page.extract_text_lines():
                chars = [c for c in raw["chars"] if c["x0"] > 0]
                if not chars:
                    continue
                first = chars[0]
                letter = pua(first["text"])
                option = (letter if OPTION_FONT in first["fontname"]
                          and letter in "ABCDEFGH" else None)
                number = "".join(pua(c["text"]) for c in chars
                                 if NUMBER_FONT in c["fontname"] and c["text"].strip())
                # The circled option letter and the circled item number are
                # both typography rather than content, and both come out of the
                # text as a stray capital or a stray "e". Keep the fact, drop
                # the character.
                body = [c for c in (chars[1:] if option else chars)
                        if NUMBER_FONT not in c["fontname"]]
                sizes = [round(c["size"], 1) for c in body] or [1.0]
                biggest = max(sizes)
                line = {
                    "text": line_text(body),
                    "option": option,
                    "number": number or None,
                    "x0": round(first["x0"]),
                    "page": index,
                    # An exponent or a subscript is set smaller than the text
                    # around it and loses its meaning when flattened onto the
                    # line: "2 cubed" arrives as "23". Either mark makes the
                    # line unusable, and both are geometry rather than guesswork.
                    "stacked": any(s < SMALL_TYPE * biggest for s in sizes) or any(
                        raw["top"] - BAR_REACH < (b["top"] + b["bottom"]) / 2
                        < raw["bottom"] + BAR_REACH for b in bars),
                }
                if raw["top"] > FOOTER_TOP * height:
                    footers.append(line["text"])
                else:
                    lines.append(line)
            seen = [int(m.group(1)) for m in
                    (FOOTER_NUM.search(t) for t in footers) if m]
            pages.append({"index": index, "printed": None,
                          "seen": seen[-1] if seen else None, "lines": lines})

    offsets = Counter(p["seen"] - p["index"] for p in pages if p["seen"] is not None)
    if not offsets:
        return None, "no page numbers in any footer"
    offset, _ = offsets.most_common(1)[0]
    if sum(offsets.values()) < 0.5 * len(pages):
        return None, (f"only {sum(offsets.values())} of {len(pages)} pages have a "
                      f"readable page number")
    for page in pages:
        # The footer this page actually carries, where it read; the consensus
        # offset only fills the gaps. Front matter and the key tables at the
        # back are numbered separately from the body and would drag a
        # whole-document offset off by a page or two.
        page["printed"] = page["seen"] if page["seen"] is not None else page["index"] + offset

    # The running head repeats at the top of every page of a session
    # ("Civics State Performance Task"). It is not part of the question below
    # it, and left in place it lands in the stem of whichever item happens to
    # start that page.
    heads = Counter(p["lines"][0]["text"] for p in pages if p["lines"])
    running = {t for t, n in heads.items() if n >= 3 and t}
    for page in pages:
        page["lines"] = [l for l in page["lines"] if l["text"] not in running]
    numbered = [p["printed"] for p in pages if any(l["option"] for l in p["lines"])]
    if numbered != sorted(numbered):
        return None, "the pages holding options are not in page-number order"
    return pages, ""


# --------------------------------------------------------------------------
# the answer key
# --------------------------------------------------------------------------

# SR is selected response and is the only type that prints options; CR is
# constructed response, SA short answer (a number written into a grid) and ES
# the ELA essay.
ITEM_TYPES = {"SR", "CR", "SA", "ES"}

# What an answer looks like when it is an answer: a single option letter, or a
# comma-separated set of them for a multi-select. Everything else a key cell
# can hold - "N/A", a number an item was gridded into, or the run-together
# "BN/A" a two-part item prints - is not one, and must not be read as one.
MC_ANSWER = re.compile(r"[A-H](?:,[A-H])*")

ANSWER_COL = re.compile(r"correct\s*answer", re.IGNORECASE)
ITEM_COL = re.compile(r"item\s*no", re.IGNORECASE)
PAGE_COL = re.compile(r"page\s*no", re.IGNORECASE)
TYPE_COL = re.compile(r"item\s*type", re.IGNORECASE)


def parse_key(pdf_path) -> tuple[dict | None, str]:
    """Read the "Released Operational Items" table off the back of the booklet.

    The table is ruled, so pdfplumber recovers it cell-for-cell and the columns
    can be found by their headers instead of by counting them - the column
    order and count differ between subjects and years (STE has a "Practice
    Category", maths does not). The "Unreleased Operational Items" table that
    follows has no answer column at all, which is exactly how it is told apart.

    Returns {item_no: {"pages": [...], "type": "SR", "answer": "C",
    "parts": 1}}.
    """
    import pdfplumber

    key: dict[int, dict] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                header = next((r for r in table
                               if any(c and ANSWER_COL.search(c) for c in r)), None)
                if header is None:
                    continue
                col = {}
                for i, cell in enumerate(header):
                    if not cell:
                        continue
                    for name, pat in (("item", ITEM_COL), ("page", PAGE_COL),
                                      ("type", TYPE_COL), ("answer", ANSWER_COL)):
                        if pat.search(cell):
                            col[name] = i
                if len(col) < 4:
                    return None, "key table is missing one of its four columns"
                for row in table[table.index(header) + 1:]:
                    cells = [(c or "").strip() for c in row]
                    if max(col.values()) >= len(cells):
                        continue
                    if not re.fullmatch(r"\d{1,2}", cells[col["item"]]):
                        continue
                    item_no = int(cells[col["item"]])
                    pages = [int(p) for p in re.findall(r"\d{1,4}", cells[col["page"]])]
                    # "10-11" means the item runs across both pages
                    if len(pages) == 2 and 0 < pages[1] - pages[0] <= 3:
                        pages = list(range(pages[0], pages[1] + 1))
                    # One item can answer in several parts. The parts are
                    # separated by a semicolon, or by the line break the cell
                    # is laid out with, which survives as a newline.
                    answer = [re.sub(r"\s+", "", p).upper()
                              for p in re.split(r"[;\n]", cells[col["answer"]])
                              if p.strip()]
                    kind = cells[col["type"]].strip().upper()
                    if item_no in key:
                        return None, f"key lists item {item_no} twice"
                    key[item_no] = {"pages": pages, "type": kind, "answer": answer}
    if not key:
        return None, "no released-items table with a Correct Answer column"
    if sorted(key) != list(range(1, max(key) + 1)):
        return None, f"key items are not 1..{max(key)}"

    for item_no, item in key.items():
        if item["type"] not in ITEM_TYPES:
            # An unrecognised type means this booklet is laid out in some way
            # the parser has not been shown. Guessing what it is is exactly the
            # move that mislabels a form silently.
            return None, f"item {item_no} has unknown item type {item['type']!r}"
        item["answers"] = [p for p in item["answer"] if MC_ANSWER.fullmatch(p)]
        if item["type"] == "SR" and not item["answers"]:
            # A selected-response row with no letter answer means the answer
            # column did not line up, and every item after it would be
            # relabelled without anything looking wrong.
            return None, f"item {item_no} is SR but its answer {item['answer']} has no letters"
        if not item["pages"] and item["answers"]:
            return None, f"item {item_no} has no page number"
    return key, ""


# The inline marker 2019-2023 booklets print above each item: internal item ID,
# "OP" for operational, then the answer. "X" appears where an answer would be
# on a stimulus span or a constructed response, so it is not an answer.
MARKER = re.compile(r"^([A-Z]{2}[A-Za-z0-9_]{3,})\s+(OP|MC|EQ)\b\s*(.*)$")


def parse_markers(pages) -> list[str]:
    out = []
    for page in pages:
        for line in page["lines"]:
            m = MARKER.match(line["text"])
            if m:
                out.append(re.sub(r"\s+", "", m.group(3).upper()))
    return out


# --------------------------------------------------------------------------
# the booklet
# --------------------------------------------------------------------------

def option_groups(lines) -> list[dict]:
    """Every run of option letters starting at A, with its text and its stem.

    An option letter is known by its font, so this cannot mistake a stem line
    beginning with a capital for an option. The letters must run A, B, C, ...
    without a gap; a run that restarts at A begins the next question. The
    number of options is NOT fixed at four - grade 8 Civics prints three.
    """
    marks = [i for i, l in enumerate(lines) if l["option"]]
    groups, i = [], 0
    while i < len(marks):
        if lines[marks[i]]["option"] != "A":
            i += 1
            continue
        run = [marks[i]]
        while i + 1 < len(marks):
            nxt, cur = lines[marks[i + 1]]["option"], lines[marks[i]]["option"]
            if ord(nxt) != ord(cur) + 1:
                break
            i += 1
            run.append(marks[i])
        i += 1

        indent = lines[run[0]]["x0"]
        # The last option's text stops when the indent returns to the stem, and
        # in any case at the end of the page: an option list never runs over a
        # page break, so anything on the next page is the next question, a
        # session divider or the reference sheet, and letting the scan reach it
        # glues the whole back of the booklet onto option D.
        end, page = run[-1] + 1, lines[run[-1]]["page"]
        while end < len(lines) and lines[end]["x0"] > indent + 6 \
                and lines[end]["page"] == page \
                and not lines[end]["option"] and not lines[end]["number"]:
            end += 1
        bounds = list(run) + [end]
        texts = []
        for p in range(len(run)):
            head = lines[bounds[p]]["text"]
            tail = [lines[x]["text"] for x in range(bounds[p] + 1, bounds[p + 1])]
            texts.append(" ".join([head] + tail).strip())
        groups.append({"letters": [lines[x]["option"] for x in run],
                       "options": texts, "start": run[0], "end": end,
                       "page": lines[run[0]]["page"]})
    return groups


SESSION = re.compile(r"^(?:Grade\s+\d+|SESSION\s+\d+|This session contains)", re.I)

# MCAS announces a shared stimulus explicitly and always in the same voice.
# Everything from the announcement to the end of the session is treated as
# passage-bound, which removes essentially all of ELA - that is correct, not a
# parser bug.
ANNOUNCE = re.compile(
    r"answer\s+(?:the\s+)?questions?\s+\d"
    r"|questions?\s+that\s+follow"
    r"|the following (?:section|performance task|passage|article|selection)"
    r"|read and examine|read the (?:passage|article|selection|information|sources?)"
    r"|use (?:the|this|these) .{0,40}\bto answer"
    r"|based on the (?:passage|article|selection|source|text)"
    r"|refer to the",
    re.IGNORECASE,
)

STIMULUS_WORDS = 45      # this much prose ahead of a question is a passage
PROSE_LINE_WORDS = 7     # shorter lines than this are table cells, not prose
STIMULUS_REACH = 4       # questions an unnumbered announcement is assumed to cover

# "Then answer questions 2-4." - the announcement usually says exactly which
# questions it covers, which beats guessing at how far it reaches.
COVERS = re.compile(r"questions?\s+(\d{1,2})\s*(?:[\u2013\u2014-]|through|and)\s*(\d{1,2})",
                    re.IGNORECASE)


def prose_words(texts) -> int:
    return sum(len(w) for w in (t.split() for t in texts) if len(w) >= PROSE_LINE_WORDS)


def shared_stimulus(lines, groups) -> tuple[set, set]:
    """Which questions are bound to a passage, source or scenario they share.

    MCAS says so out loud and in a house style: "Read and examine the sources.
    Then answer questions 2-4." Where the announcement names the questions it
    covers, those exact item numbers are taken. Where it does not, it is
    assumed to reach over the next few questions and no further - an earlier
    version let an announcement run to the end of the session, which condemned
    most of a science booklet on the strength of one passage on its first page.
    """
    starts = [g["start"] for g in groups]
    by_group, by_item = set(), set()
    for i, line in enumerate(lines):
        if not ANNOUNCE.search(line["text"]):
            continue
        covers = COVERS.search(line["text"])
        if covers:
            lo, hi = int(covers.group(1)), int(covers.group(2))
            if lo <= hi <= lo + 12:
                by_item.update(range(lo, hi + 1))
                continue
        first = next((n for n, s in enumerate(starts) if s > i), len(groups))
        by_group.update(range(first, min(first + STIMULUS_REACH, len(groups))))
    return by_group, by_item


AMBIGUOUS_PAGES = 0.25   # more than this fraction unexplained and the form goes


def parse_form(pages, key) -> tuple[list[dict] | None, str, int]:
    """Split the booklet into items, or refuse the booklet.

    Alignment is done one printed page at a time. The key says which items sit
    on each page and how many option groups each of them should print; the page
    is trusted only when exactly that many groups are found on it, in which
    case the two lists are matched in order - both are in item order, because
    that is the order a booklet prints in.

    Working per page rather than over the whole document is what makes an
    off-by-one survivable. The page number is read off the page footer and the
    key's page column is written by the publisher, so the two are independent;
    a group can only ever be matched to an item the key already places on that
    same page, and a miscount is confined to the page it happens on instead of
    relabelling every item that follows.

    Pages that do not agree are dropped whole. The commonest reason is honest:
    a constructed-response item whose Part A is multiple choice prints a group
    the key gives no answer for, because DESE publishes answers for
    selected-response items only. That is unresolvable rather than wrong, so
    those items are counted separately as `unkeyed_page`. A booklet where this
    happens on more than a quarter of its pages is not being read correctly at
    all, and is refused entirely.
    """
    lines = [l for page in pages for l in page["lines"]]
    printed = {page["index"]: page["printed"] for page in pages}
    groups = option_groups(lines)
    if not groups:
        return None, "no option groups anywhere in the booklet", 0

    wanted = defaultdict(list)
    for item_no in sorted(key):
        item = key[item_no]
        for part, answer in enumerate(item["answers"]):
            # a multi-page item prints its options on one of its pages; the
            # page bucket it belongs to is settled below, by elimination
            wanted[item["pages"][0]].append((item_no, part, answer))
    found = defaultdict(list)
    for group in groups:
        found[printed[group["page"]]].append(group)

    # An item keyed to "10-11" prints its options on whichever of the two the
    # layout put them on. Move a page's surplus onto the neighbour the key
    # allows, so that a spread is not read as two disagreeing pages.
    for item_no in sorted(key):
        item = key[item_no]
        if len(item["pages"]) < 2:
            continue
        head = item["pages"][0]
        for page in item["pages"][1:]:
            while (len(found[head]) < len(wanted[head]) and found[page]
                   and len(found[page]) > len(wanted[page])):
                found[head].append(found[page].pop(0))

    index = {id(g): n for n, g in enumerate(groups)}
    shared_groups, shared_items = shared_stimulus(lines, groups)
    items, unkeyed, bad_pages = [], 0, 0
    for page in sorted(set(wanted) | set(found)):
        if len(found[page]) != len(wanted[page]):
            bad_pages += 1
            unkeyed += len(wanted[page])
            continue
        for group, (item_no, part, answer) in zip(found[page], wanted[page]):
            n = index[id(group)]
            start = groups[n - 1]["end"] if n else 0
            # The circled item number is where the question starts. It is not
            # trustworthy enough to align on - a multi-part item numbers only
            # its first sub-question - but where one is present it is the exact
            # boundary, and it keeps the previous item's constructed-response
            # prompt out of this item's stem. Where there is none, fall back to
            # the end of the previous question's options.
            numbered = [i for i in range(start, group["start"]) if lines[i]["number"]]
            span = [lines[i] for i in range(numbered[-1] if numbered else start,
                                            group["start"])]
            stem_lines = [l["text"] for l in span
                          if l["text"] and not MARKER.match(l["text"])
                          and not SESSION.match(l["text"])]
            # Where the item number is set in ordinary type rather than in the
            # circled-number font, it opens the stem as a bare digit.
            if stem_lines:
                stem_lines[0] = re.sub(rf"^{item_no}\b\s*", "", stem_lines[0])
                stem_lines = [s for s in stem_lines if s]
            # the inline marker is printed ABOVE the item number, so it is
            # looked for over the whole gap rather than the trimmed stem
            marked = [MARKER.match(lines[i]["text"]) for i in range(start, group["start"])]
            marked = [re.sub(r"\s+", "", m.group(3).upper()) for m in marked if m]
            items.append({
                "item_no": item_no,
                "part": part,
                "parts": len(key[item_no]["answers"]),
                "stem_lines": stem_lines,
                "letters": group["letters"],
                "options": group["options"],
                "answer": answer,
                "marker": marked[-1] if marked else None,
                "stacked": any(l["stacked"] for l in span)
                           or any(lines[i]["stacked"]
                                  for i in range(group["start"], group["end"])),
                "under_stimulus": n in shared_groups or item_no in shared_items
                                  or prose_words(stem_lines) >= STIMULUS_WORDS,
            })
    pages_used = len(set(wanted) | set(found))
    if bad_pages > AMBIGUOUS_PAGES * max(pages_used, 1):
        return None, (f"{bad_pages} of {pages_used} pages hold a different number of "
                      f"option groups than the key expects"), 0
    items.sort(key=lambda it: (it["item_no"], it["part"]))
    return items, "", unkeyed


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------

VISUAL = re.compile(
    r"\b(?:charts?|diagrams?|figures?|graphs?|maps?|pictures?|images?|photos?"
    r"|photographs?|illustrations?|drawings?|models?|tables?|grids?|number lines?"
    r"|timelines?|cartoons?|posters?|sketch(?:es)?|schematics?|shown"
    r"|show[ns]?\s+(?:above|below|here)"
    # "Which of the following boxes has the greatest volume?" - the boxes are
    # drawn, and what survives of them is a row of dimension labels
    r"|which of the (?:following|these)\s+(?:\w+\s+)?(?:boxes|prisms|cubes|solids"
    r"|shapes|containers|arrays|nets|objects|angles|triangles|rectangles|polygons)"
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

FIGURE_GLYPHS = re.compile(r"[\u2190-\u21FF\u2500-\u27BF\u2B00-\u2BFF\uFFFD\u25A0-\u25FF]")
BAD_ENCODING = re.compile(r"\(cid:|\uFFFD")
NO_VOWEL = re.compile(r"\b(?![A-Z]{2,}\b)[A-Za-z]{4,}\b")

# Fractions, exponents and subscripts are two-dimensional; flattening them into
# a line does not fail loudly, it produces a plausible string that means
# something else. MCAS stacks mixed numbers, so "x = 5 1/2" arrives as the
# three lines "A 1", "x = 5", "2".
FLATTENED_MATH = re.compile(
    r"\)\s*\d"
    r"|\b(?:cm|mm|km|in|ft|yd|m|s)\.?\d\b"
    r"|(?:\b\d\b ){3,}"
    r"|[\u00d7\u00f7+\u2212]\s*$"
    r"|\u22c5"
    r"|\S {2,}\S"
    r"|^\d+\s*$"
    # A fraction bar set as a dash rather than drawn as a rule: "3 - feet"
    # is three and a half feet, and the halves have landed either side of it.
    r"|(?:^|\s)[\u2013\u2014](?=\s)"
    # an option opening with two numbers running together: "1 35 inches ..."
    r"|^\d+\s+\d"
)

# A placeholder box the student writes into. It extracts as a stray "?", and
# takes the noun of the sentence with it: "What number belongs in the to make
# the student's subtraction problem true?"
LOST_BOX = re.compile(r"\?\s+\S")

SHARED = re.compile(
    r"\b(?:the\s+)?(?:passage|selection|selections|article|articles|poem|poems|story|stories"
    r"|excerpt|essay|interview|memoir|play|speech|text\s+box|source|sources"
    r"|paragraph|paragraphs|stanza|stanzas|line\s+\d+|lines\s+\d+"
    r"|the\s+author|the\s+narrator|the\s+speaker|the\s+poet|the\s+writer"
    r"|both\s+selections|these\s+selections)\b",
    re.IGNORECASE,
)

# A stem is a question or a sentence-completion prompt. Anything else is a
# fragment torn out of a figure.
STEM_END = re.compile(r"[?\u2014\u2013:.-]\s*$")

# The sub-prompt label of a multi-part question: "B. Identify the type of ..."
PART_LABEL = re.compile(r"^(?:[A-D]\.\s+[A-Z]|Part\s+[A-D]\b)")

# Which option letters a subject prints. Maths, science and ELA always print
# four, so three means one was lost and the item is no longer the one that was
# sat. Grade 8 Civics genuinely mixes three-option and four-option questions,
# which is the reason nothing here assumes four.
OPTION_SETS = {"math": ("ABCD",), "science": ("ABCD",),
               "reading": ("ABCD",), "civics": ("ABC", "ABCD")}

DROP_REASONS = ("passage_bound", "not_single_select", "multi_part", "parse",
                "shared_stimulus", "image", "debris", "degenerate")


def classify(item: dict, subject: str) -> str:
    """Return "" to keep, otherwise the reason to drop."""
    if subject == "reading":
        # Every question on the MCAS ELA test hangs off a passage printed
        # beside it - that is what the test is. There is no such thing as a
        # standalone MCAS reading item, so the subject is dropped as a whole
        # rather than left to the phrase filters, which let through the ones
        # that name their passage instead of referring to "the passage"
        # ("Based on The View from Saturday, ...").
        return "passage_bound"
    answer, letters = item["answer"], item["letters"]
    stem_head = item["stem_lines"][0] if item["stem_lines"] else ""
    # A two-part question prints "A." and "B." sub-prompts. Part A carries the
    # scenario and stands alone; everything after it is written to be read
    # against what came before ("B. Identify the type of force ...") and cannot
    # be lifted out of the item.
    if item["part"] > 0 or PART_LABEL.match(stem_head):
        return "multi_part"
    if len(answer) != 1 or answer not in letters:
        # multi-select ("A,C"), or an answer naming an option that is not here
        return "not_single_select"

    stem_lines = [s for s in item["stem_lines"] if s]
    stem = " ".join(stem_lines).strip()
    options = [o.strip() for o in item["options"]]
    blob = stem + " \n " + " \n ".join(options)

    # Exactly the letters the subject prints, with none missing. A four-option
    # item that arrives with three has lost one, and although the answer may
    # still be right the item is no longer the one that was sat.
    if "".join(letters) not in OPTION_SETS[subject] or not all(options):
        return "parse"
    if not stem or not STEM_END.search(stem) or len(stem.split()) < 5:
        return "parse"
    if LOST_BOX.search(stem):
        return "debris"

    if item["under_stimulus"] or SHARED.search(blob):
        return "shared_stimulus"
    if VISUAL.search(blob) or FIGURE_GLYPHS.search(blob):
        return "image"

    if item["stacked"]:
        return "debris"
    if BAD_ENCODING.search(blob) or any(FLATTENED_MATH.search(o) for o in options):
        return "debris"
    if FLATTENED_MATH.search(stem) or any(
            not re.search(r"[aeiouyAEIOUY]", w) for w in NO_VOWEL.findall(blob)):
        return "debris"

    if len(set(o.lower() for o in options)) != len(options):
        return "degenerate"
    if any(len(re.sub(r"\W", "", o)) <= 2 for o in options):
        return "degenerate"
    return ""


TIDY = (("\u201a", ","), ("\u2212\u2212", "\u2212"), ("\u2013\u2013", "\u2013"))


def tidy(text: str) -> str:
    for bad, good in TIDY:
        text = text.replace(bad, good)
    # "A piece of wire has a length of 110 inches. Part A Which of the
    # following ...": the label numbers the sub-prompt within the booklet and
    # means nothing once the sub-prompt is standing on its own.
    text = re.sub(r"\bPart\s+[A-D]\b\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def extract_form(path, year: int, grade: int, subject: str, stats: Counter) -> list[dict]:
    label = path.stem
    pages, why = read_pages(path)
    if pages is None:
        stats["form_unreadable"] += 1
        print(f"  {label}: DROPPED whole form, {why}")
        return []
    if not any(l["option"] for page in pages for l in page["lines"]):
        # 2024 is the whole of this case: DESE published the item-description
        # tables that year but not the items, so the file is eight pages of
        # front matter and a table. Nothing to parse and nothing wrong.
        stats["form_no_items"] += 1
        print(f"  {label}: DROPPED whole form, the document contains no test items")
        return []

    key, why = parse_key(path)
    if key is None:
        stats["form_no_key"] += 1
        print(f"  {label}: DROPPED whole form, {why}")
        return []

    items, why, unkeyed = parse_form(pages, key)
    if items is None:
        stats["form_misaligned"] += 1
        stats["dropped_by_misaligned_form"] += len(key)
        print(f"  {label}: DROPPED whole form, {why}")
        return []

    stats["unkeyed_page"] += unkeyed

    # 2019-2023 booklets print an inline marker above each item carrying that
    # item's own answer. It is written by the publisher into a different part
    # of the document than the key table, so where it exists it is a second
    # opinion on every label, item by item, and one disagreement condemns the
    # booklet - a form that is wrong about even one answer cannot be trusted
    # about the rest.
    checked = [it for it in items if it["marker"] and re.fullmatch(r"[A-H]", it["marker"])]
    wrong = [it for it in checked if it["marker"] != it["answer"]]
    if wrong:
        stats["form_key_conflict"] += 1
        stats["dropped_by_misaligned_form"] += len(key)
        first = wrong[0]
        print(f"  {label}: DROPPED whole form, {len(wrong)} of {len(checked)} items "
              f"disagree with their inline marker (item {first['item_no']}: key says "
              f"{first['answer']}, the booklet says {first['marker']})")
        return []
    if checked:
        stats["forms_cross_checked"] += 1
        stats["items_cross_checked"] += len(checked)

    stats["raw"] += len(items)
    kept, reasons = [], Counter()
    for item in items:
        reason = classify(item, subject)
        if reason:
            reasons[reason] += 1
            stats[reason] += 1
            continue
        kept.append({
            "question": tidy(" ".join(item["stem_lines"])),
            "choices": [tidy(o) for o in item["options"]],
            "gold_idx": item["letters"].index(item["answer"]),
            "hint": None,
            "source": "ma",
            "subject": subject,
            "grade": grade,
            "year": year,
            "item_no": item["item_no"],
        })
    stats["kept"] += len(kept)
    detail = " ".join(f"{r}={reasons[r]}" for r in DROP_REASONS if reasons[r])
    print(f"  {label}: {len(items)} questions, {detail or 'no drops'} -> kept {len(kept)}")
    return kept


def gold_stats(items) -> tuple[float, float]:
    lengths = [len(it["choices"][it["gold_idx"]].split()) for it in items]
    if not lengths:
        return 0.0, 0.0
    return statistics.median(lengths), 100 * sum(1 for n in lengths if n == 1) / len(lengths)


def main():
    ap = argparse.ArgumentParser(
        description="Extract self-contained MC items from released MCAS PDFs. "
                    "OUTPUT IS DESE-COPYRIGHTED AND GITIGNORED - do not commit it.")
    ap.add_argument("--download", action="store_true", help="scrape and fetch the PDFs first")
    ap.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    ap.add_argument("--out", default=str(STATE_DIR / "ma_items.jsonl"))
    ap.add_argument("--show", type=int, default=0, help="print N kept items in full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if args.download:
        print("scraping the release index ...")
        forms, report = discover(args.years)
        for url, n in report["pages"]:
            print(f"  {url}  {n} English forms")
        for url in report["missing_pages"]:
            print(f"  {url}  NO SUCH PAGE")
        print(f"  skipped links: {dict(report['skipped'])}")
        got = download(forms)
        print(f"  downloaded {len(got['got'])}, failed {len(got['failed'])}")
        for form in got["failed"]:
            print(f"    failed: {form[3]}")

    stats = Counter()
    items = []
    print("\nextracting ...")
    for year in sorted(args.years):
        for path in sorted(PDF_DIR.glob(f"{year}-g*.pdf")):
            m = re.match(r"(\d{4})-g(\d{1,2})-([\w-]+)$", path.stem)
            if not m:
                continue
            grade, subject = int(m.group(2)), SUBJECTS.get(m.group(3))
            if subject is None:
                continue
            items += extract_form(path, year, grade, subject, stats)

    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    print(f"\nwrote {len(items)} items -> {args.out}   (COPYRIGHTED, GITIGNORED)")
    print(f"\n{'questions in the parsed forms':<36} {stats['raw']}")
    for reason in DROP_REASONS:
        if stats[reason]:
            print(f"{'  dropped: ' + reason:<36} {stats[reason]}")
    for label, name in (("forms dropped, no items published", "form_no_items"),
                        ("forms dropped, unreadable pages", "form_unreadable"),
                        ("forms dropped, no key", "form_no_key"),
                        ("forms dropped, misaligned", "form_misaligned"),
                        ("forms dropped, key conflict", "form_key_conflict"),
                        ("questions on unresolvable pages", "unkeyed_page"),
                        ("forms cross-checked vs markers", "forms_cross_checked"),
                        ("items cross-checked vs markers", "items_cross_checked")):
        if stats[name]:
            print(f"{'  ' + label:<36} {stats[name]}")
    print(f"{'kept':<36} {stats['kept']}")

    by_subject = Counter(it["subject"] for it in items)
    print("\nkept per subject")
    for subject, n in by_subject.most_common():
        median, single = gold_stats([it for it in items if it["subject"] == subject])
        print(f"  {subject:<10} {n:4d}   gold median {median:.0f}w, {single:.0f}% single-word")
    by_grade = Counter(it["grade"] for it in items)
    print("kept per grade: " + "  ".join(f"g{g}={by_grade[g]}" for g in sorted(by_grade)))
    by_year = Counter(it["year"] for it in items)
    print("kept per year:  " + "  ".join(f"{y}={by_year[y]}" for y in sorted(by_year)))
    median, single = gold_stats(items)
    print(f"\ngold answers: median {median:.0f} words, {single:.0f}% single-word")

    if args.show:
        import random

        sample = list(items)
        random.Random(args.seed).shuffle(sample)
        seen, picked = Counter(), []
        cap = max(1, args.show // max(1, len(by_subject))) + 2
        for it in sample:
            if seen[it["subject"]] < cap:
                seen[it["subject"]] += 1
                picked.append(it)
            if len(picked) == args.show:
                break
        for n, it in enumerate(picked, 1):
            print(f"\n--- {n}. {it['subject']} grade {it['grade']} {it['year']} "
                  f"item {it['item_no']} ---")
            print(it["question"])
            for i, choice in enumerate(it["choices"]):
                print(f"  {'ABCDEF'[i]}{' <-- GOLD' if i == it['gold_idx'] else '       '} {choice}")


if __name__ == "__main__":
    main()
