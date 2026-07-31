"""Turn California's released CST test questions into self-contained multiple-choice items.

    !!! THE EXTRACTED CONTENT IS NOT REDISTRIBUTABLE. DO NOT COMMIT IT. !!!

Every RTQ booklet carries:

    Copyright (c) 2009 California Department of Education.

This repository is PUBLIC, so the PDFs and everything derived from them are
written under `data/state_tests/`, which is in `.gitignore`. This file is the
only thing that gets committed. Rebuild the data instead of committing it:

    python src/extract_ca.py --download

WHERE THE SOURCE LIVES, AND WHY IT IS AWKWARD
---------------------------------------------
The legacy California Standards Test (CST) ran 2003-2013 under the STAR
program and released a slice of each year's items as "Released Test Questions"
(RTQ) booklets - one per grade per subject, each ending in a table that gives
the correct answer, the content standard and the year of release for every
question. CAASPP replaced STAR in 2014 and the CDE has since taken the RTQ
PDFs down; what is left at `cde.ca.gov/ta/tg/sr/documents/` is a Radware
captcha that refuses scripted requests. Fighting it is not worth it.

The Wayback Machine has them. Enumerating is a CDX query:

    http://web.archive.org/cdx/search/cdx?url=cde.ca.gov/ta/tg/sr/documents/*
        &output=json&filter=urlkey:.*rtq.*

which returns ~190 rows, of which 101 are distinct URLs that archived as a
real `application/pdf` with a 200. Fetching needs a REAL snapshot timestamp
from that query - the `2id_` shorthand hands back an HTML page, not the PDF:

    https://web.archive.org/web/<timestamp>id_/<original-url>

The 101 files are four generations of the same programme, and they nest rather
than complement each other: `cst04rtq*` (2004, ~16 items/booklet), `css05rtq*`
(2005, ~32), `rtq*` (2006-2008, ~48) and `cstrtq*` (2009-2013, ~96-114). A
later booklet re-releases everything its predecessor had, so all four are
parsed and the results deduplicated on the question text; the count that
matters is the union, not the sum.

There is NO grade 8 mathematics booklet. California tested Algebra I at grade
8, so `rtqalg1.pdf` is where those students' items are. That is the programme,
not a hole in the enumeration.

WHAT IS PARSED AND WHAT IS NOT
------------------------------
Only booklets whose grade is printed on them: grades 2-11 mathematics, ELA,
science and history-social science. The end-of-course booklets (Algebra I and
II, Geometry, Biology, Chemistry, Physics, Earth Science) are skipped, because
they carry no grade and the schema has a `grade` field that would have to be
invented. History-social science is the reason this source is here at all: the
grade 8, 10 and 11 booklets are ~340 items of it, and the rest of the project
has essentially none.

HOW THE PAGES ARE READ
----------------------
`pypdf` drops intra-word spaces on these files exactly as it does on STAAR, so
this uses `pdfplumber`, lazily imported and not in `requirements.txt`:

    python -m venv /tmp/pdfenv_ca && /tmp/pdfenv_ca/bin/pip install pdfplumber

The booklets are TWO-COLUMN, which `page.extract_text()` does not know: it
reads across the gutter and interleaves the columns, so item 1's stem arrives
spliced into item 3's. Every page is therefore split at its gutter and the
columns read one after the other. The gutter is found by counting how many
printed LINES straddle each candidate x: on a single-column page nearly every
line does and on a two-column page almost none do. Counting words instead
fails, because the history booklets lay a full-width map over two columns of
text and its labels land wherever the cartographer put them. Pages with no
gutter - the standards lists, and the answer-key table - are read whole, and
the test insists both halves look like FULL columns so that the key table,
which also has whitespace down its middle, is never torn in half.

Header and footer are cut by position rather than by pattern. They span the
full width, so they would defeat the gutter test if they survived to it.

Structure the parser leans on:

  * Items are numbered at line start, sometimes with the stem on the same
    line, sometimes with the number alone. Options are always A B C D on this
    programme - but that is read off each key, never assumed, because the
    STAAR work showed a booklet can letter its even items F G H J and nothing
    in the text says so.
  * Every item ends with an internal tracking code (`3N012302`, `CSH10255`).
    It is dropped, otherwise it lands inside option D.
  * Standards-list pages are interleaved between the item pages, not gathered
    into front matter, so there is no single point at which the body begins.
    They are left in and the item-number search steps over them; they contain
    no `1 Something` lines to be confused by.
  * Fractions, exponents and the (cid:N) glyphs the booklets use for operators
    flatten into text that reads plausibly and means something else. Those
    items are detected and dropped rather than repaired.

HOW THE LABELS ARE KEPT HONEST
------------------------------
The same rule as STAAR, for the same reason: a misaligned key mislabels a
whole booklet silently, which is worse than extracting nothing. The key is
read first and is the authority on how many items there are and how they are
lettered; the parser only has to agree with it. Any disagreement - an
unparseable key, a letter scheme that is neither A-D nor A-D/F-J by parity, an
item whose answer is not among the letters its options actually use - drops
the entire booklet.

The CST key table is a five-column affair at the end of the booklet:

    Question Number   Correct Answer   Standard   Skills   Year of Test
    1                 D                US11.1.3   HI 1     2004

so the answer is the SECOND column, not the last one as on STAAR, and the last
column is a per-item release year that is worth keeping. Some booklets render
it with spaces sprayed through every cell ("Cor rectAn swe r", "20 0 4",
"WH1 0.7.1", and item 41 as "4 1"), which is why the row parser despaces
before it reads anything.

Then there is a check STAAR could not run. Because the four generations
re-release each other's items, roughly half the surviving items are extracted
twice or more, from booklets typeset years apart and parsed independently.
Those copies have to agree on which option is correct; where they do not, one
parse is misaligned and there is no telling which, so both are dropped. It is
not a formality - it catches a real off-by-one in the 2006 grade 8 history
booklet - and the agreement rate it prints, 99.5%, is the best evidence
available that the labels are right.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

import paths
import staar_extract as staar

CA_DIR = paths.DATA / "state_tests"
PDF_DIR = CA_DIR / "pdf_ca"
OUT = CA_DIR / "ca_items.jsonl"

CDX = ("http://web.archive.org/cdx/search/cdx?url=cde.ca.gov/ta/tg/sr/documents/*"
       "&output=json&filter=urlkey:.*rtq.*&collapse=urlkey"
       "&fl=urlkey,timestamp,original,mimetype,statuscode,length")


# --------------------------------------------------------------------------
# which booklets, and what grade/subject they are
# --------------------------------------------------------------------------

# Read off the cover of each booklet. The grade is the grade the test was
# ADMINISTERED at, which is not always the grade of the standards it assesses:
# the grade 8 history-social science test draws on the grade 6, 7 and 8
# standards (WH6.x, WH7.x, US8.x) because that is how California organised the
# subject, and the booklet still says GRADE 8 on its cover.
NAMED = [
    (r"^cst04rtqela(?:gr)?(\d+)$", "reading"),
    (r"^cst04rtqmath(?:gr)?(\d+)$", "math"),
    (r"^cst04rtqhss(?:gr)?(\d+)$", "social_studies"),
    (r"^css05rtqgr(\d+)ela$", "reading"),
    (r"^css05rtqgr(\d+)math$", "math"),
    (r"^css05rtqgr(\d+)sci$", "science"),
    (r"^css05rtqhistgr(\d+)$", "social_studies"),
    (r"^css05rtqgr(\d+)hist$", "social_studies"),
    (r"^rtqgr(\d+)(?:ela|english|grammar)$", "reading"),
    (r"^rtqgr(\d+)math$", "math"),
    (r"^rtqgr(\d+)science$", "science"),
    (r"^rtqgr(\d+)history$", "social_studies"),
    (r"^cstrtqela(?:gr)?(\d+)(?:nw)?$", "reading"),
    (r"^cstrtqmath(\d+)$", "math"),
    (r"^cstrtqscience(\d+)$", "science"),
    (r"^cstrtqscigr(\d+)$", "science"),
    (r"^cstrtqhss(\d+)$", "social_studies"),
]

# Booklets whose filename does not carry its grade. The grade comes from the
# standards the key cites, which `check_grade` re-verifies against the PDF.
ODD = {
    "css05rtqgr68hist": ("social_studies", 8),    # grade 8 test, grade 6-8 standards
    "cstrtqgr5elajul2012": ("reading", 5),
    "cstrtqhssworld": ("social_studies", 10),     # World History, cites WH10.x
    "rtqgrworldhist": ("social_studies", 10),     # ditto
    "cstrtqhssmar18": ("social_studies", 11),     # US History, cites US11.x
}

# End-of-course booklets: no grade on the cover and none implied by the
# standards they cite, so there is nothing honest to put in the `grade` field.
EOC = re.compile(
    r"alg(?:ebra)?\d?|geom|bio|chem|physic|earthsci|earthscience|sciearth|13earthsci")

# Later generations re-release the earlier ones' items, so parse the biggest
# booklets first and let the deduplicator keep their (better typeset) copy.
GENERATION = {"cstrtq": 0, "rtq": 1, "css05": 2, "cst04": 3}


def classify_file(stem: str) -> tuple[str, int] | None:
    if stem in ODD:
        return ODD[stem]
    for pattern, subject in NAMED:
        m = re.match(pattern, stem)
        if m:
            return subject, int(m.group(1))
    return None


def generation(stem: str) -> int:
    for prefix, rank in GENERATION.items():
        if stem.startswith(prefix):
            return rank
    return 9


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------

def wayback_index() -> list[tuple[str, str]]:
    """(timestamp, original url) for every RTQ PDF that archived successfully."""
    req = urllib.request.Request(CDX, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        rows = json.load(r)[1:]
    return [(r[1], r[2]) for r in rows
            if r[3] == "application/pdf" and r[4] == "200"]


def fetch(url: str, tries: int = 5) -> bytes | None:
    """Wayback throttles hard. Back off on 429/503 rather than giving up."""
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read()
            return body if body[:5] == b"%PDF-" else None
        except urllib.error.HTTPError as e:
            if e.code not in (429, 502, 503, 504):
                return None
            time.sleep(8 * (attempt + 1))
        except (urllib.error.URLError, OSError):
            time.sleep(5 * (attempt + 1))
    return None


def download_all(pause: float = 1.2) -> dict:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    index = wayback_index()
    report = {"enumerated": len(index), "fetched": 0, "cached": 0, "failed": []}
    for timestamp, original in sorted(index, key=lambda r: r[1]):
        name = original.rsplit("/", 1)[-1]
        dest = PDF_DIR / name
        if dest.exists() and dest.stat().st_size > 4096:
            report["cached"] += 1
            continue
        body = fetch(f"https://web.archive.org/web/{timestamp}id_/{original}")
        if body is None:
            report["failed"].append(name)
        else:
            dest.write_bytes(body)
            report["fetched"] += 1
        time.sleep(pause)          # be a good citizen; the archive is a charity
    return report


# --------------------------------------------------------------------------
# text extraction
# --------------------------------------------------------------------------

HEAD_FRACTION = 0.085     # the running header sits above this
FOOT_FRACTION = 0.905     # the page number and copyright line sit below it
LINE_TOLERANCE = 3.5      # points of vertical slop within one printed line
GUTTER_CLEARANCE = 6      # points either side of the split that must be empty

BOILERPLATE = re.compile(
    r"^\s*(?:"
    r"[\u2014\u2013-]\s*\d+\s*[\u2014\u2013-]"          # the page number
    r"|This is a sample of California.*|based on performance.*"
    r"|CALIFORNIA\s*ST.*|.*STANDARDS\s*TEST.*|Released Te?\s?st Questions.*"
    r"|G\s*R\s*A\s*D\s*E|History[\u2013-]Social Science|Math|Science|English.*"
    r"|Copyright.*"
    r"|[A-Z]{2,4}\d{4,8}"                               # item tracking code
    r"|[\u25a0\u25aa\ufffd\u2022\s]+"                   # the item bullet
    r"|\(cid:\d+\)(?:\s*\(cid:\d+\))*"
    r")\s*$",
    re.IGNORECASE,
)

# The tracking code that closes every item. Case-sensitive and digit-heavy so
# it cannot eat an option that happens to be a short capitalised word.
TRACKING = re.compile(r"^[A-Z0-9]{6,12}$")


def _is_noise(line: str) -> bool:
    if BOILERPLATE.match(line):
        return True
    return bool(TRACKING.match(line) and sum(c.isdigit() for c in line) >= 4)


KERN = 0.4        # points; below this two "words" are one word split by a glyph


def _join(line: list[dict]) -> str:
    """Re-space one line from the geometry rather than from the word split.

    The booklets set fi/fl/ffi as single ligature glyphs, which pdfplumber
    emits as separate words butted up against their neighbour at a gap of
    exactly zero - so "Nullification" arrives as "Nullifi" + "cation" and
    joining on spaces gives "Nullifi cation". A real space is worth ~2.5pt at
    this body size, so the gap tells the two cases apart.
    """
    line = sorted(line, key=lambda w: w["x0"])
    out = [line[0]["text"]]
    for previous, w in zip(line, line[1:]):
        out.append("" if w["x0"] - previous["x1"] < KERN else " ")
        out.append(w["text"])
    return "".join(out)


def rows_of(words: list[dict]) -> list[list[dict]]:
    """Words -> printed lines, top to bottom, left to right within a line."""
    out, current, top = [], [], None
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if top is None or w["top"] - top <= LINE_TOLERANCE:
            top = w["top"] if top is None else top
            current.append(w)
        else:
            out.append(current)
            current, top = [w], w["top"]
    if current:
        out.append(current)
    return out


def gutter(rows: list[list[dict]], width: float) -> float | None:
    """The x of the two-column gutter, or None if the page is not two-column.

    Counted in LINES that straddle the candidate x, not in words. A clean
    empty band is too much to ask for: history booklets print a full-width map
    over two columns of text, and its labels sit wherever the cartographer put
    them. What actually separates the two cases is that on a single-column
    page nearly every line crosses the middle and on a two-column page almost
    none do, and a handful of stray map labels do not move that.

    Both halves also have to look like FULL columns - starting near their own
    margin and running most of the way to the gutter - which is what keeps
    narrow centred tables from being torn down the middle.
    """
    tally = {}
    for split in range(int(0.42 * width), int(0.59 * width)):
        tally[split] = sum(1 for r in rows if any(
            w["x0"] < split - GUTTER_CLEARANCE and w["x1"] > split + GUTTER_CLEARANCE
            for w in r))
    crossing = min(tally.values())
    if crossing > 0.25 * len(rows):
        return None
    # Among the near-quietest, take the x nearest the page centre, where these
    # templates put the gutter. Position matters more than the last two
    # crossings: drifting left strands the tail of a left-column line
    # ("...attempting to") at the head of the right column, and drifting right
    # swallows the right column's item NUMBER into the left column's line,
    # which loses the item its opener and with it the whole booklet. Splitting
    # an over-long option line in half only costs that one item.
    tolerance = crossing + max(2, int(0.05 * len(rows)))
    split = min((x for x in tally if tally[x] <= tolerance),
                key=lambda x: abs(x - width / 2))

    left = [w for r in rows for w in r if w["x0"] < split]
    right = [w for r in rows for w in r if w["x0"] >= split]
    # a column can be nearly empty - a page whose right-hand item is one line
    # of stem over four lettered pictures has six words in it
    if len(left) < 6 or len(right) < 6:
        return None
    if min(w["x0"] for w in left) > 0.25 * width:
        return None
    if max(w["x1"] for w in left) < 0.35 * width:
        return None
    if min(w["x0"] for w in right) > 0.65 * width:
        return None
    if max(w["x1"] for w in right) < 0.75 * width:
        return None
    return split


def page_lines(page) -> list[str]:
    words = [w for w in page.extract_words(use_text_flow=False)
             if w["top"] > HEAD_FRACTION * page.height
             and w["bottom"] < FOOT_FRACTION * page.height]
    if not words:
        return []
    rows = rows_of(words)
    split = gutter(rows, page.width)
    if split is None:
        columns = [words]
    else:
        columns = [[w for w in words if w["x0"] < split],
                   [w for w in words if w["x0"] >= split]]
    lines = []
    for column in columns:
        lines += [_join(r) for r in rows_of(column)]
    return [line for line in (l.strip() for l in lines) if line and not _is_noise(line)]


def read_pdf(path) -> tuple[list[list[str]], list[str]]:
    """(per-page column-ordered lines, per-page raw full-width text)."""
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        return ([page_lines(p) for p in pdf.pages],
                [p.extract_text() or "" for p in pdf.pages])


# --------------------------------------------------------------------------
# answer key
# --------------------------------------------------------------------------

KEY_HEADER = re.compile(r"questionnumber\s*correctanswer", re.IGNORECASE)
# A key row seen from the body side. Short booklets run the last item and the
# key table down the same page, so the body has to be cut at the table rather
# than at the page, and the header line is not always the first thing reached.
KEY_ROWISH = re.compile(r"^\d{1,3}\s+[A-D]\s+\S*\d")
# `4 1 C WH10.9.1 HI2 2004` - the item number itself can come out with a space
# in it, so the leading digits are collected loosely and despaced afterwards.
KEY_ROW = re.compile(r"^([\d ]{1,6}?)\s+([A-Za-z])\s+(\S.*)$")
YEAR = re.compile(r"(20[01]\d)$")
# `US11.1.3`, `WH1 0.7.1`, `8USH8.12.5`, `3MG 1.1` - despaced, then the grade
# is the run of digits before the first dot, after any subject prefix.
STANDARD = re.compile(r"^(?:[A-Z]{2,4})?(\d{1,2})[A-Z]*\d*\.")


def key_pages(raw: list[str]) -> int | None:
    """Index of the first page of the answer-key table.

    Matched on despaced text: several booklets render the header as
    "Cor rectAn swe r".
    """
    for i, text in enumerate(raw):
        if KEY_HEADER.search(re.sub(r"\s+", "", text)):
            return i
    return None


def parse_key(raw: list[str], start: int) -> list[tuple[int, str, str, int | None]]:
    """(item number, answer letter, standard, year) for every row, in order.

    Rows arrive in order and are accepted only when the number is the one
    expected next; anything else is a repeated header or a footnote. A row
    that fails to match truncates the key, which shortens the booklet - so the
    despacing is done before any field is read, not after.
    """
    rows: list[tuple[int, str, str, int | None]] = []
    for text in raw[start:]:
        for line in text.split("\n"):
            m = KEY_ROW.match(line.strip())
            if not m:
                continue
            number = re.sub(r"\s+", "", m.group(1))
            if not number.isdigit() or int(number) != len(rows) + 1:
                continue
            tail = re.sub(r"\s+", "", m.group(3))
            year = YEAR.search(tail)
            if year:
                tail = tail[:year.start()]
            rows.append((int(number), m.group(2).upper(), tail,
                         int(year.group(1)) if year else None))
    return rows


def key_grade(rows) -> int | None:
    """The grade the cited standards belong to, when they agree on one."""
    grades = Counter()
    for _, _, standard, _ in rows:
        m = STANDARD.match(standard)
        if m:
            grades[int(m.group(1))] += 1
    if not grades:
        return None
    grade, n = grades.most_common(1)[0]
    return grade if 2 <= grade <= 11 and n >= 0.5 * len(rows) else None


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------

# California prints far more of its stimulus material inline than Texas does -
# a quotation, a poster, a letter, a sentence to be edited - and the item then
# points at it. Anything that points at something the text cannot carry goes.
CA_SHARED = re.compile(
    r"\b(?:the\s+(?:sentence|sentences|quotation|quote|poster|cartoon|document"
    r"|excerpt|letter|speech|headline|advertisement|list|recipe|note|report"
    r"|outline|web|diary|editorial|proclamation|amendment\s+above)"
    r"|these\s+(?:sentences|quotations|documents|events|words)"
    r"|this\s+(?:sentence|quotation|excerpt|document|outline|list|note|report)"
    r"|underlined|the\s+box\s+below|the\s+information\s+(?:above|below)"
    r"|which\s+sentence\s+(?:from|in)\b)",
    re.IGNORECASE,
)

# The booklets announce an inline stimulus - a map, a timeline, an excerpt -
# on the line before the item it belongs to, and always for a single item
# ("the following question", never "questions 30 and 31"). That line sits
# outside the item's own block, so it has to be looked for explicitly.
DIRECTIVE = re.compile(
    r"Use the (?:\w+\s+){0,2}(?:excerpt|map|timeline|quotation|quote|information"
    r"|proof|passage|graph|table|chart|diagram|list|document|poster|cartoon)\b"
    r"|to (?:answer|complete) the (?:following|question|statement)",
    re.IGNORECASE,
)

# Stems that are a whole sentence to be completed by the option, which the
# CST prints with no terminal punctuation at all ("...delivered to the ancient
# Hebrews by"). STAAR never does this, so its STEM_END rule cannot be reused.
COMPLETION_END = re.compile(r"[A-Za-z,]$")
QUESTION_END = re.compile(r"[?:\u2014\u2013]$")

DROP_REASONS = ("parse", "shared_stimulus", "image", "debris", "degenerate")

# Superscripts flatten into the line and the result still reads as arithmetic,
# just different arithmetic: 3^2 x 5 becomes "32 x 5", 1725 ft^3 becomes
# "1725ft3". STAAR's rule for this misses both, because it looks for a word
# boundary before the unit that a digit-then-letter run does not have, and
# because nothing marks a lost exponent on a bare number. Anything that trades
# in exponents is dropped on the strength of the vocabulary instead. "Power"
# alone is not in the list: in the history booklets it is always political.
CA_FLATTENED = re.compile(
    r"\bprime factorization\b|\bexponents?\b|\bscientific notation\b"
    r"|to the \w+ power\b|\d\s?(?:cm|mm|km|in|ft|yd|m)\d\b"
    r"|[\u00d7x]\s*10\d",              # 5.048 x 10^2 cm, printed as "x 102 cm"
    re.IGNORECASE,
)

# STAAR's visual vocabulary plus the shapes California points at without ever
# naming them as a picture ("Which shapes make up this solid object?").
CA_VISUAL = re.compile(
    r"\bthis\s+(?:\w+\s+)?(?:solid|object|shape|prism|cylinder|cone|cube|pyramid|sphere"
    r"|polygon|triangle|rectangle|square|circle|angle|line segment|pattern"
    r"|design|sequence|net|spinner|arrangement|figure)\b"
    r"|\bthese\s+(?:shapes|solids|objects)\b",
    re.IGNORECASE,
)

PRODUCT = re.compile(r"(\d+)\s*[\u00d7x]\s*(\d+)")


def lost_exponent(stem: str, options: list[str]) -> bool:
    """A stem that states a product none of its whole-number options is.

    "Which expression is equivalent to 7^5 x 7^10?" flattens to "75 x 710"
    with 7^15 among the options as "715", and every part of it - stem and
    options alike - is a well-formed number, so nothing about the text looks
    damaged. What gives it away is the arithmetic: if the stem multiplies two
    numbers and not one option is the answer, the numbers are not the numbers
    that were printed.
    """
    m = PRODUCT.search(stem)
    if not m:
        return False
    values = [o.replace(",", "").strip() for o in options]
    if not all(re.fullmatch(r"\d+", v) for v in values):
        return False
    return str(int(m.group(1)) * int(m.group(2))) not in values


# An option that is wreckage on its own, whatever the stem looks like.
CA_BROKEN_OPTION = (
    re.compile(r"^\s*[=\u00d7\u00f7+\u2212]"),      # "=50 4" - an equation torn apart
    re.compile(r"\b\d \d\b"),                       # "5 0" - one number split in two
    # "C H O 6 12 6" is C6H12O6, "BF 3" is BF3: element symbols with their
    # subscripts shaken loose and parked at the end of the line
    re.compile(r"^[A-Za-z]{1,3}(?:\s+[A-Za-z]{1,3})*\s+\d[\d\s]*$"),
)

# The booklets set each operator as its own positioned glyph, so what survives
# is spaced by where it was drawn rather than by what it means: "5+ 12",
# "12 ÷5", "$10 −$2.50". Respace anything BETWEEN two operands, and only that,
# so a leading minus sign on a negative number is left where it is.
OPERATOR = re.compile(r"(?<=[\w)])\s*([\u00d7\u00f7\u2212+=])\s*(?=[\w($])")


def tidy(text: str) -> str:
    return OPERATOR.sub(r" \1 ", staar.tidy(text))


def classify(item: dict) -> str:
    """Return "" to keep, otherwise the reason to drop. Errs towards dropping."""
    if item["options"] is None:
        return "parse"
    stem = " ".join(item["stem_lines"]).strip()
    options = [o.strip() for o in item["options"]]
    if not stem or len(options) != 4 or not all(options):
        return "parse"
    if len(stem.split()) < 5:
        return "parse"
    # a completion stem is only recognisable by its options finishing the
    # sentence, so require them to be punctuated as sentence endings
    if not QUESTION_END.search(stem):
        finished = sum(1 for o in options if o.rstrip().endswith((".", "!", "?")))
        if not (COMPLETION_END.search(stem) and finished >= 3):
            return "parse"

    blob = stem + " \n " + " \n ".join(options)
    if item["under_stimulus"] or staar.SHARED.search(blob) or CA_SHARED.search(blob):
        return "shared_stimulus"
    if (staar.VISUAL.search(blob) or staar.FIGURE_GLYPHS.search(blob)
            or CA_VISUAL.search(blob)):
        return "image"
    if (staar.BAD_ENCODING.search(blob) or staar.FLATTENED_MATH.search(blob)
            or CA_FLATTENED.search(blob)):
        return "debris"
    if any(p.search(o) for o in options for p in CA_BROKEN_OPTION):
        return "debris"
    if lost_exponent(stem, options):
        return "debris"
    if any(not re.search(r"[aeiouyAEIOUY]", w) for w in staar.NO_VOWEL.findall(blob)):
        return "debris"
    if any(staar.looks_like_caption(line) for line in item["stem_lines"][:-1]):
        return "debris"
    if len(set(o.lower() for o in options)) != 4:
        return "degenerate"
    if any(len(re.sub(r"\W", "", o)) <= 2 for o in options):
        return "degenerate"
    # The scan for the end of the last option runs on until it meets the next
    # item's number, so a figure standing between the two - the "39.06" of a
    # vertical multiplication, say - is read as more of option D. Detectable
    # because it leaves D longer than any of its siblings and ending in a
    # detached number. Dropped rather than trimmed: an option really can end
    # in a number, and mangling a gold answer is worse than losing an item.
    tail = options[3].split()
    if (len(tail) > max(len(o.split()) for o in options[:3])
            and re.fullmatch(r"[\d.,$%]+", tail[-1])):
        return "debris"
    return ""


# --------------------------------------------------------------------------
# one booklet
# --------------------------------------------------------------------------

AXIS_ROW = re.compile(r"^\d+(?:\s+\d+){2,}$")   # "0 1 2 3 4 5 6 7 8" down a graph


def opener_lines(lines: list[str], item_no: int) -> list[int]:
    """Every line that could be where item `item_no` starts.

    Looser than the STAAR version, which requires the stem to begin with a
    capital. California opens items with a bracketed read-aloud script in the
    grade 2 booklets ("1 [A NUMBER HAS NINE ONES...]") and with bare arithmetic
    in the mathematics ones ("6 9000 3782"), and requiring a capital loses
    both - which loses the whole booklet, since a single unfindable item
    number stops the alignment.

    Being loose is affordable because the caller scores candidates by whether
    the item's options actually follow, so a wrong candidate costs a point
    rather than derailing the run. Two prunings are still worth it: a row of
    bare numbers is usually a graph axis, and a bare number sitting inside a
    consecutive run of bare numbers is usually an axis label. Both are only
    PREFERENCES - if they would leave the item with no candidate at all they
    are dropped, because a doubtful line handed to the scorer is better than a
    refused booklet, and an arithmetic item flattened out of its own layout
    looks exactly like an axis ("51 18 3 3 2" is (18 + 3) / (3 - 2) = ).
    """
    exact = re.compile(rf"^{item_no}(?:$|\s+\S)")
    hits, doubtful = [], []
    for i, line in enumerate(lines):
        if not exact.match(line):
            continue
        near = [lines[x] for x in (i - 1, i + 1) if 0 <= x < len(lines)]
        if AXIS_ROW.match(line) or (
                line.isdigit()
                and any(n.isdigit() and abs(int(n) - item_no) == 1 for n in near)):
            doubtful.append(i)
        else:
            hits.append(i)
    return (hits or doubtful)[:staar.MAX_CANDIDATES]


def item_starts(lines: list[str], key: dict[int, str],
                alternating: bool) -> tuple[dict[int, int] | None, str]:
    """Find the line that opens each item.

    The STAAR routine with `opener_lines` swapped for the one above; the
    reasoning behind it is in `staar_extract.item_starts` and is unchanged.
    Briefly: alignment is anchored on item numbers because they are complete
    and monotone even when an item's body extracts as garbage, and the whole
    assignment is chosen at once - over all strictly increasing candidate
    sequences, the one that gets the most items to have their options where
    they should be - because committing item by item lets one bad early pick
    run the cursor past everything after it.
    """
    n_items = max(key)
    cands: dict[int, list[int]] = {}
    good: dict[int, set[int]] = {}
    for item_no in range(1, n_items + 1):
        cands[item_no] = opener_lines(lines, item_no)
        good[item_no] = set()
        if key.get(item_no) in staar.ALL_LETTERS:
            letters = staar.letters_for(item_no, alternating)
            for c in cands[item_no]:
                block = staar.find_block(
                    lines, c, min(c + staar.LOOKAHEAD, len(lines)), letters)
                if block and not any(c < other < block[0] for other in cands[item_no]):
                    good[item_no].add(c)
        if not cands[item_no]:
            return None, f"no line opens item {item_no} of {n_items}"

    nxt: dict[int, dict[int, int]] = {}
    best: dict[int, dict[int, int]] = {n_items + 1: {len(lines): 0}}
    for item_no in range(n_items, 0, -1):
        best[item_no], nxt[item_no] = {}, {}
        later = best[item_no + 1]
        for c in cands[item_no]:
            options = [(later[j] + (c in good[item_no]), -j) for j in later if j > c]
            if not options:
                continue
            score, pick = max(options)
            best[item_no][c] = score
            nxt[item_no][c] = -pick
    if not best[1]:
        return None, f"no run of {n_items} numbered lines is in order"

    start = max(best[1], key=lambda c: (best[1][c], -c))
    placed, item_no = {}, 1
    while item_no <= n_items:
        placed[item_no] = start
        start = nxt[item_no][start]
        item_no += 1
    return placed, ""


def parse_booklet(lines: list[str], key: dict[int, str],
                  subject: str) -> tuple[list[dict] | None, str]:
    """Split a booklet into items, or refuse the whole booklet.

    Alignment is anchored on ITEM NUMBERS rather than on option letters, as on
    STAAR and for the same reason: an item whose options are unreadable would
    otherwise shift every label after it by one - silently, since each item
    still looks individually fine.

    Stimulus detection differs from STAAR, though. There, one passage serves a
    long run of items, so the "under a stimulus" flag latches on and stays on.
    California only does that in the ELA booklets; its mathematics, science
    and history booklets print a stimulus inside the single item that uses it
    and announce it on the line above ("Use the map to answer the following
    question" - always singular). Latching in those booklets condemns the rest
    of the booklet the first time a stray figure caption looks like prose,
    which on the grade 8 history booklet was 108 items out of 109.
    """
    alternating, why = staar.key_scheme(key)
    if alternating is None:
        return None, why
    starts, why = item_starts(lines, key, alternating)
    if starts is None:
        return None, why

    latching = subject == "reading"
    bounds = {n: (starts[n], starts.get(n + 1, len(lines))) for n in starts}
    items = []
    lead = lines[:starts[min(starts)]]
    stimulus_seen = staar.prose_words(lead) >= staar.STIMULUS_WORDS
    previous_tail = 0
    for item_no in sorted(key):
        lo, hi = bounds[item_no]
        answer = key[item_no]
        letters = staar.letters_for(item_no, alternating)
        if answer not in letters:
            return None, f"item {item_no} answers {answer} but its options are {letters}"
        found = staar.find_block(lines, lo, hi, letters)
        start, tail, texts = found if found else (hi, hi, None)

        run_up = lines[previous_tail:lo]
        under = (stimulus_seen
                 or staar.prose_words(run_up) >= staar.STIMULUS_WORDS
                 or any(DIRECTIVE.search(line) for line in run_up))

        stem_lines = [s for s in lines[lo:start] if s]
        if stem_lines:
            stem_lines[0] = re.sub(rf"^{item_no}\s*", "", stem_lines[0]).strip()
            stem_lines = [s for s in stem_lines if s]
        items.append({
            "item_no": item_no,
            "stem_lines": stem_lines,
            "letters": letters,
            "options": texts,
            "answer": answer,
            "under_stimulus": under,
        })
        stimulus_seen = latching and under
        previous_tail = tail
    return items, ""


def extract_booklet(path, subject: str, grade: int, stats: Counter) -> list[dict]:
    stem = path.stem
    pages, raw = read_pdf(path)
    start = key_pages(raw)
    if start is None:
        stats["form_no_key"] += 1
        print(f"  {stem}: DROPPED, no answer-key table found")
        return []
    rows = parse_key(raw, start)
    if not rows:
        stats["form_no_key"] += 1
        print(f"  {stem}: DROPPED, answer-key table did not parse")
        return []

    cited = key_grade(rows)
    if cited is not None and cited != grade and stem not in ODD:
        print(f"  {stem}: NOTE, cover says grade {grade}, standards cite {cited}")

    key = {n: a for n, a, _, _ in rows}
    years = {n: y for n, _, _, y in rows}
    # Everything up to the key table is item pages and standards lists. The
    # table itself has to go, or its rows are read as `1 D ...` item openers -
    # and it has to be cut at the table, not at the page, because a short
    # booklet prints its last item above the table on the same page.
    lines = [line for page in pages[:start] for line in page]
    for line in pages[start]:
        if KEY_HEADER.search(re.sub(r"\s+", "", line)) or KEY_ROWISH.match(line):
            break
        lines.append(line)

    items, why = parse_booklet(lines, key, subject)
    if items is None:
        stats["form_misaligned"] += 1
        stats["dropped_by_misaligned_form"] += len(key)
        print(f"  {stem}: DROPPED whole booklet, {why}")
        return []

    stats["raw"] += len(key)
    kept, reasons = [], Counter()
    for item in items:
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
            "source": "ca",
            "subject": subject,
            "grade": grade,
            "year": years.get(item["item_no"]),
            "booklet": stem,
            "item_no": item["item_no"],
        })
    stats["kept"] += len(kept)
    detail = " ".join(f"{r}={reasons[r]}" for r in DROP_REASONS if reasons[r])
    print(f"  {stem:24s} {subject:15s} g{grade:<3} {len(key):4d} items, "
          f"{detail or 'no drops'} -> kept {len(kept)}")
    return kept


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def main():
    ap = argparse.ArgumentParser(
        description="Extract self-contained MC items from California's released CST "
                    "questions. OUTPUT IS CDE-COPYRIGHTED AND GITIGNORED - do not commit it.")
    ap.add_argument("--download", action="store_true",
                    help="enumerate the Wayback CDX index and fetch the PDFs first")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only", nargs="*", help="restrict to these booklet stems")
    ap.add_argument("--show", type=int, default=0, help="print N kept items in full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    CA_DIR.mkdir(parents=True, exist_ok=True)
    if args.download:
        print("enumerating the Wayback CDX index for cde.ca.gov/ta/tg/sr/documents/ ...")
        rep = download_all()
        print(f"  archived PDFs: {rep['enumerated']}   fetched: {rep['fetched']}   "
              f"already had: {rep['cached']}   failed: {len(rep['failed'])}")
        for name in rep["failed"]:
            print(f"    failed: {name}")

    booklets, skipped = [], []
    for path in sorted(PDF_DIR.glob("*.pdf")):
        if args.only and path.stem not in args.only:
            continue
        what = classify_file(path.stem)
        if what is None:
            skipped.append(path.stem)
            continue
        booklets.append((generation(path.stem), path, *what))
    booklets.sort(key=lambda b: (b[0], b[3], b[2]))

    unclassified = [s for s in skipped if not EOC.search(s)]
    print(f"\n{len(booklets)} graded booklets, {len(skipped)} skipped "
          f"(end-of-course, no grade on the cover)")
    if unclassified:
        print(f"  UNRECOGNISED, not parsed: {' '.join(unclassified)}")

    stats = Counter()
    print("\nextracting ...")
    found: dict[str, list[dict]] = defaultdict(list)
    for _, path, subject, grade in booklets:
        for item in extract_booklet(path, subject, grade, stats):
            found[normalized(item["question"])].append(item)

    # The four generations re-release each other's items, which is a free
    # check on the labelling that no amount of internal consistency could
    # give: two booklets typeset years apart, parsed independently, have to
    # agree on which option is correct. Where they do not, one of the two is
    # misaligned and there is no way to tell which, so both go. (This is not
    # hypothetical - it catches an item the grade 8 history booklet of 2006
    # answers A and the 2009 one answers B.)
    #
    # Agreement is on the option's POSITION, and then on its text only as far
    # as the shorter copy runs. An older booklet sometimes trails figure
    # wreckage into its last option, and that is a difference in how well the
    # page extracted, not a difference about which option is correct.
    items: list[dict] = []
    for copies in found.values():
        golds = sorted({normalized(c["choices"][c["gold_idx"]]) for c in copies}, key=len)
        if (len({c["gold_idx"] for c in copies}) > 1
                or not all(g.startswith(golds[0]) for g in golds)):
            stats["contradicted"] += len(copies)
            continue
        stats["duplicate"] += len(copies) - 1
        stats["cross_checked"] += len(copies) > 1
        items.append(copies[0])
    items.sort(key=lambda i: (i["subject"], i["grade"], i["booklet"], i["item_no"]))

    with open(args.out, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    print(f"\nwrote {len(items)} items -> {args.out}   (COPYRIGHTED, GITIGNORED)")
    print(f"\n{'raw items in the parsed booklets':<36} {stats['raw']}")
    for reason in DROP_REASONS:
        if stats[reason]:
            print(f"{'  dropped: ' + reason:<36} {stats[reason]}")
    if stats["duplicate"]:
        print(f"{'  dropped: re-release of an earlier':<36} {stats['duplicate']}")
    if stats["contradicted"]:
        print(f"{'  dropped: re-releases disagree on gold':<36} {stats['contradicted']}")
    if stats["form_no_key"]:
        print(f"{'  booklets dropped, no key':<36} {stats['form_no_key']}")
    if stats["form_misaligned"]:
        print(f"{'  booklets dropped, key misaligned':<36} {stats['form_misaligned']} "
              f"({stats['dropped_by_misaligned_form']} items)")
    print(f"{'kept, deduplicated':<36} {len(items)}")
    if stats["cross_checked"]:
        agreed = stats["cross_checked"]
        total = agreed + stats["contradicted"] // 2
        print(f"  of which {agreed} appear in two or more booklets and every copy "
              f"agrees on the gold option ({agreed / max(1, total):.1%} of {total})")

    print("\nkept by subject x grade")
    grid = defaultdict(Counter)
    for item in items:
        grid[item["subject"]][item["grade"]] += 1
    for subject in sorted(grid, key=lambda s: -sum(grid[s].values())):
        row = "  ".join(f"g{g}={grid[subject][g]}" for g in sorted(grid[subject]))
        print(f"  {subject:<16} {sum(grid[subject].values()):4d}   {row}")
    by_year = Counter(item["year"] for item in items)
    print("years: " + "  ".join(f"{y}={by_year[y]}" for y in sorted(by_year, key=str)))

    if items:
        print("\ngold answer length" + " " * 12 + "n   median  mean  1-word")
        for label, group in [("all", items)] + [
                (s, [i for i in items if i["subject"] == s]) for s in sorted(grid)]:
            lengths = [len(i["choices"][i["gold_idx"]].split()) for i in group]
            print(f"  {label:<24} {len(lengths):5d} {statistics.median(lengths):6.0f} "
                  f"{statistics.mean(lengths):6.1f} "
                  f"{sum(1 for n in lengths if n == 1) / len(lengths):7.1%}")

    if args.show:
        import random

        sample = list(items)
        random.Random(args.seed).shuffle(sample)
        per_subject = max(1, args.show // max(1, len(grid)))
        seen_subject, picked = Counter(), []
        for item in sample:
            if seen_subject[item["subject"]] < per_subject + 2:
                seen_subject[item["subject"]] += 1
                picked.append(item)
            if len(picked) == args.show:
                break
        for n, item in enumerate(picked, 1):
            print(f"\n--- {n}. {item['subject']} grade {item['grade']} {item['year']} "
                  f"({item['booklet']} item {item['item_no']}) ---")
            print(item["question"])
            for i, choice in enumerate(item["choices"]):
                mark = " <-- GOLD" if i == item["gold_idx"] else "        "
                print(f"  {'ABCD'[i]}{mark} {choice}")


if __name__ == "__main__":
    main()
