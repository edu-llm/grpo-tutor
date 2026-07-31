"""Turn released New Jersey test PDFs into self-contained multiple-choice items.

    !!! THE EXTRACTED CONTENT IS NOT REDISTRIBUTABLE. DO NOT COMMIT IT. !!!

The NJ booklets carry either "Copyright (c) New Jersey Department of Education"
or, for the 2007 sampler, a Riverside Publishing notice that is stricter still.
This repository is PUBLIC, so everything downloaded or derived lands under
`data/state_tests/`, which is in `.gitignore`, and this file is the only thing
that gets committed. Rebuild the data instead of committing it:

    python src/extract_nj.py --download

This is the New Jersey sibling of `src/staar_extract.py` and borrows its
machinery wholesale: the magic-bytes download check, the pdfplumber line
reader, the item-number-anchored option finder, and the filter battery. Read
that file first; only the differences are explained here.

WHAT NEW JERSEY ACTUALLY HAS
----------------------------
NJ is not one archive but four, spanning 2000-2026, and the interesting part is
that the *newest* source is the worst one:

  1. NJSLA-Adaptive (current, Cambium portal). Paper practice tests for
     grades 3-8 in maths and ELA plus per-grade answer keys. The keys are
     public but well hidden: the portal renders from JSON, and the key PDFs
     only appear after walking
     /widgets/en/resources-njsla-practice-test-paper ->
     content/resourcelist/en/... -> content/resourceitem/en/..., which finally
     names a file with an EN DASH in it ("... Practice Test - Answer Key.pdf").
     Guessing the filename 404s; the walk is in `cambium_answer_keys()`.
  2. NJSLA-Science (current, Measurement Inc portal, nj.mymisupport.com).
     Grades 5, 8 and 11, three units each, every test paired with an
     "Answer & Alignment Document" that states each item's TYPE as well as its
     key - the only source here that hands over "Item Type: Multiple Choice"
     instead of making us infer it.
  3. GEPA 2000-2001 Sample Form, grade 8, four subjects (Wayback). The only
     New Jersey SOCIAL STUDIES items that exist anywhere with answers.
  4. NJ ASK released samples, 2006 (grades 3-4) and 2007 (grades 5-7)
     (Wayback), with scoring keys in appendices.

Sources probed and rejected, so nobody re-runs them:

  * nj.mypearsonsupport.com - NXDOMAIN. NJ left Pearson for Cambium; every
    "NJSLA released items" link on the web still points here and none resolve.
  * nj.digitalitemlibrary.com - NXDOMAIN, same reason. This was the good one:
    it served released operational items with bulk PDF export. It is gone, and
    it is why this extraction is as small as it is.
  * PARCC released items / New Meridian - the NJ-facing links are all via the
    two dead hosts above.
  * ESPA 2004 maths and 2006 science (grade 4), NJ ASK 2003 release samples -
    the booklets are on Wayback but NO answer key was ever posted with them and
    none is embedded, so they are dropped whole rather than guessed at.
  * GEPA social studies scoring guide - never published. The archived index of
    /education/assessment/ms/sample/ lists la_, math_ and science_ guides and
    no fourth. See below for why social studies survives anyway.

THE OLD BOOKLETS ARE TWO-COLUMN, AND THAT IS NOT COSMETIC
---------------------------------------------------------
GEPA and the NJ ASK samplers print two columns per page. pdfplumber reads a
page in reading order by y then x, so it INTERLEAVES the columns:

    46. Which of the following is a characteristic   DIRECTIONS FOR QUESTION 48
    of a free-market economy?                        to the open-ended question

comes out as one line of each, alternating, and every item on the page is
shredded into a plausible-looking mixture of two different items. Nothing
downstream can detect that. So `page_lines()` finds the gutter first - the
widest vertical band in the middle 36% of the page that essentially no word
crosses - and crops the page into two half-pages, left then right. The
tolerance matters: figure captions ("Liver Pancreas Stomach") straddle both
columns on an otherwise two-column page, and demanding a perfectly empty band
misses those pages and silently falls back to interleaving.

WHERE THE ANSWERS COME FROM, AND WHY SOCIAL STUDIES IS TRUSTED
--------------------------------------------------------------
Four key formats, one per family; see the `*_key` functions. The GEPA one is
worth spelling out because it looks like guesswork and is not.

The GEPA booklets print a per-item tracking code after each item's options:

    3. A raft is floating on a lake. ...
    A. Force equals mass times acceleration.
    ...
    8SPCCF-050C

In maths and science the code's LAST CHARACTER is the correct answer. In
social studies the code carries no letter and the answer is printed on the
following line as "A *****". That is a claim about the documents, so it is
tested rather than assumed: maths and science also have official scoring
guides, and `extract_gepa()` parses both and refuses the form if the embedded
letters CONTRADICT the official key anywhere. They never do - 30/34 and 35/37
items carry a readable annotation and every one of them agrees, with zero
conflicts. Social studies has no official guide and is accepted on the
strength of that, plus a structural argument: this key is printed NEXT TO ITS
OWN ITEM rather than in a separate list, so the failure that motivates rule 4
- a list that slips by one and silently relabels everything after it - cannot
happen here. An item whose annotation is unreadable loses itself and nothing
else, and an item that somehow collects two is dropped rather than guessed at.

That failure is not hypothetical, incidentally. It happened once, in the 2006
maths sampler, and `key_page()` records how.

WHAT CAME OUT
-------------
59 items from 21 forms, no form refused: 22 maths (grades 3-8), 23 science
(grades 4, 5, 8) and 14 social studies (grade 8). Gold answers run to a median
of 4 words with 17% single-word, against 5w/13% for STAAR and 2w/31% for
OpenBookQA, so the answers are sentences rather than tokens.

463 raw items went in and 404 were dropped, which is the honest number for New
Jersey rather than a parser that needs more work:

  * 153 are not single-answer multiple choice AT ALL - drag-and-drop,
    multi-select, equation editor, constructed response. These are counted
    from the keys, not guessed. NJSLA is a next-generation assessment and this
    is most of it: of 111 maths items across grades 3-8, 28 have a single
    letter as their answer.
  * 91 are image-dependent and 41 sit under a shared stimulus. NJSLA-Science
    is entirely cluster-based - every item hangs off a shared "Phenomenon"
    scenario with a diagram - so almost none of it can survive, and 5 items
    from the whole modern corpus do.
  * 119 lost to parse failures, extraction debris and degenerate options.

So the yield is carried by the 2000s-era booklets, which are ordinary
four-option paper tests, and the newest and most accessible source is the one
that gives almost nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import paths

NJ_DIR = paths.DATA / "state_tests"
PDF_DIR = NJ_DIR / "pdf_nj"

CAMBIUM = "https://nj.portal.cambiumast.com"
MEASINC = "https://nj.mymisupport.com/resources/Practice-Tests/science"
NJDOE = "www.state.nj.us/education/assessment"

GRADES = (3, 4, 5, 6, 7, 8)
LETTERS = "ABCD"

# Wayback keeps several captures per file and not all of them are the PDF; the
# `{year}id_` form redirects to the nearest capture, so a handful of years is a
# cheaper way to find a live one than walking the CDX index. The 2007 sampler
# key is the exception - only one capture of it ever succeeded.
WAYBACK_YEARS = ("2013", "2008", "2016", "2011", "2006")
WAYBACK_EXACT = {"final_NJASK07_Sampler20Key_Rubric_Exemplars.pdf": "20121114194427"}


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------

def fetch(url: str, timeout: int = 180) -> bytes | None:
    """Return the body only if it is really a PDF.

    Same reasoning as the STAAR fetcher: the Cambium portal answers unknown
    paths with a 403 HTML page and Wayback answers with a 200 HTML page, so
    neither the status code nor the length can be trusted. The magic bytes can.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return None
    return body if body[:5] == b"%PDF-" else None


def wayback(path: str) -> bytes | None:
    name = path.rsplit("/", 1)[-1]
    stamps = [WAYBACK_EXACT[name]] if name in WAYBACK_EXACT else list(WAYBACK_YEARS)
    for stamp in stamps:
        body = fetch(f"https://web.archive.org/web/{stamp}id_/http://{path}")
        if body:
            return body
    return None


def cambium_answer_keys() -> dict[int, str]:
    """grade -> URL of the NJSLA maths answer key.

    The portal is a JSON-rendered SPA and the key filenames are not guessable
    (they contain an en dash), so the only reliable way in is the walk the page
    itself does: the paper-practice-test widget names a ResourceList, the
    ResourceList names ResourceItems, and each ResourceItem names a PDF.
    """
    def load(url):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError, ValueError):
            return None

    widget = load(f"{CAMBIUM}/widgets/en/resources-njsla-practice-test-paper")
    if not widget:
        return {}
    lists = [c["contentPath"] for p in widget.get("panels", [])
             for c in p.get("contentData", [])
             if c and c.get("contentType") == "ResourceList"]
    out: dict[int, str] = {}
    for path in lists:
        rlist = load(f"{CAMBIUM}/{path}")
        if not rlist:
            continue
        for item_path in rlist.get("resourceItemsPath", []):
            item = load(f"{CAMBIUM}/{item_path}")
            if not item:
                continue
            resource = (item.get("languageObject") or {}).get("resourcePath")
            title = item.get("displayTitle", "")
            m = re.search(r"Grade (\d+) Mathematics", title)
            if resource and m:
                grade = int(m.group(1))
                if grade in GRADES:
                    out[grade] = CAMBIUM + urllib.parse.quote(resource)
    return out


def download_plan() -> dict[str, str]:
    """local filename -> URL, for everything this extractor can use."""
    plan: dict[str, str] = {}
    base = f"{CAMBIUM}/content/contentresources/en/"
    for grade in GRADES:
        name = f"NJSLA-Adaptive Grade {grade} Mathematics Paper-Based Practice Test.pdf"
        plan[f"math_g{grade}_test.pdf"] = base + urllib.parse.quote(name)
    for grade, key_url in cambium_answer_keys().items():
        plan[f"math_g{grade}_key.pdf"] = key_url

    for grade in (5, 8):
        for unit in (1, 2, 3):
            stem = f"NJSLAS_PracticeTest_Unit{unit}_Grade{grade}"
            plan[f"sci_g{grade}_u{unit}_test.pdf"] = f"{MEASINC}/{stem}.pdf"
            plan[f"sci_g{grade}_u{unit}_key.pdf"] = (
                f"{MEASINC}/{urllib.parse.quote(stem + chr(95) + 'Answer&Alignment_Document_20260122')}.pdf")

    for name in ("test_book_math", "test_book_sci", "test_book_ss",
                 "math_scoring_guide", "science_score_guide"):
        plan[f"gepa_{name}.pdf"] = f"WB:{NJDOE}/ms/sample/{name}.pdf"
    plan["njask06_math.pdf"] = f"WB:{NJDOE}/es/sample/NJ06_Math_sample.pdf"
    plan["njask06_sci_g4.pdf"] = f"WB:{NJDOE}/es/sample/NJ06_G4Sci_sample.pdf"
    for grade, stem in ((5, "NJ07_TB_G5_SamplerFinal"),
                        (6, "NJ07_TB_G6_FA_SamplerFinal"),
                        (7, "NJ07_TB_G7_FA_SamplerFinal")):
        plan[f"njask07_g{grade}.pdf"] = f"WB:{NJDOE}/ms/samples2007/{stem}.pdf"
    plan["njask07_key.pdf"] = (
        f"WB:{NJDOE}/ms/samples2007/final_NJASK07_Sampler20Key_Rubric_Exemplars.pdf")
    return plan


def download_all(workers: int = 6) -> dict:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    plan = download_plan()

    def one(pair):
        name, url = pair
        dest = PDF_DIR / name
        if dest.exists() and dest.stat().st_size > 1024:
            return name, "cached"
        body = wayback(url[3:]) if url.startswith("WB:") else fetch(url)
        if not body:
            return name, "MISSING"
        dest.write_bytes(body)
        return name, "ok"

    report = {}
    # Wayback throttles hard; a small pool keeps it answering.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for name, status in ex.map(one, sorted(plan.items())):
            report[name] = status
    return report


# --------------------------------------------------------------------------
# text extraction
# --------------------------------------------------------------------------

BOILERPLATE = re.compile(
    r"^\s*(?:"
    r"Page\s*\d+|GO ON(?:\s+TO THE NEXT PAGE\.?)?|STOP|Go On|Continue"
    r"|GEPA\b.*|NJ\s*ASK\b.*|NJSLA[\u2013-]?S?\b.*|Grade\s*\d+\s*(?:Mathematics|Science|Social Studies)?"
    r"|Mathematics|Science|Social Studies|Language Arts(?: Literacy)?|Unit\s*\d+"
    r"|Copyright\s*©.*|All rights reserved.*|SECURE MATERIAL.*|DO NOT (?:COPY|WRITE).*"
    r"|New Jersey.*Assessment.*|Assessment Samples|RELEASED SAMPLE.*|DAY\s*\d+"
    r"|Directions:?|Student Name|Practice Test|.*Practice Test"
    r"|Calculators? (?:may|MAY).*|\(?(?:Non-)?Calculator\)?"
    r"|(?:\(cid:\d+\)\s*)+"           # answer-grid bubbles, whole lines of them
    r"|[\u2022\u00b7~`*_\-\s]+"       # rule lines and bubble-sheet debris
    r"|\d{1,4}"                       # bare page numbers
    r")\s*$",
    re.IGNORECASE,
)

# The GEPA/NJ ASK per-item tracking codes. They double as the answer key for
# GEPA (see the module docstring) and are always noise inside an item's text.
GEPA_CODE = re.compile(r"^[\d*][A-Z0-9*]{3,}[-\u2013][A-Z0-9*]{3,}$")
GEPA_ANSWER_LINE = re.compile(r"^([A-D])\s+\*{3,}$")

# Directions prose sits in front of item 1 and reads as forty-odd words of
# ordinary English, so left in place it is measured as a shared stimulus and
# condemns the entire booklet - every item inherits `under_stimulus`. Each
# phrase below is boilerplate printed verbatim in these booklets.
INSTRUCTIONS = re.compile(
    r"Read each question|determine the best answer|four answer choices"
    r"|answer folder|answer document|Then fill in|choose the best answer"
    r"|fill in the circle|Choose the best of the answer choices"
    r"|Sample (?:Multiple-Choice|Open-Ended|Item|Question)|Answers to Sample"
    r"|Work as rapidly as you can|Do not spend too much time"
    r"|blank spaces in the test booklet|RECORD ALL OF YOUR ANSWERS"
    r"|No credit will be given|DIRECTIONS FOR|made up of three parts"
    r"|Respond fully|Show your work|Write your answer|record your answer"
    r"|explain your answer|graded on the correctness|accuracy of your answer"
    r"|use words, tables, diagrams|If you (?:have time|finish|do not know)"
    r"|review your (?:work|answers)|go on to the next question"
    r"|remember these important things|You will (?:take|be taking|write|select)"
    r"|Today you will|This is a test of|you may use a calculator"
    r"|you may NOT use a calculator|Select the one that is best"
    r"|incomplete statements below|do NOT include sales tax"
    r"|when you are told to do so|come back to the skipped question",
    re.IGNORECASE,
)

TIDY = (("\u201a", ","), ("\u2212\u2212", "\u2212"), ("\u2013\u2013", "\u2013"))

# An item's tracking code printed at the end of its last option rather than on
# a line of its own, where the line filter would have caught it.
INLINE_CODE = re.compile(r"\s*\b\d[A-Z][A-Z0-9*]{2,}[-\u2013][A-Z0-9*]{3,}\b")


def tidy(text: str) -> str:
    for bad, good in TIDY:
        text = text.replace(bad, good)
    text = INLINE_CODE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _crossers(words, x: float) -> int:
    return sum(1 for w in words if w["x0"] < x - 1 and w["x1"] > x + 1)


def _balanced(words, x: float, limit: int = 0) -> bool:
    """Is `x` a usable split: few words across it and real text on both sides?"""
    if _crossers(words, x) > limit:
        return False
    left = sum(1 for w in words if w["x1"] <= x)
    right = sum(1 for w in words if w["x0"] >= x)
    return min(left, right) >= 0.25 * len(words)


def gutter_band(page, min_gap: float = 12.0) -> tuple[float, float] | None:
    """The x-interval of a two-column page's gutter, or None if single column.

    Measured over LINE boxes, and over an interval that no line occupies rather
    than one that few lines straddle. Both details are paid for:

    "Few straddle it" is not enough. A left-column line ending at x=270
    straddles a split at x=254 not at all, so that test accepted the split with
    the line's tail stranded on the right, and every right-column line came out
    with the left column's tail glued to its front ("ean's 8. A computer
    manufacturer..."). Items 8-12 of the GEPA maths booklet went that way.

    Measured over WORDS, not over pdfplumber's text lines. Its lines are
    clusters by vertical position across the whole page, so on a two-column
    page a left line and a right line at the same height are ONE line spanning
    the full width - the very interleaving this is trying to undo - and no
    gutter is ever visible in them.

    A figure that genuinely straddles the columns fills the gap and hides it;
    such a page returns None and the caller falls back to the house gutter.
    """
    words = page.extract_words()
    if len(words) < 25:
        return None
    lo = min(w["x0"] for w in words)
    hi = max(w["x1"] for w in words)
    if hi - lo < 100:
        return None
    spans = sorted((w["x0"], w["x1"]) for w in words)

    gaps = []
    reach = spans[0][1]
    for a, b in spans[1:]:
        if a - reach >= min_gap:
            gaps.append((reach, a))
        reach = max(reach, b)
    middle = [g for g in gaps
              if lo + 0.30 * (hi - lo) <= (g[0] + g[1]) / 2 <= lo + 0.70 * (hi - lo)]
    if not middle:
        return None
    centre = (lo + hi) / 2
    a, b = max(middle, key=lambda g: (g[1] - g[0], -abs((g[0] + g[1]) / 2 - centre)))
    left = sum(1 for x0, x1 in spans if x1 <= a)
    right = sum(1 for x0, x1 in spans if x0 >= b)
    if min(left, right) < 0.25 * len(spans):
        return None
    return (a, b)


def house_gutter(bands: list[tuple[float, float] | None]) -> float | None:
    """The one x that lies inside as many pages' gutter bands as possible.

    A gutter belongs to the TEMPLATE, so the document as a whole pins it far
    better than any single page can: pages whose columns run full-length give a
    narrow band, and the point they agree on is the real gutter. Pages with a
    loose band then inherit it instead of guessing at their own midpoint.
    """
    found = [b for b in bands if b is not None]
    if len(found) < 3:
        return None
    best = None
    for edge in sorted({x for band in found for x in band}):
        for x in (edge + 0.5, edge - 0.5):
            hits = sum(1 for a, b in found if a <= x <= b)
            cand = (-hits, abs(x - sum((a + b) / 2 for a, b in found) / len(found)))
            if best is None or cand < best[0]:
                best = (cand, x)
    return None if best is None else best[1]


EDGE_LINES = 2      # how deep into a page a running head may reach


def body_band(pages, share: float = 0.35) -> list[tuple[float, float]]:
    """Per page, the (top, bottom) y-range that is not running head or footer.

    Naming the heads by their text does not work once columns are cropped: the
    crop cuts "GEPA - SECURE MATERIAL - DO NOT COPY" at the gutter and each
    half becomes a new string nobody listed. Left in, a head lands inside
    whichever option is last on the page and corrupts its text instead of
    failing. So they are removed geometrically and BEFORE the column split -
    a line at the very top or bottom of a page whose text repeats across a
    third of the pages is a running head, and the band is closed below it.
    """
    per_page = [p.extract_text_lines() or [] for p in pages]
    seen = Counter()
    for lines in per_page:
        edge = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        seen.update({ln["text"].strip().casefold() for ln in edge
                     if len(ln["text"].strip()) >= 4})
    floor = max(3, int(share * len(pages)))
    heads = {text for text, n in seen.items() if n >= floor}

    bands = []
    for page, lines in zip(pages, per_page):
        top, bottom = 0.0, float(page.height)
        for ln in lines[:EDGE_LINES]:
            if ln["text"].strip().casefold() in heads:
                top = max(top, ln["bottom"] + 0.5)
        for ln in reversed(lines[-EDGE_LINES:]):
            if ln["text"].strip().casefold() in heads:
                bottom = min(bottom, ln["top"] - 0.5)
        bands.append((top, bottom) if bottom - top > 0.4 * page.height
                     else (0.0, float(page.height)))
    return bands


def raw_pages(pdf_path) -> list[str]:
    """Page text, with running heads and two-column layout dealt with first.

    Pages where detection fails still get split if the house gutter is a
    plausible divide for them: on the one GEPA maths page where per-page
    detection failed (a network diagram drawn across the gutter), the
    interleaved text handed item 33's answer code to item 32.
    """
    import pdfplumber

    out = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        bands = body_band(pdf.pages)
        bodies = [p.crop((0, t, p.width, b)) for p, (t, b) in zip(pdf.pages, bands)]
        words = [p.extract_words() for p in bodies]
        gutters = [gutter_band(p) for p in bodies]
        house = house_gutter(gutters)
        # Whether the fallback below is allowed at all is a property of the
        # DOCUMENT. In a two-column booklet a page with no clean gutter is a
        # page whose figure straddles it, and splitting at the house gutter
        # recovers it. In a single-column one there is no gutter to fall back
        # to, and splitting anyway cuts words in half down the whole form
        # ("Find the exact answe" / "r: 145 + 281"). So the fallback is only
        # offered when most pages agree the document has two columns.
        pages_with_text = sum(1 for w in words if len(w) >= 25)
        agreeing = sum(1 for b in gutters if b and house is not None
                       and b[0] <= house <= b[1])
        two_column = (house is not None and agreeing >= 4
                      and agreeing >= 0.20 * max(1, pages_with_text))
        for body, band, wds in zip(bodies, gutters, words):
            split = None
            if house is not None and band is not None and band[0] <= house <= band[1]:
                split = house
            elif band is not None:
                split = (band[0] + band[1]) / 2
            elif two_column and len(wds) >= 25 and _balanced(wds, house,
                                                             int(0.06 * len(wds))):
                split = house
            if split is None:
                out.append(body.extract_text() or "")
                continue
            x0b, top, x1b, bottom = body.bbox
            halves = [body.crop((a, top, b, bottom)).extract_text() or ""
                      for a, b in ((x0b, split), (split, x1b))]
            out.append("\n".join(halves))
    return out


# Pages that teach the student how to answer, by working an example item in
# full. They are lethal rather than merely noisy: the example is numbered and
# lettered exactly like a real item, so the assignment search can anchor item 1
# on the example, item 2 on real item 1, and so on down the form - every item
# individually plausible and every one of them mislabelled. There is no way to
# tell the example from the real thing locally, so the page goes.
SAMPLE_PAGE = re.compile(
    r"The correct answer is"
    r"|(?:Sample|Multiple-Choice|Open-Ended)\s+(?:Multiple-Choice|Open-Ended|Sample)\s+"
    r"(?:Question|Answer|Item)"
    r"|Answers to Sample Questions|Sample Item \d",
    re.IGNORECASE,
)


def clean(text: str, keep_codes: bool = False) -> list[str]:
    if SAMPLE_PAGE.search(text):
        return []
    lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or BOILERPLATE.match(line) or INSTRUCTIONS.search(line):
            continue
        if not keep_codes and (GEPA_CODE.match(line) or GEPA_ANSWER_LINE.match(line)):
            continue
        lines.append(line)
    return lines


def page_lines(pdf_path, keep_codes: bool = False) -> list[str]:
    out = []
    for page in raw_pages(pdf_path):
        out += clean(page, keep_codes)
    return out


# --------------------------------------------------------------------------
# answer keys - one parser per family
# --------------------------------------------------------------------------

def key_rows(lines: list[str], pattern: re.Pattern, group: int = 2) -> dict[int, str]:
    """Rows of `<item no> ... <answer> ...`, taken in order.

    Like the STAAR key reader, a row is only accepted when its number is the
    one expected next. Key tables repeat their headers and carry footnotes,
    and a stray match that jumps ahead truncates the key, which silently
    shortens the form.
    """
    answers: dict[int, str] = {}
    for line in lines:
        m = pattern.match(line)
        if not m:
            continue
        item_no = int(m.group(1))
        if item_no == len(answers) + 1:
            answers[item_no] = m.group(group).strip()
    return answers


# "1 B 1, 3rd B 4" (NJ ASK 2006) and "1 B 1A 06 10 Problem Solving" (GEPA):
# item number, answer, then classification columns.
PLAIN_KEY = re.compile(r"^(\d{1,2})\s+(See [Rr]ubric|rubric|[A-D])\b")
# "1. C" (NJ ASK 2007 appendix)
DOTTED_KEY = re.compile(r"^(\d{1,2})\.\s+([A-D])\s*$")
# "2 B 5.NF.B.4b": item number, answer, standard. The NJSLA maths key is the
# only one here that puts the standard LAST and the answer in the middle, and
# an item is plain multiple choice exactly when that middle is one letter.
NJSLA_MC = re.compile(r"^([A-D])\s+[0-9A-Za-z]+(?:\.[A-Za-z0-9]+){2,}\s*$")
NJSLA_MULTI = re.compile(r"^[A-H](?:,\s*(?:and\s+)?[A-H])+\b")


def njsla_math_key(pdf_path) -> tuple[dict[int, str], dict[int, str]]:
    """(item -> answer, item -> reason it is not plain MC).

    Rows are taken by their leading number rather than by matching the whole
    row, because plenty of rows are not one row: an answer that is a fraction
    prints its numerator and denominator on separate lines and leaves the
    answer column of the numbered line EMPTY ("4 5.NF.A.2"), and a table
    completion runs over four lines ("11 Row 1: Infinitely many solutions").
    A strict full-row pattern stalls on the first of these and silently
    truncates the key to whatever came before it.

    Grade 6 prints two tables, "Non-Calculator Section" then "Calculator
    Section", each numbered from 1, while the booklet numbers straight through,
    so the second table is offset by the length of the first.
    """
    answers: dict[int, str] = {}
    skipped: dict[int, str] = {}
    offset = 0
    expect = 1
    for line in page_lines(pdf_path):
        if re.match(r"^Calculator Section", line, re.I):
            offset += expect - 1
            expect = 1
            continue
        m = re.match(r"^(\d{1,2})\s+(\S.*)$", line)
        if not m:
            continue
        n = int(m.group(1))
        # Rows are taken in order, but a row whose answer is a bare fraction
        # prints the item number ALONE on its line ("2" then ", or any
        # equivalent fraction" then "5"), and a lone number is stripped as a
        # page number long before this. Allowing the cursor to step over a
        # short gap recovers the rest of the key; refusing to jump further
        # keeps a stray number in the answer text from running away with it.
        if n < expect or n > expect + 2:
            continue
        for missing in range(expect, n):
            skipped[missing + offset] = "not_plain_mc"
        rest = m.group(2).strip()
        item_no = n + offset
        expect = n + 1
        if NJSLA_MC.match(rest):
            answers[item_no] = rest[0]
        elif NJSLA_MULTI.match(rest):
            skipped[item_no] = "multi_select"
        else:
            # equation editor, table completion, "any value between ..."
            skipped[item_no] = "not_plain_mc"
    return answers, skipped


ITEM_HEAD = re.compile(r"^Item\s+(\d{1,2})\s*$")
ITEM_TYPE = re.compile(r"^Item Type:\s*(.+?)\s*$", re.I)
ITEM_KEY = re.compile(r"^Key:\s*([A-D])\s*$")


def njsla_sci_key(pdf_path) -> tuple[dict[int, str], dict[int, str]]:
    """The one key here that states each item's TYPE, so nothing is inferred.

    Only "Multiple Choice" items are taken. Technology-enhanced items also
    print a letter ("SR/AT/Paper Key: D") for the screen-reader form of the
    item, which is a different item from the one in the booklet, so matching
    any line with a letter on it would quietly import the wrong thing.
    """
    lines = page_lines(pdf_path)
    answers: dict[int, str] = {}
    skipped: dict[int, str] = {}
    current: int | None = None
    kind: str | None = None
    for line in lines:
        m = ITEM_HEAD.match(line)
        if m:
            current, kind = int(m.group(1)), None
            continue
        m = ITEM_TYPE.match(line)
        if m and current is not None:
            kind = m.group(1).strip().lower()
            if kind != "multiple choice":
                skipped[current] = ("constructed_response"
                                    if "constructed" in kind else "tech_enhanced")
            continue
        m = ITEM_KEY.match(line)
        if m and current is not None and kind == "multiple choice":
            answers[current] = m.group(1)
    return answers, skipped


def appendix_pages(pdf_path, heading: re.Pattern,
                   stop: re.Pattern | None = None) -> list[str]:
    """The run of pages from a heading to the next stop marker.

    Scanning the whole document would collect the item numbers of every other
    grade's key in the same file - the 2006 maths sampler holds grades 3 and 4,
    the 2007 key holds grades 5, 6 and 7 - so the scan is bounded at both ends.
    """
    out: list[str] = []
    started = False
    # Matched against the RAW page, not the cleaned lines: the headings name
    # the test ("Answer Key for NJ ASK 2007 Grade 5 Mathematics ...") and the
    # boilerplate filter deletes anything that names the test.
    for text in raw_pages(pdf_path):
        if not started:
            if not heading.search(text):
                continue
            started = True
        elif stop is not None and stop.search(text):
            break
        out += clean(text)
    return out


def appendix_key(pdf_path, heading: re.Pattern, pattern: re.Pattern,
                 stop: re.Pattern | None = None) -> dict[int, str]:
    return key_rows(appendix_pages(pdf_path, heading, stop), pattern)


PAIR = re.compile(r"(?:^|\s)(\d{1,2})\.\s+([A-D])(?=\s|$)")


def paired_key(lines: list[str]) -> dict[int, str]:
    """A key laid out as a wide table: "1. D 13. A" is two entries, not one.

    The 2007 sampler prints its answers in two or three side-by-side columns
    that are too close together for the gutter finder to separate, so a row
    holds several items and they arrive out of order. Order therefore cannot
    be used to validate the scan the way `key_rows` does; instead every pair is
    collected and the result is required to be a contiguous run from 1 with no
    number claiming two different answers. Prose in the scoring rubric below
    the table can also look like a pair, and that check is what catches it.
    """
    answers: dict[int, str] = {}
    for line in lines:
        for m in PAIR.finditer(line):
            n, a = int(m.group(1)), m.group(2)
            if answers.get(n, a) != a:
                return {}
            answers[n] = a
    if not answers:
        return {}
    run = max(answers)
    missing = [n for n in range(1, run + 1) if n not in answers]
    # the last item of these samplers is open-ended, so one trailing gap is
    # expected; a hole in the middle means the table was misread
    if any(n < run - 1 for n in missing):
        return {}
    return answers


def key_page(pdf_path, *required: str) -> list[str]:
    """The lines of the first page matching ALL of `required`, else empty.

    Two markers rather than one, because one is not enough to find a key in a
    file that holds two whole tests. The 2006 maths sampler is grade 3 then
    grade 4, and "Grade 4 M" matches the TABLE OF CONTENTS line ("Grade 4
    Mathematics Assessment Samples ... 48") long before it matches Appendix D.
    Scanning from there picked up Appendix A, so every grade 4 item was
    labelled with grade 3's answer - each one individually plausible, two of
    them provably wrong ("Which group of numbers is in order from least to
    greatest?" answered with the descending list). Demanding the column header
    as well pins it to the table itself.

    Not read off the rows, tempting though it is: they do carry a grade, but it
    is the grade of the STANDARD the item assesses, and a grade 4 form assesses
    grade 3 standards often enough to shred the key.
    """
    patterns = [re.compile(p, re.M) for p in required]
    for text in raw_pages(pdf_path):
        if all(p.search(text) for p in patterns):
            return clean(text)
    return []


def gepa_embedded_key(pdf_path) -> dict[int, str]:
    """The answers the GEPA booklets print beside their own items.

    Two shapes, both validated in `extract_gepa`: maths and science end the
    tracking code with the answer letter, social studies puts the letter on the
    next line. Either way the answer belongs to the most recent item number,
    which is what makes this immune to the off-by-one failure a separate key
    list has - a lost item loses only itself.
    """
    answers: dict[int, str] = {}
    poisoned: set[int] = set()
    current: int | None = None
    for line in page_lines(pdf_path, keep_codes=True):
        m = re.match(r"^(\d{1,2})\.\s+\S", line)
        if m:
            n = int(m.group(1))
            # item numbers only ever go up; a "3." inside a figure or a list
            # would otherwise reset the cursor into already-read text
            if current is None or n > current:
                current = n
            continue
        if current is None:
            continue
        m = GEPA_CODE.match(line)
        letter = line[-1] if m and line[-1] in LETTERS else None
        if letter is None:
            m = GEPA_ANSWER_LINE.match(line)
            letter = m.group(1) if m else None
        if letter is None:
            continue
        # A second annotation under the same item means an item number went
        # unread and two items are sharing a cursor. Which of the two letters
        # belongs to which is not recoverable, so both are given up.
        if current in answers:
            poisoned.add(current)
        answers[current] = letter
    return {n: a for n, a in answers.items() if n not in poisoned}


def open_ended(key: dict[int, str]) -> dict[int, str]:
    """The item numbers a letter key skips: the booklet's open-ended items.

    Worth naming rather than ignoring, for two reasons. They belong in the raw
    count - a form is 48 items whether or not 43 of them are scorable here -
    and, more usefully, listing them puts a boundary in the line stream at each
    one. Without that boundary an open-ended item's stem, which is several
    lines of prose, falls inside the PRECEDING item's region and is measured
    there as a shared passage.
    """
    if not key:
        return {}
    return {n: "constructed_response" for n in range(1, max(key) + 1) if n not in key}


def key_scheme(key: dict[int, str]) -> tuple[str | None, str]:
    """Which four letters label the options, read off the key, never assumed.

    Texas taught this one: its 2019 forms alternate A-B-C-D on odd items with
    F-G-H-J on even ones while its 2018 forms use A-D throughout, and assuming
    either would have mislabelled half a corpus without raising anything. New
    Jersey has always used A-D on every form seen here, but that is a fact to
    be confirmed per form rather than hard-coded, so a form whose key uses any
    other letter is refused instead of guessed at.
    """
    letters = sorted({a for a in key.values() if len(a) == 1 and a.isalpha()})
    if not letters:
        return None, "key has no letter answers at all"
    if not set(letters) <= set(LETTERS):
        return None, f"key uses letters {''.join(letters)}, not A-D"
    # A long key that only ever answers one letter did not parse, it collapsed.
    # Short keys are left alone: an NJSLA science unit can honestly contain two
    # multiple-choice items that both answer B.
    if len(key) >= 6 and len(letters) < 2:
        return None, f"key of {len(key)} items only ever answers {letters[0]}"
    return LETTERS, ""


# --------------------------------------------------------------------------
# test booklet - item and option location
# --------------------------------------------------------------------------

def _starts(line: str, letter: str) -> str | None:
    """The text of an option line, if this line opens with that option letter.

    NJ prints options as "A." (GEPA, NJSLA, 2007) or bare "A" (2006 NJ ASK),
    so both are accepted; requiring a following non-space keeps "A" alone on a
    line - a figure label - from counting.
    """
    m = re.match(rf"^{letter}[.)]?\s+(\S.*)$", line)
    return m.group(1) if m else None


def _option_end(lines: list[str], start: int, hi: int) -> int:
    j = start
    while j < hi:
        if re.match(r"^\d{1,2}[.)]?(\s|$)", lines[j]):
            break
        if any(_starts(lines[j], c) for c in LETTERS):
            break
        j += 1
    return j


def find_block(lines: list[str], lo: int, hi: int) -> tuple[int, int, list[str] | None] | None:
    """Locate one item's four options in [lo, hi), exactly as STAAR does it.

    Anchored on B rather than A: "A" opens plenty of ordinary sentences ("A
    student uses..."), so scanning forward from the first "A ..." swallows stem
    lines as option A, while anchoring on B and taking the last "A ..." before
    it cannot.

    Returns texts=None when the options are demonstrably present but their
    layout is unreadable, which the caller must tell apart from "not here".
    """
    l0, l1, l2, l3 = LETTERS
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
        head = [" ".join([_starts(lines[a], LETTERS[p])] + lines[a + 1:b]).strip()
                for p, (a, b) in enumerate(((i, j), (j, k), (k, m)))]
        # The last option is the only one with no next option letter to stop
        # it, so everything between it and the next item is swept into it:
        # page furniture, the tail of a figure, the directions for the
        # open-ended item that follows. It comes out as a real option with
        # rubbish welded on the end ("...reaching Egyptian cities. to the
        # open-ended question that follows. Show your work..."), which no later
        # filter can see.
        #
        # What bounds it is that an option is ONE sentence or phrase. A wrapped
        # option is cut mid-sentence, so the part read so far does not end in a
        # full stop; once it does, the option is complete and whatever follows
        # belongs to something else. Options that never take a full stop ("a
        # circle", "$12,000") are bounded instead by shape - a caption-like
        # line is not a continuation of a phrase - and by a generous cap.
        budget = 2 * max(len(h.split()) for h in head) + 4
        last = [_starts(lines[m], l3)]
        for x in range(m + 1, end):
            if re.search(r"[.?!]\s*$", " ".join(last)):
                break
            # A short line that starts lower-case is the tail of the wrapped
            # option ("...produced by the" / "government, and citizens
            # receive"), not a caption; only the capitalised ones are page
            # furniture ("TICS - PART 1").
            if looks_like_caption(lines[x]) and not lines[x][:1].islower():
                break
            if len(" ".join(last + [lines[x]]).split()) > budget:
                break
            last.append(lines[x])
        return i, end, head + [" ".join(last).strip()]

    for i in range(lo, hi):
        run = split_run_together(lines[i])
        if run:
            return i, i + 1, run

    where = {c: next((x for x in range(lo, hi) if _starts(lines[x], c)), None)
             for c in LETTERS}
    if all(v is not None for v in where.values()):
        return min(where.values()), hi, None
    return None


RUN_TOGETHER = re.compile(r"(?:^|\s)([A-D])[.)]?\s+")


def split_run_together(line: str) -> list[str] | None:
    """Recover options printed on one line, where the letters are delimiters.

    Only accepted when all four appear exactly once and in order: option text
    itself contains bare capitals ("Boron, B"), so a looser rule invents
    options out of ordinary prose.
    """
    hits = [(m.start(1), m.group(1)) for m in RUN_TOGETHER.finditer(line)]
    picked, want = [], list(LETTERS)
    for at, letter in hits:
        if want and letter == want[0]:
            picked.append(at)
            want.pop(0)
    if want or len(picked) != 4:
        return None
    bounds = picked + [len(line)]
    return [line[bounds[p] + 1:bounds[p + 1]].strip(". ") for p in range(4)]


LOOKAHEAD = 140
STIMULUS_WORDS = 45
STIMULUS_CARRY = 4    # items a stimulus is assumed to serve, if it does not say
MAX_CANDIDATES = 120
PROSE_LINE_WORDS = 7

# These booklets announce a shared stimulus in so many words, and usually say
# how far it reaches: "Use the maps below to answer the next three questions",
# "Use the graph below to answer questions 37 through 39".
COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
ANNOUNCE_WORD = re.compile(
    r"answer\s+the\s+(?:next\s+|following\s+)?(\w+)\s+questions", re.I)
ANNOUNCE_RANGE = re.compile(
    r"questions?\s+(\d{1,2})\s*(?:through|to|-|\u2013|and)\s*(\d{1,2})", re.I)


def announced_reach(lines: list[str]) -> int:
    """How many items a stimulus says it serves, or 0 if it does not say."""
    text = " ".join(lines)
    m = ANNOUNCE_RANGE.search(text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if 0 < hi - lo < 10:
            return hi - lo + 1
    m = ANNOUNCE_WORD.search(text)
    if m:
        return COUNT_WORDS.get(m.group(1).lower(), STIMULUS_CARRY)
    return 0


def _opener(item_no: int) -> re.Pattern:
    # "7. One apple...", "7 One apple..." or a bare "7"; never "70 apples"
    return re.compile(rf"^{item_no}\.?(?:$|\s+(?=[A-Z\u201c\u2018\"'(]))")


def opener_lines(lines: list[str], item_no: int) -> list[int]:
    pattern = _opener(item_no)
    out = []
    for i, line in enumerate(lines):
        if not pattern.match(line):
            continue
        if line.rstrip(".").isdigit():
            near = [lines[x].rstrip(".") for x in (i - 1, i + 1) if 0 <= x < len(lines)]
            if any(n.isdigit() and abs(int(n) - item_no) == 1 for n in near):
                continue
        out.append(i)
    return out


def prose_words(lines: list[str]) -> int:
    """Words in lines long enough to be sentences rather than table cells."""
    return sum(len(w) for w in (line.split() for line in lines)
               if len(w) >= PROSE_LINE_WORDS)


def item_starts(lines: list[str], wanted: list[int],
                has_options: set[int]) -> tuple[dict[int, int] | None, str]:  # noqa: C901
    """Find the line that opens each item, choosing the whole assignment at once.

    Anchored on ITEM NUMBERS, not option letters: an item whose options are
    unreadable would otherwise be skipped, the scan would match the NEXT item's
    options, and every label after it would be off by one - silently, since
    each item still looks fine on its own.

    Numbers are ambiguous in both directions (figures are full of small
    integers; sampler booklets number their directions), so rather than
    committing item by item this scores every strictly increasing assignment by
    how many items end up with their options where they should be, and takes
    the best. One bad item then costs one item instead of derailing the rest.
    """
    cands: dict[int, list[int]] = {}
    good: dict[int, set[int]] = {}
    lost: list[int] = []
    for item_no in list(wanted):
        cands[item_no] = opener_lines(lines, item_no)[:MAX_CANDIDATES]
        if not cands[item_no]:
            # An item whose number never appears is unlocatable, but it cannot
            # shift anything else: every other item is found by ITS OWN number,
            # not by counting on from this one. So it is given up on its own
            # rather than taking the form with it - which is the whole reason
            # for anchoring on numbers instead of on option letters.
            lost.append(item_no)
            wanted.remove(item_no)
            del cands[item_no]
            continue
        good[item_no] = set()
        if item_no not in has_options:
            continue
        for c in cands[item_no]:
            block = find_block(lines, c, min(c + LOOKAHEAD, len(lines)))
            if block and not any(c < other < block[0] for other in cands[item_no]):
                good[item_no].add(c)

    order = sorted(wanted)
    if not order:
        return None, "no item number appears in the booklet at all"

    # An item may also be LEFT OUT of the assignment. Demanding that all of
    # them place in increasing order fails whole forms over one item that the
    # column split emits out of reading order - the 2007 grade 7 booklet has
    # item 11 ahead of item 9 - and, again, leaving an item unplaced cannot
    # shift its neighbours, because each of them is found by its own number.
    # Placing an item is preferred to skipping it, and placing one whose
    # options are where they should be counts for more than placing one whose
    # options are not, which is what stops the search from anchoring items on
    # stray numbers in tables.
    memo: dict[tuple[int, int], tuple[int, int | None]] = {}

    def solve(idx: int, prev: int) -> tuple[int, int | None]:
        if idx == len(order):
            return 0, None
        hit = memo.get((idx, prev))
        if hit is not None:
            return hit
        best, choice = solve(idx + 1, prev)[0], None
        for c in cands[order[idx]]:         # ascending, so ties keep the
            if c <= prev:                   # earliest line: table cells sit
                continue                    # after the item they follow
            score = solve(idx + 1, c)[0] + (4 if c in good[order[idx]] else 1)
            if score > best:
                best, choice = score, c
        memo[(idx, prev)] = (best, choice)
        return best, choice

    placed: dict[int, int] = {}
    prev = -1
    for idx, item_no in enumerate(order):
        _, choice = solve(idx, prev)
        if choice is not None:
            placed[item_no] = choice
            prev = choice
        else:
            lost.append(item_no)
    if not placed:
        return None, "no item number appears in the booklet at all"
    return placed, ("" if not lost else
                    f"{len(lost)} items unplaceable: {sorted(lost)}")


def parse_form(lines: list[str], key: dict[int, str],
               skipped: dict[int, str]) -> tuple[list[dict] | None, str]:
    """Split a booklet into items, or refuse the whole form.

    The key is the authority on how many items there are and which are plain
    multiple choice; this only has to agree with it. A mismatch returns None
    for the whole form rather than a best effort, because one relabelled form
    is worse than one missing form.
    """
    scheme, why = key_scheme(key)
    if scheme is None:
        return None, why

    every = sorted(set(key) | set(skipped))
    wanted = list(every)
    starts, why = item_starts(lines, wanted, set(key))
    if starts is None:
        return None, why

    found = sorted(starts)
    bounds = {n: (starts[n], starts.get(after, len(lines)))
              for n, after in zip(found, found[1:] + [None])}
    items = [{"item_no": n, "skip": "parse"} for n in every if n not in starts]

    # Texas latches this: once a passage appears, every later item is under it.
    # That is right for a reading booklet and wrong here. No ELA form is
    # extracted - they are all passage-based and correctly excluded wholesale -
    # so a stimulus in these booklets is a map or a graph serving two to four
    # items, not a passage governing the rest of the paper. Latched, the first
    # one condemns the form: the GEPA social studies booklet lost items 23-47
    # to a political cartoon printed before item 16. So it is a countdown, and
    # the booklet is taken at its word about how far a stimulus reaches.
    #
    # Nor is prose before item 1 counted, as Texas counts it: the only thing in
    # front of item 1 in these booklets is the directions block.
    reach = 0
    for item_no in sorted(starts):
        lo, hi = bounds[item_no]
        # A skipped item's own region is not measured. Open-ended items are
        # prose by definition, and reading each one as a shared passage would
        # re-arm the countdown at every one of them.
        if item_no in skipped:
            items.append({"item_no": item_no, "skip": skipped[item_no]})
            reach = max(0, reach - 1)
            continue
        block = find_block(lines, lo, hi)
        start, tail, texts = block if block else (hi, hi, None)
        stem_lines = [s for s in lines[lo:start] if s]
        if stem_lines:
            stem_lines[0] = re.sub(rf"^{item_no}\s*\.?\s*", "", stem_lines[0]).strip()
            stem_lines = [s for s in stem_lines if s]
        items.append({
            "item_no": item_no,
            "stem_lines": stem_lines,
            "options": texts,
            "answer": key[item_no],
            "skip": None,
            "under_stimulus": reach > 0,
        })
        reach = max(0, reach - 1)
        after = lines[tail:hi]
        said = announced_reach(after)
        if said:
            reach = said
        elif prose_words(after) >= STIMULUS_WORDS:
            reach = max(reach, STIMULUS_CARRY)
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
    r"|food web|periodic table|timeline|pedigree|rock strata"
    r"|these\s+(?:polygons|shapes|solids|objects|figures|angles|lines|points"
    r"|triangles|rectangles|drawings|nets|prisms|graphs))\b",
    re.IGNORECASE,
)

FIGURE_GLYPHS = re.compile(r"[\u2190-\u21FF\u2500-\u27BF\u2B00-\u2BFF\uFFFD\u25A0-\u25FF]")
BAD_ENCODING = re.compile(r"\(cid:|\uFFFD")
STRAY_LABEL = re.compile(r"(?<=\S)\s[A-D]\.(?:\s|$)")

# Real two- and three-letter words. Anything else that short at the end of an
# option is a word the column crop cut in half ("...keep the te").
SHORT_WORDS = {
    "a", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it",
    "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we", "and",
    "air", "all", "any", "are", "but", "can", "day", "dry", "end", "far", "few",
    "for", "gas", "get", "had", "has", "her", "him", "his", "hot", "ice", "its",
    "law", "led", "let", "low", "man", "may", "men", "new", "not", "now", "off",
    "oil", "old", "one", "our", "out", "own", "per", "put", "raw", "red", "run",
    "sea", "see", "set", "sit", "six", "sky", "son", "sun", "ten", "the", "them",
    "too", "top", "two", "use", "war", "was", "way", "wet", "who", "why", "you",
    "cm", "mm", "km", "ft", "yd", "lb", "oz", "kg", "ml", "mi", "hr", "pt", "qt",
}


def cut_off_word(option: str) -> bool:
    """Does this option stop in the middle of a word?

    A column crop landing a few points off cuts every line on the page in half
    ("Warm winds from the land keep the te"), and the halves are individually
    well-formed English until the very last token. Single letters are left
    alone: they are algebra ("20 - d"), not wreckage.
    """
    tokens = option.split()
    if not tokens:
        return False
    tail = tokens[-1].strip(".,;:?!\u2019\"')")
    return (tail.isalpha() and tail.islower() and 2 <= len(tail) <= 3
            and tail not in SHORT_WORDS)
NO_VOWEL = re.compile(r"\b(?![A-Z]{2,}\b)[A-Za-z]{4,}\b")

# Two-dimensional maths flattened into a line does not fail loudly - it makes a
# plausible string that means something else. The 2006 NJ ASK booklets are the
# worst offenders: their operators are Type-3 glyphs that come out as
# "(cid:1)", so "145 + 281" reads "145 (cid:1) 281" and BAD_ENCODING takes it.
FLATTENED_MATH = re.compile(
    r"\s\.\s"                       # a fraction bar left standing alone
    r"|\)\s*\d"
    r"|\b(?:cm|mm|km|in|ft|yd|m|s)\.?\d\b"
    r"|(?:\b\d\b ){3,}"
    r"|[\u00d7\u00f7+\u2212]\s*$"
    r"|\u22c5"
    r"|\S {2,}\S"
)

SHARED = re.compile(
    r"\b(?:the\s+)?(?:passage|selection|selections|article|articles|poem|poems|story|stories"
    r"|excerpt|essay|interview|memoir|play|speech\s+above|text\s+box"
    r"|paragraph|paragraphs|stanza|stanzas|line\s+\d+|lines\s+\d+"
    r"|the\s+author|the\s+narrator|the\s+speaker|the\s+poet|the\s+writer"
    r"|both\s+selections|these\s+selections"
    # NJSLA-Science hangs every item off a shared scenario it calls a
    # "phenomenon". Its items point back at it with a definite article and no
    # antecedent - "based on the data", "which observation", "student 1" - and
    # those phrases are the only trace of the stimulus left in the text.
    r"|phenomenon|the\s+investigation|the\s+experiment|the\s+simulation"
    r"|the\s+data|the\s+results|the\s+readings|the\s+observations"
    r"|the\s+student|the\s+students|the\s+scientist|the\s+class"
    r"|trial\s*\d|student\s*\d|group\s*[A-Z]\b|sample\s*\d|day\s*\d)\b",
    re.IGNORECASE,
)

STEM_END = re.compile(r"[?.\u2014\u2013:-]\s*$")


def well_formed_stem(stem: str) -> bool:
    """A stem is a question, or a sentence the options finish.

    STAAR stems always close with punctuation, so Texas can just require it.
    New Jersey writes a lot of sentence-completion items whose stem ends on a
    bare function word - "...contributed to the decline of feudalism EXCEPT
    the", "...can affect the outcomes of elections by" - and demanding
    punctuation throws every one of them away. The point of the check is to
    reject fragments torn out of figures, so the alternative asks for what a
    fragment does not have: a capitalised start, real length, and a lowercase
    word at the break where the option continues.
    """
    words = stem.split()
    if len(words) < 5:
        return False
    if STEM_END.search(stem):
        return True
    return (stem[:1].isupper() and len(words) >= 8
            and words[-1].isalpha() and words[-1].islower())

DROP_REASONS = (
    "not_plain_mc", "multi_select", "tech_enhanced", "constructed_response",
    "parse", "shared_stimulus", "image", "debris", "degenerate",
)


# "this cycle", "these boxes", "this Dalmatian": a demonstrative whose noun was
# never introduced. Its antecedent is a picture, and the picture is not here.
DEICTIC = re.compile(r"\b(?:this|these|those)\s+([A-Za-z]{3,})\b", re.I)
# Demonstratives that point back at the sentence rather than at a figure.
ABSTRACT = {
    "action", "actions", "situation", "statement", "statements", "information",
    "question", "questions", "problem", "change", "changes", "reason", "event",
    "events", "kind", "type", "case", "fact", "facts", "idea", "ideas", "way",
    "example", "examples", "difference", "differences", "amount", "number",
}


def dangling_reference(stem: str) -> bool:
    """Does the stem point at something it never mentions?"""
    words = [w.strip(".,;:?!\u2019's").lower() for w in stem.split()]
    for m in DEICTIC.finditer(stem):
        noun = m.group(1).lower()
        if noun in ABSTRACT:
            continue
        before = words[:len(stem[:m.start()].split())]
        stems = {noun, noun.rstrip("s"), noun + "s"}
        if not stems & set(before):
            return True
    return False


def looks_like_caption(line: str) -> bool:
    """A figure caption or axis label rather than prose."""
    if line.startswith(("\u2022", "-", "*")):
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
    if item["skip"]:
        return item["skip"]
    if item["options"] is None:
        return "parse"
    stem_lines = item["stem_lines"]
    stem = " ".join(stem_lines).strip()
    options = [o.strip() for o in item["options"]]
    blob = stem + " \n " + " \n ".join(options)

    if not stem or len(options) != 4 or not all(options):
        return "parse"
    if not well_formed_stem(stem):
        return "parse"

    if item["under_stimulus"] or SHARED.search(blob):
        return "shared_stimulus"
    if VISUAL.search(blob) or FIGURE_GLYPHS.search(blob):
        return "image"
    if dangling_reference(stem):
        return "image"
    if BAD_ENCODING.search(blob) or FLATTENED_MATH.search(blob):
        return "debris"
    # An option label stranded in the middle of a sentence ("Ricky lives 1,000
    # meters from his best B. friend's house") is the neighbouring column
    # bleeding in, not text.
    if any(STRAY_LABEL.search(part) for part in [stem] + options):
        return "debris"
    if any(cut_off_word(o) for o in options):
        return "debris"
    if any(not re.search(r"[aeiouyAEIOUY]", w) for w in NO_VOWEL.findall(blob)):
        return "debris"
    if any(looks_like_caption(line) for line in stem_lines[:-1]):
        return "debris"

    if len(set(o.lower() for o in options)) != 4:
        return "degenerate"
    if any(len(re.sub(r"\W", "", o)) <= 2 for o in options):
        return "degenerate"
    return ""


# --------------------------------------------------------------------------
# forms
# --------------------------------------------------------------------------

def harvest(label: str, test_pdf, key: dict[int, str], skipped: dict[int, str],
            subject: str, grade: int, year: int, stats: Counter) -> list[dict]:
    """Parse one form and keep the items that survive the filters."""
    if not test_pdf.exists() or not key:
        stats["form_no_key"] += 1
        print(f"  {label}: DROPPED, no test booklet or no parsable key")
        return []

    items, why = parse_form(page_lines(test_pdf), key, skipped)
    if items is None:
        stats["form_misaligned"] += 1
        stats["dropped_by_misaligned_form"] += len(key) + len(skipped)
        print(f"  {label}: DROPPED whole form, {why}")
        return []

    stats["raw"] += len(items)
    kept, reasons = [], Counter()
    for item in items:
        reason = classify(item)
        if reason:
            reasons[reason] += 1
            stats[reason] += 1
            continue
        kept.append({
            "question": tidy(" ".join(item["stem_lines"])),
            "choices": [tidy(o) for o in item["options"]],
            "gold_idx": LETTERS.index(item["answer"]),
            "hint": None,
            "source": "nj",
            "subject": subject,
            "grade": grade,
            "year": year,
            "item_no": item["item_no"],
            "form": label,
        })
    stats["kept"] += len(kept)
    detail = " ".join(f"{r}={reasons[r]}" for r in DROP_REASONS if reasons[r])
    print(f"  {label}: {len(items)} items, {detail or 'no drops'} -> kept {len(kept)}")
    return kept


def extract_njsla_math(stats: Counter) -> list[dict]:
    out = []
    for grade in GRADES:
        test = PDF_DIR / f"math_g{grade}_test.pdf"
        key_pdf = PDF_DIR / f"math_g{grade}_key.pdf"
        if not key_pdf.exists():
            stats["form_no_key"] += 1
            print(f"  njsla-math-g{grade}: DROPPED, no answer key")
            continue
        key, skipped = njsla_math_key(key_pdf)
        out += harvest(f"njsla-math-g{grade}", test, key, skipped,
                       "math", grade, 2026, stats)
    return out


def extract_njsla_science(stats: Counter) -> list[dict]:
    out = []
    for grade in (5, 8):
        for unit in (1, 2, 3):
            test = PDF_DIR / f"sci_g{grade}_u{unit}_test.pdf"
            key_pdf = PDF_DIR / f"sci_g{grade}_u{unit}_key.pdf"
            if not key_pdf.exists():
                continue
            key, skipped = njsla_sci_key(key_pdf)
            out += harvest(f"njsla-sci-g{grade}-u{unit}", test, key, skipped,
                           "science", grade, 2026, stats)
    return out


def extract_gepa(stats: Counter, verbose: bool = True) -> list[dict]:
    """GEPA 2000-2001 grade 8, the only NJ social studies items with answers.

    Maths and science carry BOTH an official scoring guide and the letters the
    booklet prints beside its own items; the two must agree on every item or
    the form is refused. Social studies has only the embedded letters, and is
    accepted because that agreement establishes the mechanism - see the module
    docstring for why this is not a rule-4 violation.
    """
    out = []
    official = {
        "math": (PDF_DIR / "gepa_math_scoring_guide.pdf", PLAIN_KEY),
        "science": (PDF_DIR / "gepa_science_score_guide.pdf", PLAIN_KEY),
    }
    for subject, stem in (("math", "gepa_test_book_math.pdf"),
                          ("science", "gepa_test_book_sci.pdf"),
                          ("social_studies", "gepa_test_book_ss.pdf")):
        test = PDF_DIR / stem
        if not test.exists():
            stats["form_no_key"] += 1
            print(f"  gepa-g8-{subject}: DROPPED, booklet missing")
            continue
        embedded = gepa_embedded_key(test)

        if subject in official:
            key_pdf, pattern = official[subject]
            printed = key_rows(page_lines(key_pdf), pattern)
            letters = {n: a for n, a in printed.items() if a in LETTERS}
            shared = [n for n in letters if n in embedded]
            wrong = [n for n in shared if embedded[n] != letters[n]]
            if not letters or wrong:
                stats["form_misaligned"] += 1
                print(f"  gepa-g8-{subject}: DROPPED whole form, booklet answers "
                      f"contradict the official key on {len(wrong)}/{len(shared)} items")
                continue
            if verbose:
                print(f"  gepa-g8-{subject}: booklet answers agree with the official "
                      f"scoring guide on {len(shared)}/{len(letters)} items, 0 conflicts")
            key = letters
        else:
            key = {n: a for n, a in embedded.items() if a in LETTERS}
        out += harvest(f"gepa-g8-{subject}", test, key, open_ended(key),
                       subject, 8, 2001, stats)
    return out


def extract_njask06(stats: Counter) -> list[dict]:
    """NJ ASK 2006 released samples: maths grades 3-4 in one file, science grade 4.

    The maths file holds two whole tests, so each grade's booklet section and
    each grade's key appendix have to be isolated from the other's.
    """
    out = []
    maths = PDF_DIR / "njask06_math.pdf"
    if maths.exists():
        pages = raw_pages(maths)
        # The booklet sections have to stop at their own appendices: a scoring
        # key row ("1 B 1, 3rd B 4") opens with a number and a capital, so left
        # in the booklet it is a perfectly good candidate opener for item 1 and
        # the assignment search happily anchors the whole form inside the key.
        starts = {g: page_index(pages, rf"GRADE {g}") for g in (3, 4)}
        ends = {g: page_index(pages, r"^Item Correct|^#\s*Answer",
                              after=starts[g]) for g in (3, 4)}
        keys = {g: key_rows(key_page(maths, r"^Item #?\s*Correct",
                                     rf"^Grade {g} Mathematics"), PLAIN_KEY)
                for g in (3, 4)}
        if keys[3] and keys[3] == keys[4]:
            print("  njask06-math: DROPPED both grades, they parsed the same key")
            keys = {3: {}, 4: {}}
        for grade in (3, 4):
            a, b = starts[grade], ends[grade]
            if a is None or not keys[grade]:
                continue
            key = {n: v for n, v in keys[grade].items() if v in LETTERS}
            lines = []
            for text in pages[a:b if b is not None else len(pages)]:
                lines += clean(text)
            out += harvest_lines(f"njask06-math-g{grade}", lines, key,
                                 open_ended(key), "math", grade, 2006, stats)

    sci = PDF_DIR / "njask06_sci_g4.pdf"
    if sci.exists():
        pages = raw_pages(sci)
        key = key_rows(key_page(sci, r"^Item #?\s*Correct", r"^Grade 4 Science"), PLAIN_KEY)
        key = {n: v for n, v in key.items() if v in LETTERS}
        end = page_index(pages, r"^Item Correct|^#\s*Answer", after=2)
        lines = []
        for text in pages[:end if end is not None else len(pages)]:
            lines += clean(text)
        out += harvest_lines("njask06-sci-g4", lines, key, open_ended(key),
                             "science", 4, 2006, stats)
    return out


def page_index(pages: list[str], pattern: str, after: int | None = None) -> int | None:
    start = 0 if after is None else after + 1
    for i in range(start, len(pages)):
        if re.search(pattern, pages[i], re.M):
            return i
    return None


def extract_njask07(stats: Counter) -> list[dict]:
    """NJ ASK 2007 samplers, grades 5-7. One shared key file for all three.

    Each booklet is Language Arts THEN Mathematics, and both sections number
    from 1, so the whole booklet handed to the parser anchors the maths key on
    the reading items - which then look like a form where every item sits under
    a passage, because they do. Only the maths half is passed through.
    """
    out = []
    key_pdf = PDF_DIR / "njask07_key.pdf"
    if not key_pdf.exists():
        return out
    for grade in (5, 6, 7):
        test = PDF_DIR / f"njask07_g{grade}.pdf"
        if not test.exists():
            continue
        key = paired_key(appendix_pages(
            key_pdf,
            re.compile(rf"Answer Key for NJ ASK 2007 Grade {grade} Mathematics"),
            re.compile(r"Open-Ended Scoring Rubric"),
        ))
        pages = raw_pages(test)
        # the section's own banner is cropped away as a running head, so the
        # marker is the one line of its directions that names the subject
        start = page_index(pages, r"taking the Mathematics")
        lines = []
        for text in pages[start or 0:]:
            lines += clean(text)
        out += harvest_lines(f"njask07-math-g{grade}", lines, key,
                             open_ended(key), "math", grade, 2007, stats)
    return out


def harvest_lines(label: str, lines: list[str], key: dict[int, str],
                  skipped: dict[int, str], subject: str, grade: int,
                  year: int, stats: Counter) -> list[dict]:
    """`harvest` for a form that is a slice of a bigger file."""
    if not lines or not key:
        stats["form_no_key"] += 1
        print(f"  {label}: DROPPED, no parsable key")
        return []
    items, why = parse_form(lines, key, skipped)
    if items is None:
        stats["form_misaligned"] += 1
        stats["dropped_by_misaligned_form"] += len(key)
        print(f"  {label}: DROPPED whole form, {why}")
        return []
    stats["raw"] += len(items)
    kept, reasons = [], Counter()
    for item in items:
        reason = classify(item)
        if reason:
            reasons[reason] += 1
            stats[reason] += 1
            continue
        kept.append({
            "question": tidy(" ".join(item["stem_lines"])),
            "choices": [tidy(o) for o in item["options"]],
            "gold_idx": LETTERS.index(item["answer"]),
            "hint": None,
            "source": "nj",
            "subject": subject,
            "grade": grade,
            "year": year,
            "item_no": item["item_no"],
            "form": label,
        })
    stats["kept"] += len(kept)
    detail = " ".join(f"{r}={reasons[r]}" for r in DROP_REASONS if reasons[r])
    print(f"  {label}: {len(items)} items, {detail or 'no drops'} -> kept {len(kept)}")
    return kept


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Extract self-contained MC items from released New Jersey "
                    "test PDFs. OUTPUT IS STATE-COPYRIGHTED AND GITIGNORED - "
                    "do not commit it.")
    ap.add_argument("--download", action="store_true", help="fetch the PDFs first")
    ap.add_argument("--out", default=str(NJ_DIR / "nj_items.jsonl"))
    ap.add_argument("--show", type=int, default=0, help="print N kept items in full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    NJ_DIR.mkdir(parents=True, exist_ok=True)
    if args.download:
        print("downloading ...")
        report = download_all()
        missing = [n for n, s in report.items() if s == "MISSING"]
        got = len(report) - len(missing)
        print(f"  {got}/{len(report)} files present")
        for name in sorted(missing):
            print(f"    MISSING: {name}")

    stats = Counter()
    items = []
    print("\nextracting ...")
    items += extract_njsla_math(stats)
    items += extract_njsla_science(stats)
    items += extract_gepa(stats)
    items += extract_njask06(stats)
    items += extract_njask07(stats)

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

    if items:
        golds = [it["choices"][it["gold_idx"]].split() for it in items]
        lengths = sorted(len(g) for g in golds)
        median = lengths[len(lengths) // 2]
        single = sum(1 for n in lengths if n == 1) / len(lengths)
        print(f"\ngold answer length: median {median} words, "
              f"{single:.0%} single-word  (STAAR 5w/13%, OpenBookQA 2w/31%)")

    if args.show:
        import random

        sample = list(items)
        random.Random(args.seed).shuffle(sample)
        seen = Counter()
        picked = []
        per = max(1, args.show // max(1, len(by_subject))) + 2
        for it in sample:
            if seen[it["subject"]] < per:
                seen[it["subject"]] += 1
                picked.append(it)
            if len(picked) == args.show:
                break
        for n, it in enumerate(picked, 1):
            print(f"\n--- {n}. {it['form']} grade {it['grade']} item {it['item_no']} ---")
            print(it["question"])
            for i, choice in enumerate(it["choices"]):
                mark = " <-- GOLD" if i == it["gold_idx"] else "       "
                print(f"  {LETTERS[i]}{mark} {choice}")


if __name__ == "__main__":
    main()
