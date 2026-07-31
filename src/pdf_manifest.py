"""Record which state-test PDFs were downloaded, so they can be deleted.

The PDFs are ~600MB of state-copyright material that must never enter git. Once
items have been extracted there is no reason to keep them on disk, but there IS
a reason to keep a record: without one, nobody can tell which forms a dataset
came from, or re-download exactly the same set.

This writes that record to docs/state_test_sources.md - filename, size and a
sha256 per file, so a later re-download can be checked against it.

    python src/pdf_manifest.py            # write the manifest
    python src/pdf_manifest.py --delete   # write it, then remove the PDFs
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import paths

OUT = paths.ROOT / "docs" / "state_test_sources.md"

# where each state's PDFs live, and how to get them back
SOURCES = {
    "tx": (paths.DATA / "staar" / "pdf", "Texas STAAR",
           "tea.texas.gov released test questions (2018-2019 only; earlier years "
           "removed, 2021+ has no answer keys, 2022+ is online-only)",
           "python src/staar_extract.py"),
    "pa": (paths.DATA / "state_tests" / "pdf_pa", "Pennsylvania PSSA",
           "pa.gov item and scoring samplers; answer key, DOK, p-values and "
           "per-option rationales are inline with each item",
           "python src/extract_pa.py"),
    "ca": (paths.DATA / "state_tests" / "pdf_ca", "California CST",
           "released test questions via the Wayback Machine; removed from "
           "cde.ca.gov and the live site is behind a captcha",
           "python src/extract_ca.py"),
    "ma": (paths.DATA / "state_tests" / "pdf_ma", "Massachusetts MCAS",
           "doe.mass.edu released items with answer keys",
           "python src/extract_ma.py"),
    "nj": (paths.DATA / "state_tests" / "pdf_nj", "New Jersey",
           "NJSLA / PARCC / NJ ASK released items",
           "python src/extract_nj.py"),
}


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true",
                    help="remove the PDF directories after writing the manifest")
    args = ap.parse_args()

    header = ["# State test PDFs that were downloaded", "",
              "The PDFs themselves are **not kept**: roughly 650MB of state-copyright",
              "material, of no further use once items are extracted, and they must never",
              "enter this public repo. This is the record of what was fetched.",
              "",
              "Re-download with the command listed for each state. Checksums are the",
              "first 16 hex characters of the sha256, enough to spot a changed file.",
              ""]
    lines: list[str] = []

    total_files = total_bytes = 0
    for key, (d, title, note, cmd) in SOURCES.items():
        pdfs = sorted(d.glob("*.pdf")) if d.exists() else []
        items_file = (paths.DATA / "state_tests" / f"{key}_items.jsonl")
        if key == "tx":
            items_file = paths.DATA / "staar" / "staar_items.jsonl"
        n_items = sum(1 for _ in open(items_file)) if items_file.exists() else 0
        size = sum(p.stat().st_size for p in pdfs)
        total_files += len(pdfs)
        total_bytes += size

        lines += [f"## {title} (`{key}`)", "",
                  f"{note}", "",
                  f"- **{len(pdfs)} PDFs**, {size / 1e6:.0f} MB, yielding "
                  f"**{n_items} extracted items**",
                  f"- rebuild: `{cmd}`", ""]
        if pdfs:
            lines += ["| file | MB | sha256[:16] |", "|---|---|---|"]
            lines += [f"| `{p.name}` | {p.stat().st_size / 1e6:.1f} | `{sha256(p)}` |"
                      for p in pdfs]
            lines.append("")

    header += [f"**Total: {total_files} PDFs, {total_bytes / 1e6:.0f} MB.**", ""]
    OUT.parent.mkdir(exist_ok=True)

    # Refuse to overwrite a fuller record with an emptier one. Re-running this
    # after some PDFs have already been deleted would silently drop their rows,
    # which is the one thing the manifest exists to prevent.
    if OUT.exists():
        had = OUT.read_text().count("\n| `")
        if had > total_files:
            raise SystemExit(
                f"{OUT} already lists {had} PDFs but only {total_files} are on disk. "
                f"Refusing to overwrite the record of files that have been deleted. "
                f"Write elsewhere with --out if you really want a partial manifest.")
    OUT.write_text("\n".join(header + lines))
    print(f"wrote {OUT}  ({total_files} PDFs, {total_bytes / 1e6:.0f} MB catalogued)")

    if args.delete:
        for key, (d, *_rest) in SOURCES.items():
            if d.exists():
                shutil.rmtree(d)
                print(f"  removed {d}")
        print("PDFs deleted; the manifest is the record.")


if __name__ == "__main__":
    main()
