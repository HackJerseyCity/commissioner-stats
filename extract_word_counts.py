#!/usr/bin/env python3
"""
Extract commissioner word counts from Hudson County Board of County Commissioners
meeting minutes PDFs.

Subcommands:
  ocr    — OCR PDFs in ./pdfs/2025/ and cache transcripts under ./ocr/
  parse  — Read the OCR cache and emit commissioner_word_counts.json
  all    — Run `ocr` (skipping already-cached files) then `parse`

Requires: pymupdf (fitz), tesseract
"""

import argparse
import fitz
import subprocess
import re
import json
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

ROOT = Path(__file__).parent
PDF_DIR = ROOT / "pdfs" / "2025"
OCR_DIR = ROOT / "ocr"
OUTPUT_JSON = ROOT / "commissioner_word_counts.json"

COMMISSIONER_PATTERNS = {
    r"CHAIRMAN\s+ROMANO": "Anthony Romano",
    r"COMMISSIONER\s+APONTE[\s-]*LIPSKI": "Yraida Aponte-Lipski",
    r"COMMISSIONER\s+BASELICE": "Robert Baselice",
    r"COMMISSIONER\s+CEDE[NÑ]O": "Fanny Cedeno",
    r"COMMISSIONER\s+CIFELLI": "Albert Cifelli",
    r"COMMISSIONER\s+KOPACZ": "Kenneth Kopacz",
    r"COMMISSIONER\s+O[’'']?DEA": "William O'Dea",
    r"COMMISSIONER\s+RODRIGUEZ": "Caridad Rodriguez",
    r"COMMISSIONER\s+WALKER": "Jerry Walker",
}

SPEAKER_RE = re.compile(
    r"(" + "|".join(COMMISSIONER_PATTERNS.keys()) + r")\s*:",
    re.IGNORECASE,
)

ANY_SPEAKER_RE = re.compile(
    r"(?:"
    + "|".join(COMMISSIONER_PATTERNS.keys())
    + r"|THE\s+CLERK|MR\.\s+\w+|MS\.\s+\w+|MRS\.\s+\w+|DR\.\s+\w+"
    + r"|UNIDENTIFIED|A\s+VOICE|AUDIENCE|THE\s+COURT"
    + r")\s*:",
    re.IGNORECASE,
)


# ---------- OCR ----------

def _ocr_one_page(pdf_path: str, page_num: int) -> str:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    text = page.get_text()
    if text.strip():
        doc.close()
        return text
    pix = page.get_pixmap(dpi=250)
    img_data = pix.tobytes("png")
    doc.close()
    result = subprocess.run(
        ["tesseract", "stdin", "stdout"],
        input=img_data,
        capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


def _ocr_pdf_to_text(pdf_path: str) -> str:
    """OCR every page of a PDF in parallel threads and return concatenated text."""
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()
    with ThreadPoolExecutor(max_workers=4) as tex:
        pages = list(tex.map(lambda i: _ocr_one_page(pdf_path, i), range(num_pages)))
    return "\n".join(pages)


def _ocr_and_cache(pdf_path_str: str) -> tuple[str, int]:
    """OCR a PDF and write the transcript to OCR_DIR. Returns (filename, char_count)."""
    pdf_path = Path(pdf_path_str)
    cache_path = OCR_DIR / (pdf_path.stem + ".txt")
    text = _ocr_pdf_to_text(str(pdf_path))
    cache_path.write_text(text, encoding="utf-8")
    return (pdf_path.name, len(text))


def cmd_ocr(args) -> None:
    if not PDF_DIR.exists():
        print(f"Error: {PDF_DIR} not found", file=sys.stderr)
        sys.exit(1)
    OCR_DIR.mkdir(exist_ok=True)

    minutes_files = sorted(PDF_DIR.rglob("*_Minutes.pdf"))
    if not minutes_files:
        print(f"No *_Minutes.pdf files found under {PDF_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(minutes_files)} minutes files in {PDF_DIR}", file=sys.stderr)

    if args.force:
        for p in minutes_files:
            (OCR_DIR / (p.stem + ".txt")).unlink(missing_ok=True)

    to_ocr = [p for p in minutes_files
              if not (OCR_DIR / (p.stem + ".txt")).exists()]
    cached = len(minutes_files) - len(to_ocr)
    if cached:
        print(f"  {cached} already cached in {OCR_DIR}/ — skipping", file=sys.stderr)
    if not to_ocr:
        print("  Nothing to OCR. (Use `--force` to re-OCR everything.)", file=sys.stderr)
        return

    num_workers = min(multiprocessing.cpu_count(), 6)
    print(f"OCR'ing {len(to_ocr)} files with {num_workers} workers...", file=sys.stderr)

    completed = 0
    total = len(to_ocr)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_ocr_and_cache, str(p)): p for p in to_ocr}
        for future in as_completed(futures):
            pdf_path = futures[future]
            completed += 1
            try:
                name, chars = future.result()
                print(
                    f"  [{completed}/{total}] {name}: {chars:,} chars cached",
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f"  [{completed}/{total}] {pdf_path.name}: OCR ERROR — {e}",
                    file=sys.stderr,
                )

    print(f"OCR cache: {OCR_DIR}/", file=sys.stderr)


# ---------- Parse ----------

def clean_text(text: str) -> str:
    text = re.sub(r"Page\s+\d+", "", text)
    text = re.sub(r"Veritext Legal Solutions", "", text)
    text = re.sub(r"800-227-8440", "", text)
    text = re.sub(r"973-410-4040", "", text)
    text = re.sub(r"^\s*\d{1,2}\s{2,}", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_commissioner_words(text: str) -> dict[str, int]:
    text = clean_text(text)
    word_counts = defaultdict(int)
    matches = list(SPEAKER_RE.finditer(text))
    for match in matches:
        speaker_label = match.group(1)
        canonical_name = None
        for pattern, name in COMMISSIONER_PATTERNS.items():
            if re.match(pattern, speaker_label, re.IGNORECASE):
                canonical_name = name
                break
        if canonical_name is None:
            continue
        start = match.end()
        next_speaker = ANY_SPEAKER_RE.search(text, start)
        end = next_speaker.start() if next_speaker else len(text)
        segment = text[start:end].strip()
        word_counts[canonical_name] += len(segment.split())
    return dict(word_counts)


def parse_filename(filename: str) -> tuple[str, str]:
    """Parse 2025-01-07_Caucus_Minutes.pdf → (date, meeting_type)."""
    stem = Path(filename).stem
    parts = stem.split("_")
    date = parts[0]
    meeting_type = parts[1] if len(parts) > 1 else "Unknown"
    return date, meeting_type


def parse_cached_transcript(text_path: Path) -> dict:
    text = text_path.read_text(encoding="utf-8")
    pdf_filename = text_path.stem + ".pdf"
    date, meeting_type = parse_filename(pdf_filename)
    word_counts = extract_commissioner_words(text)
    return {
        "date": date,
        "meeting_type": meeting_type,
        "file": pdf_filename,
        "commissioner_word_counts": word_counts,
        "total_commissioner_words": sum(word_counts.values()),
    }


def cmd_parse(args) -> None:
    if not OCR_DIR.exists():
        print(
            f"Error: no OCR cache at {OCR_DIR}/. "
            f"Run `{Path(__file__).name} ocr` (or `all`) first.",
            file=sys.stderr,
        )
        sys.exit(1)

    text_files = sorted(OCR_DIR.glob("*_Minutes.txt"))
    if not text_files:
        print(
            f"Error: {OCR_DIR}/ contains no *_Minutes.txt files. "
            f"Run `{Path(__file__).name} ocr` first.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Parsing {len(text_files)} cached transcripts from {OCR_DIR}/",
          file=sys.stderr)

    results = {
        "meetings": [],
        "summary_by_type": {},
        "summary_by_commissioner": {},
    }
    type_totals = defaultdict(lambda: defaultdict(int))
    commissioner_totals = defaultdict(lambda: defaultdict(int))

    for i, text_path in enumerate(text_files, 1):
        try:
            record = parse_cached_transcript(text_path)
            results["meetings"].append(record)
            for commissioner, wc in record["commissioner_word_counts"].items():
                type_totals[record["meeting_type"]][commissioner] += wc
                commissioner_totals[commissioner][record["meeting_type"]] += wc
            print(
                f"  [{i}/{len(text_files)}] {record['file']}: "
                f"{record['total_commissioner_words']} words, "
                f"{len(record['commissioner_word_counts'])} commissioners",
                file=sys.stderr,
            )
        except Exception as e:
            date, meeting_type = parse_filename(text_path.stem + ".pdf")
            print(
                f"  [{i}/{len(text_files)}] {text_path.name}: PARSE ERROR — {e}",
                file=sys.stderr,
            )
            results["meetings"].append({
                "date": date,
                "meeting_type": meeting_type,
                "file": text_path.stem + ".pdf",
                "error": str(e),
            })

    results["meetings"].sort(key=lambda m: m["date"])

    for mtype, commissioners in sorted(type_totals.items()):
        results["summary_by_type"][mtype] = {
            "commissioner_word_counts": dict(
                sorted(commissioners.items(), key=lambda x: -x[1])
            ),
            "total_words": sum(commissioners.values()),
            "meeting_count": sum(
                1 for m in results["meetings"]
                if m.get("meeting_type") == mtype and "error" not in m
            ),
        }

    for commissioner, types in sorted(commissioner_totals.items()):
        results["summary_by_commissioner"][commissioner] = {
            "by_meeting_type": dict(sorted(types.items())),
            "total_words": sum(types.values()),
        }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {OUTPUT_JSON}", file=sys.stderr)
    print(f"Total meetings: {len(results['meetings'])}", file=sys.stderr)


# ---------- All ----------

def cmd_all(args) -> None:
    cmd_ocr(args)
    cmd_parse(args)


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="{ocr,parse,all}")

    p_ocr = sub.add_parser("ocr", help="OCR PDFs and cache text to ./ocr/")
    p_ocr.add_argument("--force", action="store_true",
                       help="Re-OCR every PDF even if a cached transcript exists")
    p_ocr.set_defaults(func=cmd_ocr)

    p_parse = sub.add_parser("parse", help="Parse the cached OCR text into JSON")
    p_parse.set_defaults(func=cmd_parse)

    p_all = sub.add_parser("all", help="Run `ocr` (using cache) then `parse`")
    p_all.add_argument("--force", action="store_true",
                       help="Re-OCR every PDF even if a cached transcript exists")
    p_all.set_defaults(func=cmd_all)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
