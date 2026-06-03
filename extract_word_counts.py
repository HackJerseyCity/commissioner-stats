#!/usr/bin/env python3
"""
Extract commissioner word counts from Hudson County Board of County Commissioners
meeting minutes PDFs. Outputs JSON summarizing each commissioner's word counts
per meeting type.

Uses parallel OCR across files and pages for speed.

Requires: pymupdf (fitz), tesseract
"""

import fitz
import subprocess
import re
import json
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Known commissioner speaker labels (as they appear in transcripts)
COMMISSIONER_PATTERNS = {
    r"CHAIRMAN\s+ROMANO": "Anthony Romano",
    r"COMMISSIONER\s+APONTE[\s-]*LIPSKI": "Yraida Aponte-Lipski",
    r"COMMISSIONER\s+BASELICE": "Robert Baselice",
    r"COMMISSIONER\s+CEDE[NÑ]O": "Fanny Cedeno",
    r"COMMISSIONER\s+CIFELLI": "Albert Cifelli",
    r"COMMISSIONER\s+KOPACZ": "Kenneth Kopacz",
    r"COMMISSIONER\s+O[\u2019'']?DEA": "William O'Dea",
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


def ocr_page(args: tuple) -> tuple[int, str]:
    """OCR a single page. Designed to run in a worker process."""
    pdf_path, page_num = args
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # Try native text first
    text = page.get_text()
    if text.strip():
        doc.close()
        return (page_num, text)

    # OCR fallback
    pix = page.get_pixmap(dpi=250)
    img_data = pix.tobytes("png")
    doc.close()

    result = subprocess.run(
        ["tesseract", "stdin", "stdout"],
        input=img_data,
        capture_output=True,
    )
    text = result.stdout.decode("utf-8", errors="replace")
    return (page_num, text)


def ocr_pdf_parallel(pdf_path: str, pool: ProcessPoolExecutor) -> str:
    """Extract text from a scanned PDF using parallel page OCR."""
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()

    tasks = [(pdf_path, i) for i in range(num_pages)]
    futures = {pool.submit(ocr_page, t): t[1] for t in tasks}

    pages = {}
    for future in as_completed(futures):
        page_num, text = future.result()
        pages[page_num] = text

    # Reassemble in order
    return "\n".join(pages[i] for i in range(num_pages))


def clean_text(text: str) -> str:
    """Clean OCR artifacts from text."""
    text = re.sub(r"Page\s+\d+", "", text)
    text = re.sub(r"Veritext Legal Solutions", "", text)
    text = re.sub(r"800-227-8440", "", text)
    text = re.sub(r"973-410-4040", "", text)
    text = re.sub(r"^\s*\d{1,2}\s{2,}", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_commissioner_words(text: str) -> dict[str, int]:
    """Parse transcript text and count words spoken by each commissioner."""
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
    """Parse filename like 2025-01-07_Caucus_Minutes.pdf into (date, type)."""
    stem = Path(filename).stem
    parts = stem.split("_")
    date = parts[0]
    meeting_type = parts[1] if len(parts) > 1 else "Unknown"
    return date, meeting_type


def process_one_pdf(pdf_path: str) -> dict:
    """Process a single PDF file end-to-end (used for file-level parallelism)."""
    filename = Path(pdf_path).name
    date, meeting_type = parse_filename(filename)

    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()

    # OCR all pages in parallel within this process using threads
    # (tesseract is the bottleneck, and it releases the GIL via subprocess)
    from concurrent.futures import ThreadPoolExecutor

    def ocr_one_page(page_num):
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

    with ThreadPoolExecutor(max_workers=4) as tex:
        page_texts = list(tex.map(ocr_one_page, range(num_pages)))

    full_text = "\n".join(page_texts)
    word_counts = extract_commissioner_words(full_text)

    return {
        "date": date,
        "meeting_type": meeting_type,
        "file": filename,
        "commissioner_word_counts": word_counts,
        "total_commissioner_words": sum(word_counts.values()),
    }


def main():
    base_dir = Path(__file__).parent / "pdfs" / "2025"

    if not base_dir.exists():
        print(f"Error: {base_dir} not found", file=sys.stderr)
        sys.exit(1)

    minutes_files = sorted(base_dir.rglob("*_Minutes.pdf"))
    print(f"Found {len(minutes_files)} minutes files to process", file=sys.stderr)

    # Use process-level parallelism across files
    # Each process uses thread-level parallelism across pages
    num_workers = min(multiprocessing.cpu_count(), 6)
    print(f"Using {num_workers} parallel workers", file=sys.stderr)

    results = {
        "meetings": [],
        "summary_by_type": {},
        "summary_by_commissioner": {},
    }

    type_totals = defaultdict(lambda: defaultdict(int))
    commissioner_totals = defaultdict(lambda: defaultdict(int))

    completed = 0
    total = len(minutes_files)

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        future_to_path = {
            pool.submit(process_one_pdf, str(p)): p for p in minutes_files
        }

        for future in as_completed(future_to_path):
            pdf_path = future_to_path[future]
            completed += 1
            try:
                record = future.result()
                results["meetings"].append(record)

                for commissioner, wc in record["commissioner_word_counts"].items():
                    type_totals[record["meeting_type"]][commissioner] += wc
                    commissioner_totals[commissioner][record["meeting_type"]] += wc

                print(
                    f"  [{completed}/{total}] {record['file']}: "
                    f"{record['total_commissioner_words']} words, "
                    f"{len(record['commissioner_word_counts'])} commissioners",
                    file=sys.stderr,
                )
            except Exception as e:
                filename = pdf_path.name
                date, meeting_type = parse_filename(filename)
                print(f"  [{completed}/{total}] {filename}: ERROR - {e}", file=sys.stderr)
                results["meetings"].append({
                    "date": date,
                    "meeting_type": meeting_type,
                    "file": filename,
                    "error": str(e),
                })

    # Sort meetings by date
    results["meetings"].sort(key=lambda m: m["date"])

    # Build summary by meeting type
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

    # Build summary by commissioner
    for commissioner, types in sorted(commissioner_totals.items()):
        results["summary_by_commissioner"][commissioner] = {
            "by_meeting_type": dict(sorted(types.items())),
            "total_words": sum(types.values()),
        }

    # Output JSON
    output_path = Path(__file__).parent / "commissioner_word_counts.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {output_path}", file=sys.stderr)
    print(f"Total meetings processed: {len(results['meetings'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
