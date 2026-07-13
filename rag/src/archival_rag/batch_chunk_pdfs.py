"""
Batch-process an entire folder of PDFs into chunks, using the chunking
logic from pdf_to_chunks_v3.py.

WHAT IT DOES
------------
1. Loops over every .pdf file in an input folder.
2. Runs chunk_pdf() on each one, wrapped in try/except so ONE broken
   file (e.g. a badly scanned 1971 document) does not stop the whole
   batch.
3. Tries to guess `regulation_name` and `document_date` from the
   filename / PDF text if you don't supply a lookup table (see
   FILENAME_META below) -- you can improve this mapping for your
   actual filenames.
4. Writes:
     - all_chunks.json          <- every chunk from every PDF, ready
                                    to embed and upsert into Qdrant
     - processing_report.json   <- per-file status: success / error,
                                    chunk counts, error messages
     - processing_report.txt    <- same thing, human-readable summary

USAGE
-----
    python batch_chunk_pdfs.py /path/to/pdf_folder

If no path is given, defaults to ./pdfs relative to this script.
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from typing import Any

from archival_rag.pdf_to_chunks_v3 import Chunk, chunk_pdf

# ------------------------------------------------------------------
# OPTIONAL: fill this in if you know regulation names / dates per file.
# Key = exact filename (with extension). Anything not listed here
# falls back to best-effort guessing from the filename itself.
# ------------------------------------------------------------------
FILENAME_META: dict[str, dict[str, str]] = {
    # "brd_bodenleger_2002.pdf": {
    #     "regulation_name": "Bodenleger",
    #     "document_date": "2002-06-17",
    # },
}


def guess_meta_from_filename(filename: str) -> dict[str, str]:
    """
    Best-effort fallback when a file isn't in FILENAME_META.
    Assumes filenames roughly look like: brd_<beruf>_<jahr>.pdf
    Adjust this regex if your actual filenames follow a different
    convention.
    """
    name = os.path.splitext(filename)[0]
    year_match = re.search(r"(19|20)\d{2}", name)
    document_date = year_match.group(0) if year_match else ""

    # strip known prefixes/year to guess the regulation name
    cleaned = re.sub(r"(19|20)\d{2}", "", name)
    cleaned = re.sub(r"^(brd|bgbl)[_\-]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip("_- ").replace("_", " ").title()

    return {"regulation_name": cleaned or name, "document_date": document_date}


def process_folder(folder_path: str, output_dir: str = ".") -> None:
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".pdf"))

    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return

    all_chunks: list[Chunk] = []
    report: list[dict[str, Any]] = []

    for filename in pdf_files:
        path = os.path.join(folder_path, filename)
        meta = FILENAME_META.get(filename) or guess_meta_from_filename(filename)

        print(f"Processing {filename} ...")
        try:
            chunks = chunk_pdf(
                path,
                regulation_name=meta["regulation_name"],
                document_date=meta["document_date"],
            )

            for c in chunks:
                c["metadata"]["source_file"] = filename

            all_chunks.extend(chunks)

            text_count = sum(1 for c in chunks if c["chunk_type"] == "text_section")
            table_count = sum(1 for c in chunks if c["chunk_type"] == "table_row")

            report.append(
                {
                    "filename": filename,
                    "status": "success",
                    "regulation_name": meta["regulation_name"],
                    "document_date": meta["document_date"],
                    "total_chunks": len(chunks),
                    "text_chunks": text_count,
                    "table_chunks": table_count,
                    "warning": (
                        "no table_row chunks found -- check this file manually"
                        if table_count == 0
                        else None
                    ),
                }
            )
            print(f"  -> {len(chunks)} chunks ({text_count} text, {table_count} table)")

        except Exception as e:
            report.append(
                {
                    "filename": filename,
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"  -> FAILED: {e}")

    chunks_path = os.path.join(output_dir, "all_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    report_json_path = os.path.join(output_dir, "processing_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # human-readable summary
    report_txt_path = os.path.join(output_dir, "processing_report.txt")
    successes = [r for r in report if r["status"] == "success"]
    errors = [r for r in report if r["status"] == "error"]
    warnings = [r for r in successes if r.get("warning")]

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("PDF Chunking Batch Report\n")
        f.write("=========================\n")
        f.write(f"Total files: {len(pdf_files)}\n")
        f.write(f"Succeeded:   {len(successes)}\n")
        f.write(f"Failed:      {len(errors)}\n")
        f.write(f"Warnings (0 table chunks found): {len(warnings)}\n\n")

        if errors:
            f.write("--- FAILED FILES ---\n")
            for r in errors:
                f.write(f"{r['filename']}: {r['error']}\n")
            f.write("\n")

        if warnings:
            f.write("--- FILES WITH NO TABLE CHUNKS (check manually) ---\n")
            for r in warnings:
                f.write(f"{r['filename']}\n")
            f.write("\n")

        f.write("--- ALL FILES ---\n")
        for r in report:
            if r["status"] == "success":
                f.write(
                    f"{r['filename']}: OK -- {r['total_chunks']} chunks "
                    f"({r['text_chunks']} text, {r['table_chunks']} table)\n"
                )
            else:
                f.write(f"{r['filename']}: FAILED -- {r['error']}\n")

    print(
        f"\nDone. {len(successes)}/{len(pdf_files)} files succeeded, "
        f"{len(errors)} failed, {len(warnings)} succeeded with 0 table chunks."
    )
    print(f"All chunks -> {chunks_path}")
    print(f"Report -> {report_txt_path}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "./pdfs"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    process_folder(folder, output_dir)
