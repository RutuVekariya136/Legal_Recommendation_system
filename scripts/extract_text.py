"""extract_text.py - Step 5: extract raw text from an uploaded PDF / .txt file.

This is the manual/CLI entry point for the upload path. The FastAPI upload
endpoint (Step 21) will call the same `app.ml.text_extraction` functions.

Usage:
    python scripts/extract_text.py path/to/case.pdf
    python scripts/extract_text.py path/to/case.pdf --preprocess
    python scripts/extract_text.py path/to/case.pdf --out data/processed/uploads/case.txt

    # No PDF handy? Round-trip a corpus document through a generated PDF:
    python scripts/extract_text.py --selftest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.ml.text_extraction import (  # noqa: E402
    ExtractionError,
    extract_and_preprocess,
    extract_text,
)
from app.utils.paths import data_processed_dir, ensure_dir  # noqa: E402


def _print_report(extracted, processed=None) -> None:
    print("=" * 70)
    print(f"source:      {extracted.source_name}  ({extracted.source_type})")
    if extracted.page_count is not None:
        print(f"pages:       {extracted.page_count}")
    print(f"characters:  {extracted.char_count:,}")
    if extracted.metadata:
        print(f"metadata:    {extracted.metadata}")
    if extracted.needs_ocr:
        print("needs_ocr:   YES  (scanned/image PDF - OCR not supported here)")
    for w in extracted.warnings:
        print(f"warning:     {w}")
    preview = extracted.raw_text[:400].replace("\n", " ")
    print(f"\npreview:     {preview}...")
    if processed is not None:
        s = processed.stats
        print(
            f"\npreprocessed: header_detected={s['header_detected']} "
            f"paragraphs={s['n_paragraphs']} sentences={s['n_sentences']} "
            f"tokens={s['n_tokens']} unique_tokens={s['n_unique_tokens']}"
        )


def _selftest() -> int:
    import pymupdf

    from app.utils.dataset import load_corpus

    corpus = load_corpus()
    doc_id = corpus.doc_ids[0]
    original = corpus.text(doc_id)

    pdf = pymupdf.open()
    rect = pymupdf.Rect(50, 50, 545, 790)
    chunk = 2800  # chars per page - comfortably fits the textbox at fontsize 9
    for start in range(0, len(original), chunk):
        page = pdf.new_page()
        page.insert_textbox(rect, original[start:start + chunk], fontsize=9)
    data = pdf.tobytes()
    pdf.close()

    extracted = extract_text(data, source_name=f"{doc_id}.pdf")
    _print_report(extracted)

    orig_words = original.split()
    got_words = extracted.raw_text.split()
    overlap = len(set(orig_words) & set(got_words)) / max(1, len(set(orig_words)))
    print(f"\nround-trip word overlap with corpus source: {overlap:.1%}")
    ok = overlap > 0.9 and not extracted.needs_ocr
    print("[ OK ]" if ok else "[FAIL]", "selftest")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, help="PDF or .txt file to extract")
    parser.add_argument("--preprocess", action="store_true", help="also run the preprocessing pipeline")
    parser.add_argument("--out", type=Path, default=None, help="write extracted text here")
    parser.add_argument("--stdout", action="store_true", help="print full extracted text to stdout")
    parser.add_argument("--selftest", action="store_true", help="round-trip a corpus doc through a generated PDF")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.file:
        parser.error("provide a file, or use --selftest")

    try:
        if args.preprocess:
            extracted, processed = extract_and_preprocess(args.file)
        else:
            extracted, processed = extract_text(args.file), None
    except ExtractionError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    _print_report(extracted, processed)

    if args.stdout:
        print("\n" + "-" * 70 + "\n" + extracted.raw_text)

    out_path = args.out
    if out_path is None and not args.stdout:
        out_path = data_processed_dir() / "uploads" / f"{args.file.stem}.txt"
    if out_path is not None:
        ensure_dir(out_path.parent)
        out_path.write_text(extracted.raw_text, encoding="utf-8")
        print(f"\n[ OK ] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
