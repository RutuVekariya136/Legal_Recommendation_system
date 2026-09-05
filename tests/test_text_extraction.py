"""Tests for app.ml.text_extraction (Step 5)."""

from __future__ import annotations

import pymupdf
import pytest

from app.ml.text_extraction import (
    ExtractedDocument,
    ExtractionError,
    extract_and_preprocess,
    extract_text,
    extract_text_from_pdf,
    extract_text_from_txt,
)

SAMPLE = (
    "IN THE SUPREME COURT OF INDIA\n"
    "A. B. Rao v Union of India\n\n"
    "1. This appeal arises under No. 204 of 1952 before this Court.\n"
    "2. The question is whether Section 66 of the Act applies here.\n"
)


def _make_pdf_bytes(text: str, *, chunk: int = 1500, repeat_header: str | None = None) -> bytes:
    doc = pymupdf.open()
    rect = pymupdf.Rect(40, 40, 555, 800)
    chunks = [text[i : i + chunk] for i in range(0, len(text), chunk)] or [""]
    for piece in chunks:
        page = doc.new_page()
        body = (repeat_header + "\n" + piece) if repeat_header else piece
        page.insert_textbox(rect, body, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def _make_scanned_pdf_bytes(pages: int = 3) -> bytes:
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()  # blank page, no text layer
    data = doc.tobytes()
    doc.close()
    return data


# -- plain text -----------------------------------------------------------

def test_extract_txt_utf8(tmp_path):
    p = tmp_path / "case.txt"
    p.write_text("Ravi v State — café résumé", encoding="utf-8")
    doc = extract_text_from_txt(p)
    assert doc.source_type == "txt"
    assert "café" in doc.raw_text
    assert doc.metadata["encoding"] in {"utf-8", "utf-8-sig"}


def test_extract_txt_latin1_fallback(tmp_path):
    p = tmp_path / "legacy.txt"
    p.write_bytes("Ravi v State \xe9tat".encode("latin-1"))
    doc = extract_text_from_txt(p)
    assert "�" not in doc.raw_text  # decoded, not mangled
    assert doc.metadata["encoding"] in {"cp1252", "latin-1"}


def test_extract_txt_empty_raises(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("   \n\n", encoding="utf-8")
    with pytest.raises(ExtractionError):
        extract_text_from_txt(p)


# -- pdf ----------------------------------------------------------------

def test_extract_pdf_roundtrip():
    doc = extract_text_from_pdf(_make_pdf_bytes(SAMPLE), source_name="s.pdf")
    assert doc.source_type == "pdf"
    assert doc.page_count >= 1
    assert not doc.needs_ocr
    assert "Union of India" in doc.raw_text
    assert "Section 66" in doc.raw_text


def test_extract_pdf_strips_repeated_headers():
    data = _make_pdf_bytes(SAMPLE * 4, chunk=200, repeat_header="CONFIDENTIAL DRAFT — Registry Copy")
    doc = extract_text_from_pdf(data, drop_repeated_lines=True)
    assert "Registry Copy" not in doc.raw_text
    assert any("repeated header/footer" in w for w in doc.warnings)


def test_extract_scanned_pdf_flags_ocr():
    doc = extract_text_from_pdf(_make_scanned_pdf_bytes(), source_name="scan.pdf")
    assert doc.needs_ocr is True
    assert any("scanned" in w.lower() or "ocr" in w.lower() for w in doc.warnings)


# -- dispatch ---------------------------------------------------------

def test_dispatch_by_extension(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text(SAMPLE, encoding="utf-8")
    assert extract_text(p).source_type == "txt"


def test_dispatch_by_content_sniff_bytes():
    assert extract_text(_make_pdf_bytes(SAMPLE), source_name="x").source_type == "pdf"
    assert extract_text(SAMPLE.encode("utf-8"), source_name="x").source_type == "txt"


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "c.docx"
    p.write_bytes(b"PK\x03\x04stuff")
    with pytest.raises(ExtractionError):
        extract_text(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ExtractionError):
        extract_text(tmp_path / "nope.pdf")


# -- extraction + preprocessing together ----------------------------

def test_extract_and_preprocess():
    extracted, processed = extract_and_preprocess(_make_pdf_bytes(SAMPLE), source_name="s.pdf")
    assert isinstance(extracted, ExtractedDocument)
    assert processed.tokens
    assert "section" in processed.tokens
    assert processed.sentences
