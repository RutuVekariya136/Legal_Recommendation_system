"""Text extraction for the upload path (pipeline stage: Text Extraction).

    [Legal Case -> Text Extraction] -> Preprocessing -> Text Embedding -> ...

Turns an uploaded file (PDF or plain text) into raw text that then goes through
the *same* ``app.ml.preprocessing`` pipeline as the indexed corpus, so an
uploaded case and a corpus case are represented identically.

PDF handling uses PyMuPDF (``pymupdf``), which is already a dependency.

Scope / limitations (surfaced, not hidden):
  * We extract the PDF **text layer** only. Scanned / image-only PDFs have no
    text layer -> we detect that and return ``needs_ocr=True`` with a warning.
    OCR itself is out of scope (no OCR engine is a dependency).
  * Password-protected PDFs are detected and rejected with a clear message
    (empty-password documents are opened transparently).
  * Repeated running headers/footers (court name, page numbers) that PDF export
    duplicates on every page can optionally be stripped
    (``drop_repeated_lines=True``); ``clean_text`` in preprocessing also removes
    "Page N" markers.

No FastAPI dependency: importable and runnable from a plain script / pytest.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pymupdf

from app.ml.preprocessing import (
    DEFAULT_CONFIG,
    PreprocessConfig,
    ProcessedDocument,
    preprocess_text,
)

PathLike = Union[str, Path]

SUPPORTED_SUFFIXES = {".pdf", ".txt"}
_TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
# Below this many extracted characters per page, a PDF is almost certainly
# scanned (image-only) rather than a born-digital judgment.
_MIN_CHARS_PER_PAGE_FOR_TEXT_LAYER = 30


class ExtractionError(RuntimeError):
    """Raised when a file cannot be turned into usable text."""


@dataclass
class ExtractedDocument:
    source_name: str
    source_type: str  # "pdf" | "txt"
    raw_text: str
    page_count: Optional[int] = None
    char_count: int = 0
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    needs_ocr: bool = False

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = len(self.raw_text)

    def preprocess(self, config: PreprocessConfig = DEFAULT_CONFIG) -> ProcessedDocument:
        return preprocess_text(self.raw_text, doc_id=None, config=config)


# --------------------------------------------------------------------------- #
# Plain text                                                                  #
# --------------------------------------------------------------------------- #

def _decode_bytes(data: bytes) -> tuple[str, str]:
    for enc in _TEXT_ENCODINGS:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    # last resort: never raise on a text upload
    return data.decode("latin-1", errors="replace"), "latin-1+replace"


def extract_text_from_txt(source: Union[PathLike, bytes], *, source_name: str = "upload.txt") -> ExtractedDocument:
    if isinstance(source, (str, Path)):
        path = Path(source)
        source_name = path.name
        data = path.read_bytes()
    else:
        data = source

    text, encoding = _decode_bytes(data)
    warnings: list[str] = []
    if "replace" in encoding:
        warnings.append("some bytes could not be decoded and were replaced")
    if not text.strip():
        raise ExtractionError(f"{source_name}: file contains no text")

    return ExtractedDocument(
        source_name=source_name,
        source_type="txt",
        raw_text=text,
        metadata={"encoding": encoding},
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# PDF                                                                         #
# --------------------------------------------------------------------------- #

def _strip_repeated_lines(pages: list[str], *, threshold: float = 0.6) -> tuple[list[str], int]:
    """Drop short lines that appear (stripped) on >= ``threshold`` of pages -
    i.e. running headers / footers duplicated by PDF export."""
    if len(pages) < 3:
        return pages, 0
    counts: Counter[str] = Counter()
    for page in pages:
        for line in {ln.strip() for ln in page.splitlines() if ln.strip()}:
            counts[line] += 1
    cutoff = max(2, int(threshold * len(pages)))
    boilerplate = {ln for ln, c in counts.items() if c >= cutoff and len(ln) <= 120}
    if not boilerplate:
        return pages, 0
    removed = 0
    cleaned = []
    for page in pages:
        kept = []
        for ln in page.splitlines():
            if ln.strip() in boilerplate:
                removed += 1
                continue
            kept.append(ln)
        cleaned.append("\n".join(kept))
    return cleaned, removed


def extract_text_from_pdf(
    source: Union[PathLike, bytes],
    *,
    source_name: str = "upload.pdf",
    drop_repeated_lines: bool = True,
) -> ExtractedDocument:
    if isinstance(source, (str, Path)):
        path = Path(source)
        source_name = path.name
        open_kwargs = {"filename": str(path)}
    else:
        open_kwargs = {"stream": source, "filetype": "pdf"}

    try:
        doc = pymupdf.open(**open_kwargs)
    except Exception as exc:  # noqa: BLE001 - pymupdf raises assorted types
        raise ExtractionError(f"{source_name}: cannot open as PDF ({exc})") from exc

    warnings: list[str] = []
    try:
        if doc.needs_pass:
            if not doc.authenticate(""):
                raise ExtractionError(
                    f"{source_name}: PDF is password-protected; remove the password and re-upload"
                )

        page_count = doc.page_count
        pages = [doc[i].get_text("text") for i in range(page_count)]
        pdf_metadata = dict(doc.metadata or {})
    finally:
        doc.close()

    removed_lines = 0
    if drop_repeated_lines:
        pages, removed_lines = _strip_repeated_lines(pages)
        if removed_lines:
            warnings.append(f"removed {removed_lines} repeated header/footer line(s)")

    raw_text = unicodedata.normalize("NFC", "\n".join(pages)).strip()

    needs_ocr = False
    if page_count and len(raw_text) < _MIN_CHARS_PER_PAGE_FOR_TEXT_LAYER * page_count:
        needs_ocr = True
        warnings.append(
            "little or no extractable text - this looks like a scanned/image PDF; "
            "OCR is required (not supported here)"
        )
    if not raw_text and not needs_ocr:
        raise ExtractionError(f"{source_name}: no text could be extracted")

    return ExtractedDocument(
        source_name=source_name,
        source_type="pdf",
        raw_text=raw_text,
        page_count=page_count,
        metadata={k: v for k, v in pdf_metadata.items() if v},
        warnings=warnings,
        needs_ocr=needs_ocr,
    )


# --------------------------------------------------------------------------- #
# Dispatch                                                                    #
# --------------------------------------------------------------------------- #

def extract_text(
    source: Union[PathLike, bytes],
    *,
    source_name: Optional[str] = None,
    **kwargs,
) -> ExtractedDocument:
    """Dispatch on file extension (path) or content sniff (bytes)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ExtractionError(f"file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return extract_text_from_pdf(path, **kwargs)
        if suffix == ".txt":
            return extract_text_from_txt(path)
        raise ExtractionError(
            f"{path.name}: unsupported file type {suffix!r}; expected one of {sorted(SUPPORTED_SUFFIXES)}"
        )

    data = source
    name = source_name or "upload"
    if data[:5] == b"%PDF-":
        return extract_text_from_pdf(data, source_name=name if name.endswith(".pdf") else f"{name}.pdf", **kwargs)
    return extract_text_from_txt(data, source_name=name if name.endswith(".txt") else f"{name}.txt")


def extract_and_preprocess(
    source: Union[PathLike, bytes],
    *,
    source_name: Optional[str] = None,
    config: PreprocessConfig = DEFAULT_CONFIG,
    **kwargs,
) -> tuple[ExtractedDocument, ProcessedDocument]:
    """Convenience: extraction + the shared preprocessing pipeline in one call."""
    extracted = extract_text(source, source_name=source_name, **kwargs)
    processed = extracted.preprocess(config=config)
    return extracted, processed
