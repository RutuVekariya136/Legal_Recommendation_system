"""Dataset loader for the base-paper corpus (CLAUDE.md §3).

Layout on disk (already present, verified):

    data/raw/
      validation/
        documents/<doc_id>.txt        143 files
        similarity_scores.csv          100 rows: doc1,doc2,score  (no header)
      test/
        documents/<doc_id>.txt        153 files
        similarity_scores.csv           90 rows

`<doc_id>` looks like ``1971_138`` or ``1955_S_34`` (``<year>_<...>``).

Facts established by inspecting the real data (not assumptions):
  * Exactly one filename, ``2011_I_16``, appears in BOTH splits with byte-identical
    content -> the corpus has **295 unique documents**, and that document simply
    belongs to both splits.
  * Every doc_id referenced in a split's ``similarity_scores.csv`` resolves to a
    document IN THAT SAME SPLIT (100% referential integrity).
  * ~60 / 296 documents have no party-style ("X v Y") first line; their ``title``
    is ``None``. This is a real limitation that later citation matching (Step 6)
    must cope with — it is surfaced, never hidden.

This module has no FastAPI / ML dependency and is runnable from a plain script.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Iterator, Optional

from app.utils.paths import data_raw_dir

SPLITS: tuple[str, ...] = ("validation", "test")

# Expected shape from the base paper — used for validation, NOT as a result.
EXPECTED_DOCUMENTS = {"validation": 143, "test": 153}
EXPECTED_PAIRS = {"validation": 100, "test": 90}

_YEAR_RE = re.compile(r"^(\d{4})_")
# A standalone "v" / "vs" / "vs." / "versus" between two parties.
_PARTY_RE = re.compile(r"\bv(?:s\.?|ersus)?\b", re.IGNORECASE)
_COURT_PREFIXES = (
    "supreme court",
    "in the supreme court",
    "high court",
    "in the high court",
)


def parse_year(doc_id: str) -> Optional[int]:
    """``"1971_138"`` -> ``1971``; ``"foo"`` -> ``None``."""
    m = _YEAR_RE.match(doc_id)
    return int(m.group(1)) if m else None


def extract_title(text: str) -> Optional[str]:
    """Best-effort case title = the first non-empty line, if it looks like a
    party line ("X v Y") and is not a court-name header. Otherwise ``None``.

    Deliberately conservative: a wrong title is worse than no title for the
    fuzzy citation matching in Step 6.
    """
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(_COURT_PREFIXES):
            return None
        if len(line) <= 300 and _PARTY_RE.search(line):
            return line
        return None
    return None


@dataclass
class CaseDocument:
    """One judgment document. Text is read lazily and cached on first access."""

    doc_id: str
    path: Path
    splits: tuple[str, ...]
    year: Optional[int]

    @cached_property
    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    @cached_property
    def title(self) -> Optional[str]:
        return extract_title(self.text)

    @property
    def has_title(self) -> bool:
        return self.title is not None

    def sha1(self) -> str:
        return hashlib.sha1(self.path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ExpertPair:
    """One expert-scored document pair from ``similarity_scores.csv``."""

    doc_id_1: str
    doc_id_2: str
    score: float
    split: str


class DatasetError(RuntimeError):
    """Raised when the on-disk dataset is structurally broken."""


@dataclass
class Corpus:
    """Loaded corpus: unique documents keyed by doc_id, plus expert pairs."""

    raw_dir: Path
    documents: dict[str, CaseDocument] = field(default_factory=dict)
    pairs_by_split: dict[str, list[ExpertPair]] = field(default_factory=dict)

    # -- convenience accessors ------------------------------------------------
    def __len__(self) -> int:
        return len(self.documents)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.documents

    def __getitem__(self, doc_id: str) -> CaseDocument:
        return self.documents[doc_id]

    @property
    def doc_ids(self) -> list[str]:
        return sorted(self.documents)

    @property
    def all_pairs(self) -> list[ExpertPair]:
        return [p for split in SPLITS for p in self.pairs_by_split.get(split, [])]

    def split_doc_ids(self, split: str) -> list[str]:
        return sorted(d.doc_id for d in self.documents.values() if split in d.splits)

    def text(self, doc_id: str) -> str:
        return self.documents[doc_id].text

    def titles(self) -> dict[str, Optional[str]]:
        return {d.doc_id: d.title for d in self.documents.values()}

    def iter_texts(self) -> Iterator[tuple[str, str]]:
        for doc_id in self.doc_ids:
            yield doc_id, self.documents[doc_id].text

    # -- integrity ----------------------------------------------------------
    def check_integrity(self) -> list[str]:
        """Return a list of human-readable problems ([] == clean)."""
        problems: list[str] = []

        for split in SPLITS:
            n_docs = len(self.split_doc_ids(split))
            if n_docs != EXPECTED_DOCUMENTS[split]:
                problems.append(
                    f"{split}: {n_docs} documents, expected {EXPECTED_DOCUMENTS[split]}"
                )
            pairs = self.pairs_by_split.get(split, [])
            if len(pairs) != EXPECTED_PAIRS[split]:
                problems.append(
                    f"{split}: {len(pairs)} expert pairs, expected {EXPECTED_PAIRS[split]}"
                )
            for p in pairs:
                for did in (p.doc_id_1, p.doc_id_2):
                    doc = self.documents.get(did)
                    if doc is None:
                        problems.append(
                            f"{split}: pair references unknown doc_id {did!r}"
                        )
                    elif split not in doc.splits:
                        problems.append(
                            f"{split}: pair references {did!r} which is not in this split"
                        )
                if not 0.0 <= p.score <= 1.0:
                    problems.append(
                        f"{split}: pair {p.doc_id_1}/{p.doc_id_2} score {p.score} out of [0,1]"
                    )

        for doc in self.documents.values():
            if not doc.path.is_file():
                problems.append(f"missing file: {doc.path}")
            elif doc.path.stat().st_size == 0:
                problems.append(f"empty file: {doc.path}")

        return problems


def _load_documents(raw_dir: Path) -> dict[str, CaseDocument]:
    by_id: dict[str, dict] = {}
    for split in SPLITS:
        docs_dir = raw_dir / split / "documents"
        if not docs_dir.is_dir():
            raise DatasetError(f"missing documents folder: {docs_dir}")
        for path in sorted(docs_dir.glob("*.txt")):
            doc_id = path.stem
            entry = by_id.get(doc_id)
            if entry is None:
                by_id[doc_id] = {"path": path, "splits": [split]}
            else:
                # Same doc_id in both splits — only OK if content is identical.
                existing = entry["path"].read_bytes()
                if hashlib.sha1(existing).digest() != hashlib.sha1(path.read_bytes()).digest():
                    raise DatasetError(
                        f"doc_id {doc_id!r} differs between splits "
                        f"({entry['path']} vs {path})"
                    )
                entry["splits"].append(split)

    return {
        doc_id: CaseDocument(
            doc_id=doc_id,
            path=e["path"],
            splits=tuple(e["splits"]),
            year=parse_year(doc_id),
        )
        for doc_id, e in by_id.items()
    }


def load_expert_pairs(split: str, raw_dir: Optional[Path] = None) -> list[ExpertPair]:
    """Parse ``data/raw/<split>/similarity_scores.csv`` (doc1,doc2,score; no header)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    raw_dir = raw_dir or data_raw_dir()
    csv_path = raw_dir / split / "similarity_scores.csv"
    if not csv_path.is_file():
        raise DatasetError(f"missing expert scores: {csv_path}")

    pairs: list[ExpertPair] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip():
                continue
            if row[0].strip().lower() in {"doc1", "doc_1", "query", "case1"}:
                continue  # tolerate an accidental header
            if len(row) < 3:
                raise DatasetError(f"{csv_path}: malformed row {row!r}")
            pairs.append(
                ExpertPair(
                    doc_id_1=row[0].strip(),
                    doc_id_2=row[1].strip(),
                    score=float(row[2]),
                    split=split,
                )
            )
    return pairs


def load_corpus(raw_dir: Optional[Path] = None, *, verify: bool = True) -> Corpus:
    """Load the full corpus (both splits merged, keyed by unique doc_id).

    Set ``verify=False`` to skip the integrity check (e.g. when deliberately
    testing against a broken fixture).
    """
    raw_dir = raw_dir or data_raw_dir()
    if not raw_dir.is_dir():
        raise DatasetError(f"data raw dir not found: {raw_dir}")

    corpus = Corpus(raw_dir=raw_dir, documents=_load_documents(raw_dir))
    corpus.pairs_by_split = {s: load_expert_pairs(s, raw_dir) for s in SPLITS}

    if verify:
        problems = corpus.check_integrity()
        if problems:
            raise DatasetError(
                "dataset integrity check failed:\n  - " + "\n  - ".join(problems)
            )
    return corpus
