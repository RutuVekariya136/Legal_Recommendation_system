"""Text preprocessing for the judgment corpus (pipeline stage: Preprocessing).

    Legal Case -> Text Extraction -> [Preprocessing] -> Text Embedding -> Text Similarity

Produces three representations from one raw judgment, because downstream methods
need different things:

  * ``clean_text``  - unicode-normalised, de-hyphenated, whitespace-collapsed,
                      still cased and punctuated. For Sentence-BERT (our
                      enhancement), the case-comparison view, and display.
  * ``sentences``   - abbreviation-aware sentence segmentation of the body.
                      For sentence-level SBERT and explainability highlights.
  * ``tokens``      - lowercased alphabetic tokens, stopwords removed. This is
                      the bag-of-words input for **TF-IDF and Doc2Vec** (the
                      base-paper text methods).

Design choices, stated so they are not mistaken for the base paper:
  * Standard English stopwords come from gensim (already a dependency; no NLTK
    download). Legal-boilerplate stopword removal ("court", "appellant", ...) is
    OUR option and is OFF by default, so the Doc2Vec/TF-IDF baseline stays a
    faithful reproduction.
  * Stemming is available (gensim Porter) but OFF by default.
  * Header stripping (court name / date / cause-title / "delivered by" line) is
    conservative and only affects ``tokens`` — ``clean_text`` keeps everything.

No FastAPI / no ML-model dependency: importable and runnable from a plain script.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from gensim.parsing.preprocessing import STOPWORDS as _GENSIM_STOPWORDS
from gensim.parsing.porter import PorterStemmer

# --------------------------------------------------------------------------- #
# Stopword sets                                                               #
# --------------------------------------------------------------------------- #

# Standard English stopwords (frozen copy).
STOPWORDS: frozenset[str] = frozenset(_GENSIM_STOPWORDS)

# OUR optional list: high-frequency legal-boilerplate tokens that carry little
# discriminative signal between Indian SC judgments. OFF by default.
LEGAL_BOILERPLATE_STOPWORDS: frozenset[str] = frozenset(
    {
        "court", "judgment", "judgement", "appeal", "appeals", "appellant",
        "appellants", "respondent", "respondents", "petitioner", "petitioners",
        "learned", "counsel", "honble", "hon", "shri", "sri", "smt", "justice",
        "order", "dated", "supreme", "india", "indian", "case", "cases",
        "para", "paragraph", "judge", "judges", "bench", "matter",
    }
)

# --------------------------------------------------------------------------- #
# Regexes                                                                     #
# --------------------------------------------------------------------------- #

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
_MULTISPACE_RE = re.compile(r"[ \t ]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")
_PAGE_MARKER_RE = re.compile(r"^\s*(?:page\s+\d+(?:\s+of\s+\d+)?|-\s*\d+\s*-)\s*$", re.I | re.M)
_TOKEN_RE = re.compile(r"[a-z]+(?:['’][a-z]+)?")

# Body starts at the first "1." / "2." numbered paragraph, else after the
# "The Judgment was delivered by ..." line, else the whole text.
_NUM_PARA_START_RE = re.compile(r"^\s{0,3}(\d{1,3})\.\s+\S", re.M)
_DELIVERED_BY_RE = re.compile(r"^.*\bdelivered by\b.*$", re.I | re.M)
_PARA_SPLIT_RE = re.compile(r"^\s{0,3}\d{1,3}\.\s+", re.M)

# Sentence segmentation: candidate break after . ! ? then whitespace then an
# opening quote/paren/capital/digit. Merged back where the break follows a known
# abbreviation or a single-letter initial.
_SENT_CANDIDATE_RE = re.compile(r"(?<=[.!?])[\"'”’)\]]?\s+(?=[\"'“(\[A-Z0-9])")
_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "v", "vs", "no", "nos", "art", "arts", "s", "ss", "sec", "secs", "cl",
        "u/s", "r/w", "w.e.f", "i.e", "e.g", "viz", "etc", "pp", "p", "para",
        "paras", "ch", "sch", "ex", "vol", "ed", "rs", "air", "scc", "scr",
        "cri", "lj", "ilr", "mr", "mrs", "ms", "dr", "smt", "sri", "shri",
        "hon", "govt", "co", "ltd", "pvt", "corp", "ors", "anr", "st", "nd",
        "rd", "th", "jj", "j", "cj", "assn", "dept", "add", "distt", "dist",
        "u/art", "u/arts", "u/ss", "w.p", "s.l.p", "o.s", "f.a", "c.a", "cr.a",
        "c.c", "l.p.a", "r.f.a", "s.a", "o.p", "w.a", "no",
    }
)

# Multi-initial run ("C.C.", "S.L.P.", "A.I.R.") or slashed abbrev ("u/art.",
# "r/w.", "u/s.") immediately before a candidate break -> not a real boundary.
_TRAILING_INITIALS_RE = re.compile(r"(?:[A-Za-z]\.){2,}$|[A-Za-z]/[A-Za-z]+\.$")

_STEMMER = PorterStemmer()


# --------------------------------------------------------------------------- #
# Config + result types                                                       #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PreprocessConfig:
    min_token_len: int = 2
    remove_stopwords: bool = True
    remove_legal_boilerplate: bool = False  # OUR option, off by default
    stem: bool = False                      # off by default
    strip_header_for_tokens: bool = True

    def token_stopwords(self) -> frozenset[str]:
        stop: frozenset[str] = frozenset()
        if self.remove_stopwords:
            stop |= STOPWORDS
        if self.remove_legal_boilerplate:
            stop |= LEGAL_BOILERPLATE_STOPWORDS
        return stop


DEFAULT_CONFIG = PreprocessConfig()


@dataclass
class ProcessedDocument:
    doc_id: Optional[str]
    clean_text: str
    header: str
    paragraphs: list[str]
    sentences: list[str]
    tokens: list[str]
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------- #
# Cleaning                                                                    #
# --------------------------------------------------------------------------- #

def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # unify quote/dash variants that OCR / typesetting scatter around
    text = (
        text.replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
        .replace("–", "-").replace("—", "-").replace("−", "-")
        .replace(" ", " ").replace("﻿", "")
    )
    return _CONTROL_RE.sub("", text)


def clean_text(text: str) -> str:
    """Structural clean-up. Idempotent. Keeps case and punctuation."""
    text = normalize_unicode(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)   # join word-\n-word
    text = _PAGE_MARKER_RE.sub("", text)
    text = _MULTISPACE_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def split_header_body(text: str) -> tuple[str, str]:
    """Best-effort split into (header, body). Conservative: on failure the whole
    text is returned as body and the header is empty."""
    m = _NUM_PARA_START_RE.search(text)
    if m and int(m.group(1)) <= 2:
        return text[: m.start()].strip(), text[m.start():].strip()

    m = _DELIVERED_BY_RE.search(text)
    if m:
        return text[: m.end()].strip(), text[m.end():].strip()

    return "", text.strip()


def split_paragraphs(body: str) -> list[str]:
    """Split a judgment body on leading paragraph numbers ('1.', '2.', ...)."""
    if not body:
        return []
    parts = _PARA_SPLIT_RE.split(body)
    paras = [re.sub(r"\s+", " ", p).strip() for p in parts]
    return [p for p in paras if p]


# --------------------------------------------------------------------------- #
# Sentence segmentation                                                       #
# --------------------------------------------------------------------------- #

def _last_word(fragment: str) -> str:
    fragment = fragment.rstrip("\"')]}.!? ")
    m = re.search(r"([A-Za-z/][\w/.]*)$", fragment)
    return m.group(1).lower() if m else ""


def _ends_with_abbrev_or_initial(fragment: str) -> bool:
    stripped = fragment.rstrip("\"')]} ")
    if not stripped.endswith((".", "!", "?")):
        return True  # candidate break wasn't a real terminator -> keep merging
    word = _last_word(fragment)
    if word in _ABBREVIATIONS:
        return True
    # single-letter initial, e.g. "M." in "M. Patanjali Sastri, J."
    if re.search(r"(?:^|\s)[A-Za-z]\.$", stripped):
        return True
    if _TRAILING_INITIALS_RE.search(stripped):
        return True
    return False


def split_sentences(text: str) -> list[str]:
    """Abbreviation-aware sentence segmentation. Rule-based, no model download."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    raw_pieces = _SENT_CANDIDATE_RE.split(text)
    sentences: list[str] = []
    buffer = ""
    for piece in raw_pieces:
        buffer = f"{buffer} {piece}".strip() if buffer else piece
        if _ends_with_abbrev_or_initial(buffer):
            continue
        sentences.append(buffer)
        buffer = ""
    if buffer:
        sentences.append(buffer)
    return [s.strip() for s in sentences if s.strip()]


# --------------------------------------------------------------------------- #
# Tokenisation                                                                #
# --------------------------------------------------------------------------- #

def tokenize(
    text: str,
    *,
    config: PreprocessConfig = DEFAULT_CONFIG,
    stopwords: Optional[Iterable[str]] = None,
) -> list[str]:
    """Lowercase -> alphabetic tokens -> stopword/length filter -> (optional) stem.

    This is the bag-of-words input for TF-IDF and Doc2Vec (base-paper methods).
    """
    stop = frozenset(stopwords) if stopwords is not None else config.token_stopwords()
    tokens = _TOKEN_RE.findall(text.lower())
    out: list[str] = []
    for tok in tokens:
        if len(tok) < config.min_token_len:
            continue
        if tok in stop:
            continue
        if config.stem:
            tok = _STEMMER.stem(tok)
            if len(tok) < config.min_token_len or tok in stop:
                continue
        out.append(tok)
    return out


# --------------------------------------------------------------------------- #
# Top-level                                                                   #
# --------------------------------------------------------------------------- #

def preprocess_text(
    text: str,
    *,
    doc_id: Optional[str] = None,
    config: PreprocessConfig = DEFAULT_CONFIG,
) -> ProcessedDocument:
    cleaned = clean_text(text)
    header, body = split_header_body(cleaned)
    paragraphs = split_paragraphs(body) or (
        [re.sub(r"\s+", " ", body).strip()] if body else []
    )
    if paragraphs:
        sentences = [s for para in paragraphs for s in split_sentences(para)]
    else:
        sentences = split_sentences(cleaned)

    token_source = body if (config.strip_header_for_tokens and body) else cleaned
    tokens = tokenize(token_source, config=config)

    stats = {
        "n_chars_raw": len(text),
        "n_chars_clean": len(cleaned),
        "header_detected": bool(header),
        "n_paragraphs": len(paragraphs),
        "n_sentences": len(sentences),
        "n_tokens": len(tokens),
        "n_unique_tokens": len(set(tokens)),
    }
    return ProcessedDocument(
        doc_id=doc_id,
        clean_text=cleaned,
        header=header,
        paragraphs=paragraphs,
        sentences=sentences,
        tokens=tokens,
        stats=stats,
    )
