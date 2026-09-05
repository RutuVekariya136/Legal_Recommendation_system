"""Case-citation extraction and resolution (pipeline stage: Citation Extraction).

    Legal Case -> [Citation Extraction] -> Hier-SPCNet -> Network Embedding -> ...

A judgment cites earlier cases by **party name** ("Rangachari v. Secretary of
State"), sometimes followed by a law-reporter citation ("[1953] S.C.R. 589").
Our corpus document ids look like ``1971_138`` - there is **no lookup table**
from a party-name citation to a document id (CLAUDE.md §3). So resolution is:

  1. regex-extract "A v B" mentions (+ optional trailing reporter year),
  2. fuzzy-match the normalised "A v B" string against the cause-title of every
     document in our own corpus (rapidfuzz), optionally constrained by the
     reporter year.

This is deliberately conservative and WILL leave most mentions unresolved -
many cited cases are English/foreign or lower-court decisions that are simply
not in our ~295-document demo corpus. The resolution rate is reported, never
hidden or inflated. Statute references are handled separately in Step 7.

This module has no FastAPI / ML-model dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from rapidfuzz import fuzz, process

# --------------------------------------------------------------------------- #
# Extraction                                                                  #
# --------------------------------------------------------------------------- #

# "v" / "v." / "vs" / "vs." / "versus" standing between two parties.
_VS_RE = re.compile(r"\s+(?:v|vs|versus)\.?\s+", re.IGNORECASE)

# A name-ish token when walking outward from the "v": a capitalised word,
# an ALLCAPS word, an initial, or a lowercase connective we allow to stay
# *inside* a party name.
_NAME_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z.'’&()/-]*$")
_INNER_CONNECTORS = {"of", "the", "and", "&", "for", "de", "van", "von"}
# Lead-in words that precede a citation but are not part of it.
_LEADIN_WORDS = {
    "in", "see", "vide", "cf", "re", "per", "and", "also", "e.g", "eg", "i.e",
    "viz", "namely", "following", "approved", "considered", "distinguished",
    "overruled", "cited", "referred", "relied", "applied", "explained",
    "discussed", "quoted", "case", "decision", "judgment", "judgement",
    "ruling", "principle", "observations", "dictum", "held", "lord", "sir",
    "justice", "j", "jj", "cj", "mr", "mrs", "the", "of", "by", "to", "on",
    "was", "as", "is", "a", "an", "that", "this", "it", "court", "bench",
}
# Reporter citation shortly after party B -> gives the cited case's year.
# e.g. "[1953] S.C.R. 589", "(1962) 2 SCR 194", "1956 AIR(Andhra ...)"
_REPORTER_RE = re.compile(
    r"[\[(]?\s*((?:18|19|20)\d{2})\s*[\])]?\s*\(?\d*\)?\s*"
    r"(?:S\.?\s?C\.?\s?R|S\.?\s?C\.?\s?C|A\.?\s?I\.?\s?R|SCALE|"
    r"Cri\.?\s?L\.?\s?J|All\.?\s?E\.?\s?R|Q\.?\s?B|K\.?\s?B|"
    r"App\.?\s?Cas|W\.?\s?L\.?\s?R|Bom\.?\s?L\.?\s?R|I\.?\s?L\.?\s?R)",
    re.IGNORECASE,
)
# how far past the "v" to look for that reporter citation
_REPORTER_WINDOW = 140

_MAX_PARTY_TOKENS = 9
_MIN_PARTY_CHARS = 3
# A party made up ONLY of these (after dropping "the") is not a usable name:
# generic institutions, corporate suffixes, and citation debris.
_GENERIC_PARTY = {
    "state", "union", "government", "govt", "crown", "king", "queen", "republic",
    "emperor", "empress", "others", "other", "another", "anr", "ors", "it",
    "him", "her", "them", "its", "ltd", "limited", "co", "company", "private",
    "pvt", "corporation", "corpn", "messrs", "m/s", "workmen", "workman",
    "petitioner", "respondent", "appellant", "defendant", "plaintiff",
}


@dataclass(frozen=True)
class RawCaseCitation:
    raw: str
    party_a: str
    party_b: str
    reporter_year: Optional[int] = None
    char_offset: int = -1


def _clean_party(tokens: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(tokens)).strip(" .,&-")


def _walk_left(prefix: str) -> list[str]:
    # don't cross a sentence boundary into the previous sentence
    prefix = prefix.rsplit(". ", 1)[-1]
    words = prefix.split()
    picked: list[str] = []
    for w in reversed(words[-(_MAX_PARTY_TOKENS + 4):]):
        wl = w.lower().strip(".,;:")
        if _NAME_TOKEN_RE.match(w) and (w[:1].isupper() or wl in _INNER_CONNECTORS):
            picked.append(w)
            if len(picked) >= _MAX_PARTY_TOKENS:
                break
        else:
            break
    picked.reverse()
    # trim leading lower-case lead-in words ("in", "see", "per", "lord", ...)
    while picked and picked[0].lower().strip(".") in _LEADIN_WORDS:
        picked.pop(0)
    return picked


def _walk_right(suffix: str) -> list[str]:
    words = suffix.split()
    picked: list[str] = []
    for w in words[: _MAX_PARTY_TOKENS + 4]:
        core = w.strip(".,;:")
        if w[:1] in "([[" or core[:1].isdigit():
            break
        wl = core.lower()
        if _NAME_TOKEN_RE.match(core) and (core[:1].isupper() or wl in _INNER_CONNECTORS):
            picked.append(core)
            if wl in {"another", "others", "ors", "anr"}:
                break
            if len(picked) >= _MAX_PARTY_TOKENS:
                break
        else:
            break
    # a trailing bare connector is not part of the name
    while picked and picked[-1].lower() in _INNER_CONNECTORS:
        picked.pop()
    return picked


def _acceptable(party: str) -> bool:
    if len(party) < _MIN_PARTY_CHARS or not re.search(r"[A-Za-z]", party):
        return False
    content = [
        w for w in re.sub(r"[^a-z ]", " ", party.lower()).split()
        if w not in _GENERIC_PARTY and w not in _INNER_CONNECTORS
    ]
    # at least one real, distinctive token (>= 3 chars)
    return any(len(w) >= 3 for w in content)


def extract_case_citations(text: str) -> list[RawCaseCitation]:
    """Extract "A v B" case mentions from judgment text."""
    out: list[RawCaseCitation] = []
    for m in _VS_RE.finditer(text):
        left = _walk_left(text[: m.start()])
        right = _walk_right(text[m.end():])
        if not left or not right:
            continue
        a, b = _clean_party(left), _clean_party(right)
        if not (_acceptable(a) and _acceptable(b)):
            continue

        ry = None
        rm = _REPORTER_RE.search(text[m.end(): m.end() + _REPORTER_WINDOW])
        if rm:
            ry = int(rm.group(1))

        raw = f"{a} v {b}"
        out.append(
            RawCaseCitation(
                raw=raw, party_a=a, party_b=b, reporter_year=ry, char_offset=m.start()
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Normalisation                                                               #
# --------------------------------------------------------------------------- #

_DROP_WORDS_RE = re.compile(
    r"\b(?:the|of|m/?s|messrs|sri|shri|smt|mr|mrs|ms|dr|and\s+others?|and\s+another|"
    r"ors|anr|etc)\b",
    re.IGNORECASE,
)


def normalize_party(name: str) -> str:
    s = name.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[.'’]", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = _DROP_WORDS_RE.sub(" ", s)
    s = re.sub(r"\b(ltd|limited)\b", "ltd", s)
    s = re.sub(r"\b(co|company|corpn|corporation)\b", "co", s)
    s = re.sub(r"\bpvt\b", "private", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_citation(party_a: str, party_b: str) -> str:
    return f"{normalize_party(party_a)} v {normalize_party(party_b)}".strip()


_GENERIC_NORM = _GENERIC_PARTY | {"v"}


def _content_tokens(normalized_citation: str) -> list[str]:
    """Distinctive tokens (drop 'v', generic party words, very short tokens)."""
    return [
        t for t in normalized_citation.split()
        if len(t) >= 3 and t not in _GENERIC_NORM
    ]


def _split_title(title: str) -> Optional[tuple[str, str]]:
    parts = _VS_RE.split(title, maxsplit=1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        return None
    # party B often has a trailing citation / place - keep only the leading name-ish run
    b = parts[1].strip()
    b = re.split(r"[,\[(]| - |\d", b, maxsplit=1)[0].strip()
    return parts[0].strip(), b or parts[1].strip()


# --------------------------------------------------------------------------- #
# Resolution                                                                  #
# --------------------------------------------------------------------------- #

DEFAULT_MIN_SCORE = 90.0


@dataclass(frozen=True)
class ResolvedCitation:
    raw: str
    to_doc_id: str
    to_title: str
    score: float
    reporter_year: Optional[int] = None


@dataclass
class CorpusTitleIndex:
    """Fuzzy index of corpus cause-titles for citation resolution."""

    normalized: dict[str, str]              # doc_id -> "a v b" normalised
    titles: dict[str, str]                  # doc_id -> original title line
    years: dict[str, Optional[int]]         # doc_id -> year

    @classmethod
    def from_corpus(cls, corpus) -> "CorpusTitleIndex":
        normalized: dict[str, str] = {}
        titles: dict[str, str] = {}
        years: dict[str, Optional[int]] = {}
        for doc_id in corpus.doc_ids:
            doc = corpus[doc_id]
            if not doc.title:
                continue
            split = _split_title(doc.title)
            if not split:
                continue
            norm = normalize_citation(*split)
            if len(norm) < 8 or norm.startswith("v ") or norm.endswith(" v"):
                continue
            normalized[doc_id] = norm
            titles[doc_id] = doc.title
            years[doc_id] = doc.year
        return cls(normalized=normalized, titles=titles, years=years)

    def __len__(self) -> int:
        return len(self.normalized)

    def resolve(
        self,
        citation: RawCaseCitation,
        *,
        exclude_doc_id: Optional[str] = None,
        min_score: float = DEFAULT_MIN_SCORE,
        year_window: int = 1,
        not_after_year: Optional[int] = None,
    ) -> Optional[ResolvedCitation]:
        query = normalize_citation(citation.party_a, citation.party_b)
        # need at least two distinctive tokens, one of them >= 4 chars, so that
        # debris like "lal v state" or "ltd v workmen" cannot score 100 by being
        # a bare subset of some real title.
        content = _content_tokens(query)
        if len(content) < 2 or not any(len(t) >= 4 for t in content):
            return None

        candidates = {
            d: n for d, n in self.normalized.items() if d != exclude_doc_id
        }
        if not_after_year is not None:
            # a judgment cannot cite a case decided later than itself
            candidates = {
                d: n for d, n in candidates.items()
                if self.years.get(d) is None or self.years[d] <= not_after_year
            }
        if citation.reporter_year is not None:
            in_window = {
                d: n
                for d, n in candidates.items()
                if self.years.get(d) is not None
                and abs(self.years[d] - citation.reporter_year) <= year_window
            }
            if in_window:
                candidates = in_window

        if not candidates:
            return None

        match = process.extractOne(
            query,
            candidates,
            scorer=fuzz.token_set_ratio,
            score_cutoff=min_score,
        )
        if match is None:
            return None
        matched_norm, score, doc_id = match
        # secondary gate: order-sensitive similarity must also be reasonable,
        # and the matched title must share >= 2 distinctive tokens with the query.
        if fuzz.token_sort_ratio(query, matched_norm) < 60:
            return None
        shared = set(_content_tokens(query)) & set(_content_tokens(matched_norm))
        if len(shared) < 2:
            return None
        return ResolvedCitation(
            raw=citation.raw,
            to_doc_id=doc_id,
            to_title=self.titles[doc_id],
            score=float(score),
            reporter_year=citation.reporter_year,
        )


def resolve_citations(
    citations: Iterable[RawCaseCitation],
    index: CorpusTitleIndex,
    *,
    exclude_doc_id: Optional[str] = None,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[ResolvedCitation]:
    resolved: list[ResolvedCitation] = []
    for c in citations:
        r = index.resolve(c, exclude_doc_id=exclude_doc_id, min_score=min_score)
        if r is not None:
            resolved.append(r)
    return resolved


# --------------------------------------------------------------------------- #
# Corpus-wide edge building (consumed by Step 9 network construction)          #
# --------------------------------------------------------------------------- #

@dataclass
class CitationEdge:
    src_doc_id: str
    dst_doc_id: str
    score: float
    mentions: int = 1
    examples: list[str] = field(default_factory=list)


def build_citation_edges(
    corpus,
    *,
    index: Optional[CorpusTitleIndex] = None,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[CitationEdge]:
    """Extract + resolve case citations for every document; return case->case edges.

    Edges are directed (citing -> cited). Self-citations are dropped. When the
    same target is resolved from several mentions the best score is kept and the
    mention count accumulated.
    """
    index = index or CorpusTitleIndex.from_corpus(corpus)
    edges: dict[tuple[str, str], CitationEdge] = {}

    for src in corpus.doc_ids:
        src_year = corpus[src].year
        mentions = extract_case_citations(corpus.text(src))
        for cit in mentions:
            r = index.resolve(
                cit, exclude_doc_id=src, min_score=min_score, not_after_year=src_year
            )
            if r is None:
                continue
            key = (src, r.to_doc_id)
            edge = edges.get(key)
            if edge is None:
                edges[key] = CitationEdge(
                    src_doc_id=src,
                    dst_doc_id=r.to_doc_id,
                    score=r.score,
                    mentions=1,
                    examples=[cit.raw],
                )
            else:
                edge.mentions += 1
                edge.score = max(edge.score, r.score)
                if cit.raw not in edge.examples and len(edge.examples) < 5:
                    edge.examples.append(cit.raw)

    return list(edges.values())
