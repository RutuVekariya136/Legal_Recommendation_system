"""Statute-reference extraction (pipeline stage: Statute Extraction).

    Legal Case -> Citation Extraction -> [Statute Extraction] -> Hier-SPCNet -> ...

Unlike case citations (party names, unresolvable), statute references in these
judgments ARE reliably regex-extractable (CLAUDE.md §3), e.g.

    "Section 66(1) of the Indian Income-tax Act (XI of 1922)"
    "art. 226 of the Constitution"
    "s. 45 of the Act"            <- anaphoric: "the Act" = the judgment's primary Act

This module produces, per document:
  * a list of `StatuteReference` (section/article number, sub-clauses, Act),
  * the resolved Act ids (canonical, via a curated alias table),
  * the primary Act (to resolve anaphoric "the Act" / "the said Act"),
  * for Acts in the SAMPLE hierarchy, the Act -> Chapter/Part -> Section path.

The hierarchy is a small HAND-CURATED sample (`resources/statute_hierarchy.json`)
for frequently-cited Acts - NOT the paper's 12,814-Act tree, which is unpublished.
Any Act not in the sample still gets an Act-level node (`in_sample_hierarchy=False`),
never dropped or invented.

No FastAPI / model dependency.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

_RESOURCE = Path(__file__).parent / "resources" / "statute_hierarchy.json"

_ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(s: str) -> Optional[int]:
    s = s.upper()
    if not _ROMAN_RE.match(s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        val = _ROMAN_VALUES[ch]
        total += -val if val < prev else val
        prev = max(prev, val)
    return total


# --------------------------------------------------------------------------- #
# Hierarchy / Act registry                                                    #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ActMatch:
    act_id: str
    canonical: str
    year: Optional[int]
    in_sample_hierarchy: bool


class StatuteHierarchy:
    """Loads the curated sample hierarchy + alias table and resolves Act names."""

    def __init__(self, data: dict):
        self._meta = data.get("_meta", {})
        self._aliases: dict[str, str] = data.get("aliases", {})
        self._acts: dict[str, dict] = data.get("acts", {})
        # section -> (level_name, container_id, container_title)
        self._section_path: dict[tuple[str, str], tuple[str, str, str]] = {}
        for act_id, act in self._acts.items():
            containers = act.get("chapters") or act.get("parts") or {}
            level = "chapter" if "chapters" in act else ("part" if "parts" in act else "chapter")
            for cid, container in containers.items():
                for sec in (container.get("sections") or {}):
                    self._section_path[(act_id, str(sec))] = (
                        level, f"{act_id}::{level}::{cid}", container.get("title", cid)
                    )

    # -- meta -----------------------------------------------------------
    @property
    def meta(self) -> dict:
        return dict(self._meta)

    @property
    def curated_act_ids(self) -> set[str]:
        return set(self._acts)

    def has_act(self, act_id: str) -> bool:
        return act_id in self._acts

    def act_canonical(self, act_id: str) -> Optional[str]:
        act = self._acts.get(act_id)
        return act["canonical"] if act else None

    # -- resolution ---------------------------------------------------
    @staticmethod
    def _normalize_act_surface(surface: str) -> str:
        s = surface.lower().strip()
        s = s.replace("&", " and ")
        s = re.sub(r"[.,'’]", "", s)
        s = re.sub(r"^(?:of|the|and|for|on|said|principal)\s+", "", s).strip()
        s = re.sub(r"\bthe\b", " ", s)
        s = re.sub(r"\((?:act\s+)?[ivxlcdm0-9]+(?:\s+of\s+\d{4})?\)", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def resolve_act(self, surface: str, *, year: Optional[int] = None) -> Optional[ActMatch]:
        if not surface:
            return None
        norm = self._normalize_act_surface(surface)
        if not norm or norm in {"act", "said act", "principal act", "code", "said code"}:
            return None  # anaphoric - handled by the caller

        # exact / year-qualified alias
        keys_to_try = []
        if year:
            keys_to_try.append(f"{norm} {year}")
        keys_to_try.append(norm)
        # "indian xyz act" -> also try without a leading "indian"
        if norm.startswith("indian "):
            if year:
                keys_to_try.append(f"{norm[7:]} {year}")
            keys_to_try.append(norm[7:])

        for key in keys_to_try:
            act_id = self._aliases.get(key)
            if act_id:
                act = self._acts.get(act_id, {})
                return ActMatch(act_id, act.get("canonical", surface.strip()),
                                act.get("year", year), act_id in self._acts)

        # non-curated Act: keep it as a bare Act node, never drop it
        slug = re.sub(r"[^a-z0-9]+", "_", norm).strip("_")
        if year:
            slug = f"{slug}_{year}"
        canonical = re.sub(r"\s+", " ", surface.strip())
        return ActMatch(slug or "unknown_act", canonical, year, False)

    def section_path(self, act_id: str, section: str) -> list[tuple[str, str, str]]:
        """Return [(level, node_id, label), ...] from Act down to the section's
        container. Empty if the Act/section isn't in the sample hierarchy."""
        hit = self._section_path.get((act_id, str(section)))
        if not hit:
            return []
        level, node_id, title = hit
        return [("act", act_id, self.act_canonical(act_id) or act_id),
                (level, node_id, title)]


@lru_cache(maxsize=1)
def load_hierarchy() -> StatuteHierarchy:
    return StatuteHierarchy(json.loads(_RESOURCE.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Extraction                                                                  #
# --------------------------------------------------------------------------- #

_LEAD = r"(?:ss?|se[cn]s?|sections?|arts?|articles?|rr?|rules?|cls?|clauses?|paras?|paragraphs?|o|orders?)"
_NUM = r"\d+[A-Z]{0,2}"
_SUB = r"(?:\s*\(\s*[0-9a-zA-Z]{1,4}\s*\))*"
_ACT_WORD = r"[A-Z][A-Za-z'’.\-]+"
_ACT_INTERIOR = rf"(?:{_ACT_WORD}|of|the|and|for|&|de|on)"
_ACT_TAIL = r"(?:Act|Code|Constitution|Rules)"
_ACT_YEAR = (
    r"(?P<actyear>\s*[,(]?\s*(?:Act\s+)?(?:[IVXLCDM]+|No\.?\s*\d+)?\s*"
    r"(?:of\s+)?(?P<year>(?:1[6-9]|20)\d{2})?\s*\)?)"
)

_STATUTE_RE = re.compile(
    rf"\b(?P<lead>{_LEAD})\.?\s*"
    rf"(?P<num>{_NUM})(?P<sub>{_SUB})"
    rf"(?P<more>(?:\s*(?:,|and|&|to|read\s+with|r/?w)\s*{_NUM}{_SUB})*)"
    rf"(?:\s*[,]?\s*of\s+(?:the\s+)?"
    rf"(?P<act>(?:{_ACT_INTERIOR}\s+){{0,6}}{_ACT_TAIL}(?:\s+of\s+India)?)"
    rf"{_ACT_YEAR})?",
    re.IGNORECASE,
)
# Case-sensitive validation of the Act group: must contain >= 1 capitalised
# word before the Act/Code tail (so bare "of the Act" is flagged anaphoric).
_ACT_OK_RE = re.compile(
    rf"^(?:{_ACT_INTERIOR}\s+){{0,6}}{_ACT_TAIL}(?:\s+of\s+India)?$"
)
_ACT_HAS_NAME_RE = re.compile(_ACT_WORD)

# "the Bombay Sales Tax Act, 1952 (hereinafter referred to as 'the Act')"
_ACT_NAME_WORD = r"(?:[A-Z][A-Za-z'’.\-]+|of|the|and|for|&|on)"
_PRIMARY_DEF_RE = re.compile(
    rf"\b(?:[Tt]he\s+)?(?P<name>(?:{_ACT_NAME_WORD}\s+){{1,7}}(?:Act|Code|Ordinance|Regulation))"
    r"[,\s]*(?P<year>(?:1[6-9]|20)\d{2})?\s*"
    r"[,(\s]*\(?\s*(?i:hereinafter|hereafter)\b[^)]{0,45}?"
    r"(?i:the\s+)?(?i:said\s+)?(?i:act|code|ordinance|regulation|rules)\b",
)

_ANAPHORIC_ACTS = {"act", "the act", "said act", "the said act", "principal act",
                   "the principal act", "code", "the code", "said code",
                   "rules", "the rules", "said rules", "the said rules",
                   "act c", "act i"}
_LEAD_TO_KIND = {"art": "article", "arts": "article", "article": "article",
                 "articles": "article"}


@dataclass
class StatuteReference:
    raw: str
    kind: str                      # "section" | "article"
    number: str                    # "66", "286", "304A"
    subsections: list[str] = field(default_factory=list)
    also_numbers: list[str] = field(default_factory=list)
    act_surface: Optional[str] = None
    act_id: Optional[str] = None
    act_canonical: Optional[str] = None
    act_year: Optional[int] = None
    in_sample_hierarchy: bool = False
    anaphoric: bool = False
    char_offset: int = -1

    @property
    def section_key(self) -> Optional[str]:
        return f"{self.act_id}::s{self.number}" if self.act_id else None


def _parse_subs(sub_text: str) -> list[str]:
    return [s.strip() for s in re.findall(r"\(\s*([0-9a-zA-Z]{1,4})\s*\)", sub_text or "")]


def extract_statute_references(text: str, *, hierarchy: Optional[StatuteHierarchy] = None) -> list[StatuteReference]:
    hierarchy = hierarchy or load_hierarchy()
    refs: list[StatuteReference] = []

    for m in _STATUTE_RE.finditer(text):
        lead = m.group("lead").lower().rstrip(".")
        kind = _LEAD_TO_KIND.get(lead, "section")
        number = m.group("num").upper()
        subs = _parse_subs(m.group("sub"))
        more = [x.upper() for x in re.findall(_NUM, m.group("more") or "")]

        act_surface = (m.group("act") or "").strip() if m.group("act") else None
        year_str = m.group("year")
        act_year = int(year_str) if year_str else None

        anaphoric = False
        act_id = act_canon = None
        in_hier = False

        if act_surface and _ACT_OK_RE.match(act_surface):
            low = re.sub(r"\s+", " ", act_surface.lower()).strip()
            if low in _ANAPHORIC_ACTS or low.rstrip("s") in {"the act", "act"} or not _ACT_HAS_NAME_RE.search(act_surface):
                anaphoric = True
            else:
                match = hierarchy.resolve_act(act_surface, year=act_year)
                if match:
                    act_id, act_canon = match.act_id, match.canonical
                    act_year = match.year or act_year
                    in_hier = match.in_sample_hierarchy
        elif act_surface:
            act_surface = None

        # Bare "article N" with no Act -> Constitution of India (always, in this corpus)
        if kind == "article" and not act_id and not anaphoric:
            m2 = hierarchy.resolve_act("Constitution of India")
            if m2:
                act_id, act_canon, in_hier = m2.act_id, m2.canonical, m2.in_sample_hierarchy
                act_surface = act_surface or "Constitution"

        if not act_surface and not anaphoric and act_id is None:
            # a bare "s. 5" with no Act nearby and not an article: keep as
            # anaphoric so document-level resolution can attach the primary Act
            anaphoric = True

        raw = re.sub(r"\s+", " ", m.group(0)).strip()
        for i, num in enumerate([number, *more]):
            refs.append(
                StatuteReference(
                    raw=raw,
                    kind=kind, number=num, subsections=subs if i == 0 else [],
                    also_numbers=[],
                    act_surface=act_surface, act_id=act_id, act_canonical=act_canon,
                    act_year=act_year, in_sample_hierarchy=in_hier, anaphoric=anaphoric,
                    char_offset=m.start(),
                )
            )
    return refs


def find_primary_act_definition(text: str, *, hierarchy: Optional[StatuteHierarchy] = None,
                                window: int = 8000) -> Optional[ActMatch]:
    """Find an explicit '<X> Act, YYYY (hereinafter ... the Act)' definition -
    the statute under challenge, to which anaphoric 'the Act' refers."""
    hierarchy = hierarchy or load_hierarchy()
    m = _PRIMARY_DEF_RE.search(text[:window])
    if not m:
        return None
    name = re.sub(r"\s+", " ", m.group("name")).strip(" ,")
    year = int(m.group("year")) if m.group("year") else None
    return hierarchy.resolve_act(name, year=year)


# --------------------------------------------------------------------------- #
# Document-level resolution                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class DocumentStatutes:
    references: list[StatuteReference]
    primary_act_id: Optional[str]
    primary_act_canonical: Optional[str]
    act_ids: list[str]
    section_keys: list[str]
    n_anaphoric_resolved: int
    n_unresolved: int

    def hierarchy_paths(self, hierarchy: Optional[StatuteHierarchy] = None) -> dict[str, list]:
        hierarchy = hierarchy or load_hierarchy()
        out: dict[str, list] = {}
        for ref in self.references:
            if ref.act_id and ref.kind in {"section", "article"}:
                path = hierarchy.section_path(ref.act_id, ref.number)
                if path:
                    out[ref.section_key] = path
        return out


def resolve_document_statutes(text: str, *, hierarchy: Optional[StatuteHierarchy] = None) -> DocumentStatutes:
    hierarchy = hierarchy or load_hierarchy()
    refs = extract_statute_references(text, hierarchy=hierarchy)

    # Primary Act (target of anaphoric "the Act"):
    #  1. an explicit "(hereinafter ... the Act)" definition, else
    #  2. the most-mentioned named Act, preferring a substantive Act over the
    #     Constitution / General Clauses Act when a choice exists.
    primary_match = find_primary_act_definition(text, hierarchy=hierarchy)
    primary_act_id = primary_match.act_id if primary_match else None
    primary_canon = primary_match.canonical if primary_match else None

    if primary_act_id is None:
        named = Counter(r.act_id for r in refs if r.act_id and not r.anaphoric)
        _procedural = {"constitution_india", "general_clauses_act_1897"}
        substantive = Counter({k: v for k, v in named.items() if k not in _procedural})
        pick = substantive or named
        primary_act_id = pick.most_common(1)[0][0] if pick else None
        primary_canon = hierarchy.act_canonical(primary_act_id) if primary_act_id else None
        if primary_act_id and primary_canon is None:
            for r in refs:
                if r.act_id == primary_act_id and r.act_canonical:
                    primary_canon = r.act_canonical
                    break

    n_anaphoric_resolved = 0
    n_unresolved = 0
    for r in refs:
        if r.anaphoric:
            if primary_act_id:
                r.act_id = primary_act_id
                r.act_canonical = primary_canon
                r.in_sample_hierarchy = hierarchy.has_act(primary_act_id)
                n_anaphoric_resolved += 1
            else:
                n_unresolved += 1
        elif r.act_id is None:
            n_unresolved += 1

    act_ids = sorted({r.act_id for r in refs if r.act_id})
    section_keys = sorted({r.section_key for r in refs if r.section_key})
    return DocumentStatutes(
        references=refs,
        primary_act_id=primary_act_id,
        primary_act_canonical=primary_canon,
        act_ids=act_ids,
        section_keys=section_keys,
        n_anaphoric_resolved=n_anaphoric_resolved,
        n_unresolved=n_unresolved,
    )
