"""Explainability (pipeline stage: Explanation).

    Top-K Recommendation -> [Explanation] + Case Comparison + Citation Graph

Turns a hybrid similarity result into a single frontend-ready object:

  * the three scores (text / network / hybrid) and the method,
  * **contribution attribution** - how much each component drove the hybrid
    score (weighted: alpha*text vs beta*network; learned-linear: coef*feature),
  * shared statutes with human labels (Act canonical name, section title from the
    curated hierarchy) and ICF,
  * shared cited cases resolved to corpus title + year,
  * shared raw (unresolved) citation strings + co-citation count,
  * a natural-language summary.

Everything is `@dataclass` with `to_dict()` (Pydantic schemas come in Step 18).
No FastAPI / DB dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from app.ml.hybrid_similarity import HybridExplanation, HybridSimilarityModel, LearnedCombiner, WeightedSumCombiner
from app.ml.network_similarity import extract_doc_features
from app.ml.statute_extraction import StatuteHierarchy, load_hierarchy


@dataclass
class ScoreContribution:
    component: str            # "text" | "network" | feature name | "intercept"
    value: float              # the raw feature value (0..1), or None for intercept
    weight: float             # alpha/beta, or learned coefficient
    contribution: float       # weight * value (signed)
    pct_of_score: float       # contribution / hybrid_score * 100 (clamped display value)


@dataclass
class SharedStatute:
    act_id: str
    act_name: Optional[str]
    kind: str                 # "act" | "section"
    section_number: Optional[str]
    section_title: Optional[str]
    icf: float
    in_curated_hierarchy: bool


@dataclass
class SharedCitedCase:
    doc_id: str
    title: Optional[str]
    year: Optional[int]


@dataclass
class SimilarityExplanation:
    query_id: str
    target_id: str
    query_in_corpus: bool
    method: str
    hybrid_score: float
    text_score: float
    network_score: float
    embedding_score: float
    statute_overlap: float
    citation_overlap: float
    contributions: list[ScoreContribution]
    shared_statutes: list[SharedStatute]
    shared_raw_citations: list[str]
    shared_cited_cases: list[SharedCitedCase]
    cocitation_count: int
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #

def _contributions(combiner, feats: dict) -> list[ScoreContribution]:
    """feats: {'text_score','network_score', ...} for the pair."""
    if isinstance(combiner, WeightedSumCombiner):
        raw = {
            "text": (feats["text_score"], combiner.alpha),
            "network": (feats["network_score"], combiner.beta),
        }
        total = sum(v * w for v, w in raw.values()) or 1.0
        return [
            ScoreContribution(name, round(v, 4), round(w, 4), round(v * w, 4),
                              round(v * w / total * 100, 1))
            for name, (v, w) in raw.items()
        ]

    if isinstance(combiner, LearnedCombiner) and combiner.model_kind == "linear" and combiner._fitted:
        cfg = combiner.config()
        coefs = cfg.get("coef_", [])
        intercept = cfg.get("intercept_", 0.0)
        parts = [
            ScoreContribution(fname, round(feats.get(fname, 0.0), 4), round(c, 4),
                              round(c * feats.get(fname, 0.0), 4), 0.0)
            for fname, c in zip(combiner.features, coefs)
        ]
        parts.append(ScoreContribution("intercept", None, round(intercept, 4),
                                       round(intercept, 4), 0.0))
        total = sum(p.contribution for p in parts) or 1.0
        for p in parts:
            p.pct_of_score = round(p.contribution / total * 100, 1)
        return parts

    # MLP or unfitted: report the inputs, no attribution
    return [
        ScoreContribution("text", round(feats["text_score"], 4), 0.0, 0.0, 0.0),
        ScoreContribution("network", round(feats["network_score"], 4), 0.0, 0.0, 0.0),
    ]


def _shared_statutes(hierarchy: StatuteHierarchy, bd: dict) -> list[SharedStatute]:
    out: list[SharedStatute] = []
    for sec in bd.get("shared_sections", []):
        key = sec["section_key"]
        act_id = key.split("::s")[0]
        num = sec.get("number") or key.split("::s")[-1]
        out.append(SharedStatute(
            act_id=act_id, act_name=hierarchy.act_canonical(act_id),
            kind="section", section_number=num,
            section_title=hierarchy.section_title(act_id, num),
            icf=sec.get("icf", 1.0), in_curated_hierarchy=hierarchy.has_act(act_id),
        ))
    section_acts = {s.act_id for s in out}
    for act in bd.get("shared_acts", []):
        aid = act["act_id"]
        if aid in section_acts:
            continue  # already represented by a shared section
        out.append(SharedStatute(
            act_id=aid, act_name=hierarchy.act_canonical(aid), kind="act",
            section_number=None, section_title=None, icf=act.get("icf", 1.0),
            in_curated_hierarchy=hierarchy.has_act(aid),
        ))
    return out


def _summary(ex: "SimilarityExplanation") -> str:
    parts: list[str] = []
    dom = max(ex.contributions, key=lambda c: c.contribution, default=None)
    if ex.text_score >= 0.5 and ex.network_score >= 0.4:
        parts.append("Both the text content and the citation/statute network point to a close match")
    elif ex.text_score >= 0.5:
        parts.append("The match is driven mainly by similar text content")
    elif ex.network_score >= 0.4:
        parts.append("The match is driven mainly by the citation/statute network")
    else:
        parts.append("A weak-to-moderate match")

    if ex.shared_statutes:
        labelled = [
            (f"{s.act_name or s.act_id} s.{s.section_number}"
             + (f" ({s.section_title})" if s.section_title else ""))
            if s.kind == "section" else (s.act_name or s.act_id)
            for s in ex.shared_statutes[:3]
        ]
        parts.append("shared provisions: " + "; ".join(labelled))
    if ex.shared_cited_cases:
        parts.append(f"{len(ex.shared_cited_cases)} common cited case(s)")
    if ex.cocitation_count:
        parts.append(f"co-cited by {ex.cocitation_count} other case(s)")
    return ". ".join(parts) + "."


class Explainer:
    def __init__(self, corpus, hybrid_model: HybridSimilarityModel, *,
                 hierarchy: Optional[StatuteHierarchy] = None, title_index=None):
        self.corpus = corpus
        self.model = hybrid_model
        self.hierarchy = hierarchy or load_hierarchy()
        self._title_index = title_index

    @property
    def title_index(self):
        if self._title_index is None:
            from app.ml.citation_extraction import CorpusTitleIndex

            self._title_index = CorpusTitleIndex.from_corpus(self.corpus)
        return self._title_index

    # -- indexed pair -------------------------------------------
    def explain_pair(self, query_id: str, target_id: str) -> SimilarityExplanation:
        hx: HybridExplanation = self.model.explain(query_id, target_id)
        bd = hx.network_breakdown
        feats = {
            "text_score": hx.text_score, "network_score": hx.network_score,
            "embedding_score": bd["embedding_score"], "statute_overlap": bd["statute_overlap"],
            "citation_overlap": bd["citation_overlap"], "cocitation_count": bd["cocitation_count"],
        }
        return self._assemble(query_id, target_id, True, hx.method, hx.hybrid_score, feats, bd)

    # -- uploaded text -------------------------------------------
    def explain_text(self, text: str, target_id: str, *, year: Optional[int] = None
                     ) -> SimilarityExplanation:
        qfeat = extract_doc_features(text, title_index=self.title_index,
                                     hierarchy=self.hierarchy, year=year)
        ni = self.model.network_index
        bd = ni.query_breakdown(qfeat, target_id, embedding_score=0.0).to_dict()
        text_score = self.model.text_index.query_scores(text).get(target_id, 0.0)
        from app.ml.hybrid_similarity import HybridFeatures

        hyb = self.model.combiner.score(HybridFeatures(
            text_score=text_score, network_score=bd["network_score"], embedding_score=0.0,
            statute_overlap=bd["statute_overlap"], citation_overlap=bd["citation_overlap"],
            cocitation_count=0,
        ))
        feats = {
            "text_score": text_score, "network_score": bd["network_score"],
            "embedding_score": 0.0, "statute_overlap": bd["statute_overlap"],
            "citation_overlap": bd["citation_overlap"], "cocitation_count": 0,
        }
        return self._assemble("<uploaded>", target_id, False, self.model.combiner.name,
                              round(hyb, 4), feats, bd)

    # -- shared assembly --------------------------------------
    def _assemble(self, query_id, target_id, in_corpus, method, hybrid_score, feats, bd
                  ) -> SimilarityExplanation:
        cited = []
        for did in bd.get("shared_citation_doc_ids", []):
            doc = self.corpus[did] if did in self.corpus else None
            cited.append(SharedCitedCase(did, getattr(doc, "title", None),
                                         getattr(doc, "year", None)))
        ex = SimilarityExplanation(
            query_id=query_id, target_id=target_id, query_in_corpus=in_corpus, method=method,
            hybrid_score=round(hybrid_score, 4),
            text_score=round(feats["text_score"], 4),
            network_score=round(feats["network_score"], 4),
            embedding_score=round(feats["embedding_score"], 4),
            statute_overlap=round(feats["statute_overlap"], 4),
            citation_overlap=round(feats["citation_overlap"], 4),
            contributions=_contributions(self.model.combiner, feats),
            shared_statutes=_shared_statutes(self.hierarchy, bd),
            shared_raw_citations=list(bd.get("shared_raw_citations", [])),
            shared_cited_cases=cited,
            cocitation_count=int(feats["cocitation_count"]),
            summary="",
        )
        ex.summary = _summary(ex)
        return ex
