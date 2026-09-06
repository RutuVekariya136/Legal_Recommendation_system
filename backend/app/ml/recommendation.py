"""Recommendation engine (pipeline stage: Case Ranking -> Top-K Recommendation).

    Hybrid Similarity -> [Case Ranking -> Top-K Recommendation] -> Explanation ...

Given a query - an indexed corpus case OR raw text of an uploaded case - rank the
corpus by hybrid similarity, apply filters, and return the Top-K with a per-result
score breakdown and a short rule-based rationale.

For an uploaded case (not in Hier-SPCNet) there is no graph node, so its network
score uses **structural overlap only** (shared statutes / citations); the
embedding term is 0. This is stated in the result (`query_in_corpus=False`) and
never hidden.

No FastAPI / DB dependency - the engine is plain objects over the Step 8/11/12
artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from app.ml.citation_extraction import CorpusTitleIndex
from app.ml.hybrid_similarity import HybridFeatures, HybridSimilarityModel
from app.ml.network_similarity import extract_doc_features
from app.ml.statute_extraction import load_hierarchy


@dataclass
class Filters:
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    exclude_ids: set[str] = field(default_factory=set)
    min_hybrid_score: float = 0.0

    def allows(self, doc_id: str, year: Optional[int]) -> bool:
        if doc_id in self.exclude_ids:
            return False
        if self.year_min is not None and (year is None or year < self.year_min):
            return False
        if self.year_max is not None and (year is None or year > self.year_max):
            return False
        return True


@dataclass
class Recommendation:
    rank: int
    doc_id: str
    title: Optional[str]
    year: Optional[int]
    hybrid_score: float
    text_score: float
    network_score: float
    embedding_score: float
    statute_overlap: float
    citation_overlap: float
    shared_acts: list[dict]
    shared_sections: list[dict]
    shared_citations: list[str]
    cocitation_count: int
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RecommendationResult:
    query: str
    query_in_corpus: bool
    method: str
    k: int
    n_candidates: int
    n_filtered_out: int
    results: list[Recommendation]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d


def _rationale(text_score: float, bd_acts: list[dict], bd_secs: list[dict],
               shared_cites: list[str], cocit: int, emb: float) -> str:
    bits: list[str] = []
    if text_score >= 0.5:
        bits.append("strong textual overlap")
    elif text_score >= 0.25:
        bits.append("moderate textual overlap")
    else:
        bits.append("little textual overlap")

    if bd_secs:
        top = ", ".join(f"s.{s['number']}" for s in bd_secs[:3])
        bits.append(f"shares {len(bd_secs)} statute section(s) ({top})")
    elif bd_acts:
        bits.append(f"shares {len(bd_acts)} Act(s) ({', '.join(a['act_id'] for a in bd_acts[:3])})")

    if shared_cites:
        bits.append(f"{len(shared_cites)} common cited case(s)")
    if cocit:
        bits.append(f"co-cited by {cocit} case(s)")
    if emb >= 0.5:
        bits.append("close in the citation-network embedding")
    return "; ".join(bits) + "."


class RecommendationEngine:
    def __init__(self, corpus, hybrid_model: HybridSimilarityModel, *,
                 title_index: Optional[CorpusTitleIndex] = None, hierarchy=None,
                 text_ann=None, network_ann=None, candidate_pool: int = 300):
        self.corpus = corpus
        self.model = hybrid_model
        self.hierarchy = hierarchy or load_hierarchy()
        self.title_index = title_index or CorpusTitleIndex.from_corpus(corpus)
        # Optional ANN indexes: when present, only a candidate pool is hybrid-scored
        # instead of the whole corpus (Phase 2, P2).
        self.text_ann = text_ann
        self.network_ann = network_ann
        self.candidate_pool = candidate_pool

    @property
    def method(self) -> str:
        return self.model.combiner.name

    def _candidate_ids(self, doc_id: str) -> Optional[list[str]]:
        """ANN candidate pool for a corpus query, or None to score everything."""
        if not self.text_ann:
            return None
        pool: set[str] = set()
        if self.text_ann.has(doc_id):
            pool.update(n.doc_id for n in self.text_ann.most_similar(doc_id, self.candidate_pool))
        if self.network_ann and self.network_ann.has(doc_id):
            pool.update(n.doc_id for n in self.network_ann.most_similar(doc_id, self.candidate_pool))
        pool.discard(doc_id)
        return sorted(pool)

    # -- indexed corpus case --------------------------------------
    def recommend_for_case(self, doc_id: str, *, k: int = 10,
                           filters: Optional[Filters] = None) -> RecommendationResult:
        if doc_id not in self.corpus:
            raise KeyError(f"{doc_id!r} not in corpus")
        filters = filters or Filters()
        filters.exclude_ids = set(filters.exclude_ids) | {doc_id}

        text_scores = self.model.text_index.scores_for(doc_id)
        candidates = self._candidate_ids(doc_id) or self.corpus.doc_ids
        scored: list[tuple[float, str]] = []
        n_filtered = 0
        for other in candidates:
            if other == doc_id or other not in self.corpus:
                continue
            if not filters.allows(other, self.corpus[other].year):
                n_filtered += 1
                continue
            score = self.model.score(doc_id, other)
            if score < filters.min_hybrid_score:
                n_filtered += 1
                continue
            scored.append((score, other))

        scored.sort(key=lambda t: -t[0])
        results = [
            self._build_case_reco(rank, doc_id, other, score, text_scores.get(other, 0.0))
            for rank, (score, other) in enumerate(scored[:k], 1)
        ]
        return RecommendationResult(
            query=doc_id, query_in_corpus=True, method=self.method, k=k,
            n_candidates=len(scored), n_filtered_out=n_filtered, results=results,
        )

    def _build_case_reco(self, rank, query_id, target, score, text_score) -> Recommendation:
        bd = self.model.network_index.breakdown(query_id, target)
        doc = self.corpus[target]
        return Recommendation(
            rank=rank, doc_id=target, title=doc.title, year=doc.year,
            hybrid_score=round(score, 4),
            text_score=round(text_score, 4),
            network_score=bd.network_score, embedding_score=bd.embedding_score,
            statute_overlap=bd.statute_overlap, citation_overlap=bd.citation_overlap,
            shared_acts=bd.shared_acts, shared_sections=bd.shared_sections,
            shared_citations=bd.shared_citation_doc_ids, cocitation_count=bd.cocitation_count,
            rationale=_rationale(text_score, bd.shared_acts, bd.shared_sections,
                                 bd.shared_citation_doc_ids, bd.cocitation_count,
                                 bd.embedding_score),
        )

    # -- uploaded / ad-hoc text ---------------------------------
    def recommend_for_text(self, text: str, *, k: int = 10, year: Optional[int] = None,
                           filters: Optional[Filters] = None) -> RecommendationResult:
        filters = filters or Filters()
        qfeat = extract_doc_features(text, title_index=self.title_index,
                                     hierarchy=self.hierarchy, year=year)
        text_scores = self.model.text_index.query_scores(text)

        candidates = self.corpus.doc_ids
        if self.text_ann is not None and self.model.text_index.embedder is not None:
            from app.ml.preprocessing import preprocess_text

            pp = preprocess_text(text)
            qvec = self.model.text_index.embedder.embed_text(pp.tokens, pp.clean_text)
            candidates = [n.doc_id for n in self.text_ann.search(qvec, self.candidate_pool)]

        scored: list[tuple[float, str, float]] = []
        n_filtered = 0
        for other in candidates:
            if other not in self.corpus:
                continue
            if not filters.allows(other, self.corpus[other].year):
                n_filtered += 1
                continue
            t = text_scores.get(other, 0.0)
            stat, cite = self.model.network_index.query_overlaps(qfeat, other)
            net = self.model.network_index.query_similarity(qfeat, other, embedding_score=0.0)
            hyb = self.model.combiner.score(HybridFeatures(
                text_score=t, network_score=net, embedding_score=0.0,
                statute_overlap=stat, citation_overlap=cite, cocitation_count=0,
            ))
            if hyb < filters.min_hybrid_score:
                n_filtered += 1
                continue
            scored.append((hyb, other, t))

        scored.sort(key=lambda x: -x[0])
        results = []
        for rank, (hyb, other, t) in enumerate(scored[:k], 1):
            bd = self.model.network_index.query_breakdown(qfeat, other, embedding_score=0.0)
            doc = self.corpus[other]
            results.append(Recommendation(
                rank=rank, doc_id=other, title=doc.title, year=doc.year,
                hybrid_score=round(hyb, 4), text_score=round(t, 4),
                network_score=bd.network_score, embedding_score=bd.embedding_score,
                statute_overlap=bd.statute_overlap, citation_overlap=bd.citation_overlap,
                shared_acts=bd.shared_acts, shared_sections=bd.shared_sections,
                shared_citations=bd.shared_citation_doc_ids, cocitation_count=0,
                rationale=_rationale(t, bd.shared_acts, bd.shared_sections,
                                     bd.shared_citation_doc_ids, 0, 0.0),
            ))
        return RecommendationResult(
            query="<uploaded text>", query_in_corpus=False, method=self.method, k=k,
            n_candidates=len(scored), n_filtered_out=n_filtered, results=results,
        )

    # -- pairwise detail ------------------------------------
    def explain_pair(self, query_id: str, target_id: str) -> dict:
        return self.model.explain(query_id, target_id).__dict__
