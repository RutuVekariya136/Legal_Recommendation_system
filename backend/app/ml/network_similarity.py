"""Network similarity (pipeline stage: Network Similarity).

    Network Embedding -> [Network Similarity] -> Hybrid Similarity

Turns the Hier-SPCNet embedding (Step 10) plus structural overlap into:

  * a single ``network_score(a, b)`` in [0, 1], and
  * a ``NetworkSimilarityBreakdown`` (shared citations, shared Acts, shared
    sections with ICF, co-citation count) that Step 14 explainability needs.

Two scoring methods:
  * ``"embedding"`` - the base-paper signal: Hier-SPCNet node cosine, only.
  * ``"blend"``     - OUR combination: embedding + ICF-weighted statute overlap +
    citation (bibliographic-coupling) overlap. Weights are configurable.

The breakdown lists are always computed regardless of scoring method.

No FastAPI / DB dependency.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from app.ml.citation_extraction import (
    CorpusTitleIndex,
    extract_case_citations,
    normalize_citation,
)
from app.ml.network_embedding import NetworkEmbeddingIndex
from app.ml.statute_extraction import load_hierarchy, resolve_document_statutes
from app.ml.text_similarity import Neighbour
from app.utils.paths import data_embeddings_dir, ensure_dir

METHODS = ("embedding", "blend")


@dataclass(frozen=True)
class BlendWeights:
    embedding: float = 0.60
    statute: float = 0.35
    citation: float = 0.05

    def normalised(self) -> "BlendWeights":
        s = self.embedding + self.statute + self.citation or 1.0
        return BlendWeights(self.embedding / s, self.statute / s, self.citation / s)


@dataclass
class DocFeatures:
    cited_doc_ids: set[str] = field(default_factory=set)
    raw_citations: set[str] = field(default_factory=set)
    act_ids: set[str] = field(default_factory=set)
    section_keys: set[str] = field(default_factory=set)


@dataclass
class NetworkSimilarityBreakdown:
    doc_id_1: str
    doc_id_2: str
    network_score: float
    embedding_score: float
    statute_overlap: float
    citation_overlap: float
    cocitation_count: int
    shared_acts: list[dict]
    shared_sections: list[dict]
    shared_citation_doc_ids: list[str]
    shared_raw_citations: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _weighted_jaccard(a: set[str], b: set[str], weights: dict[str, float]) -> float:
    if not a and not b:
        return 0.0
    inter = a & b
    union = a | b
    num = sum(weights.get(x, 1.0) for x in inter)
    den = sum(weights.get(x, 1.0) for x in union)
    return num / den if den else 0.0


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------- #
# Feature store                                                               #
# --------------------------------------------------------------------------- #

def extract_doc_features(
    text: str,
    *,
    title_index: CorpusTitleIndex,
    hierarchy=None,
    year: Optional[int] = None,
    self_doc_id: Optional[str] = None,
    min_citation_score: float = 90.0,
) -> DocFeatures:
    """Structural features for one document's text (works for an uploaded case
    that is not in the corpus)."""
    hierarchy = hierarchy or load_hierarchy()
    cited_ids: set[str] = set()
    raw: set[str] = set()
    for c in extract_case_citations(text):
        norm = normalize_citation(c.party_a, c.party_b)
        if len(norm) >= 10 and " v " in norm:
            raw.add(norm)
        r = title_index.resolve(
            c, exclude_doc_id=self_doc_id, min_score=min_citation_score, not_after_year=year
        )
        if r is not None and r.to_doc_id != self_doc_id:
            cited_ids.add(r.to_doc_id)

    ds = resolve_document_statutes(text, hierarchy=hierarchy)
    return DocFeatures(
        cited_doc_ids=cited_ids, raw_citations=raw,
        act_ids=set(ds.act_ids), section_keys=set(ds.section_keys),
    )


def build_feature_store(corpus, *, min_citation_score: float = 90.0) -> dict[str, DocFeatures]:
    title_index = CorpusTitleIndex.from_corpus(corpus)
    hierarchy = load_hierarchy()
    return {
        doc_id: extract_doc_features(
            corpus.text(doc_id), title_index=title_index, hierarchy=hierarchy,
            year=corpus[doc_id].year, self_doc_id=doc_id,
            min_citation_score=min_citation_score,
        )
        for doc_id in corpus.doc_ids
    }


def compute_statute_icf(store: dict[str, DocFeatures]) -> dict[str, float]:
    n = len(store) or 1
    df: dict[str, int] = {}
    for feats in store.values():
        for x in feats.act_ids | feats.section_keys:
            df[x] = df.get(x, 0) + 1
    return {x: math.log((n + 1) / (c + 1)) + 1.0 for x, c in df.items()}


# --------------------------------------------------------------------------- #
# Index                                                                       #
# --------------------------------------------------------------------------- #

class NetworkSimilarityIndex:
    def __init__(
        self,
        embedding_index: NetworkEmbeddingIndex,
        feature_store: dict[str, DocFeatures],
        *,
        method: str = "blend",
        weights: BlendWeights = BlendWeights(),
        corpus=None,
    ):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        self.embeddings = embedding_index
        self.features = feature_store
        self.method = method
        self.weights = weights.normalised()
        self.icf = compute_statute_icf(feature_store)
        self.doc_ids = list(embedding_index.doc_ids)
        self._corpus = corpus
        # co-citation lookup: doc_id -> set of docs that cite it
        self._cited_by: dict[str, set[str]] = {d: set() for d in self.doc_ids}
        for src, feats in feature_store.items():
            for dst in feats.cited_doc_ids:
                self._cited_by.setdefault(dst, set()).add(src)

    # -- build / persist --------------------------------------------
    @classmethod
    def build(cls, corpus, embedding_index: NetworkEmbeddingIndex, *,
              method: str = "blend", weights: BlendWeights = BlendWeights(),
              feature_store: Optional[dict[str, DocFeatures]] = None) -> "NetworkSimilarityIndex":
        store = feature_store or build_feature_store(corpus)
        return cls(embedding_index, store, method=method, weights=weights, corpus=corpus)

    def save(self, directory: Optional[Path] = None) -> Path:
        directory = ensure_dir(Path(directory) if directory
                               else data_embeddings_dir() / "network_similarity")
        payload = {
            "method": self.method,
            "weights": asdict(self.weights),
            "embedding_method": self.embeddings.method,
            "features": {
                d: {
                    "cited_doc_ids": sorted(f.cited_doc_ids),
                    "raw_citations": sorted(f.raw_citations),
                    "act_ids": sorted(f.act_ids),
                    "section_keys": sorted(f.section_keys),
                }
                for d, f in self.features.items()
            },
        }
        (directory / "feature_store.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: Path, embedding_index: NetworkEmbeddingIndex, corpus=None
             ) -> "NetworkSimilarityIndex":
        directory = Path(directory)
        payload = json.loads((directory / "feature_store.json").read_text(encoding="utf-8"))
        store = {
            d: DocFeatures(
                cited_doc_ids=set(v["cited_doc_ids"]), raw_citations=set(v["raw_citations"]),
                act_ids=set(v["act_ids"]), section_keys=set(v["section_keys"]),
            )
            for d, v in payload["features"].items()
        }
        w = payload.get("weights", {})
        return cls(embedding_index, store, method=payload.get("method", "blend"),
                   weights=BlendWeights(**w) if w else BlendWeights(), corpus=corpus)

    # -- scoring ---------------------------------------------------
    def embedding_score(self, a: str, b: str) -> float:
        return max(0.0, self.embeddings.pair_similarity(a, b))

    def _statute_overlap(self, fa: DocFeatures, fb: DocFeatures) -> float:
        return _weighted_jaccard(
            fa.act_ids | fa.section_keys, fb.act_ids | fb.section_keys, self.icf
        )

    def _citation_overlap(self, fa: DocFeatures, fb: DocFeatures) -> float:
        return _jaccard(
            fa.cited_doc_ids | fa.raw_citations, fb.cited_doc_ids | fb.raw_citations
        )

    def statute_overlap(self, a: str, b: str) -> float:
        return self._statute_overlap(self.features.get(a, DocFeatures()),
                                     self.features.get(b, DocFeatures()))

    def citation_overlap(self, a: str, b: str) -> float:
        return self._citation_overlap(self.features.get(a, DocFeatures()),
                                      self.features.get(b, DocFeatures()))

    def _combine(self, emb: float, stat: float, cite: float) -> float:
        if self.method == "embedding":
            return emb
        w = self.weights
        return max(0.0, min(1.0, w.embedding * emb + w.statute * stat + w.citation * cite))

    def similarity(self, a: str, b: str) -> float:
        return self._combine(
            self.embedding_score(a, b), self.statute_overlap(a, b), self.citation_overlap(a, b)
        )

    def breakdown(self, a: str, b: str) -> NetworkSimilarityBreakdown:
        fa, fb = self.features.get(a, DocFeatures()), self.features.get(b, DocFeatures())
        emb = self.embedding_score(a, b)
        stat = self.statute_overlap(a, b)
        cite = self.citation_overlap(a, b)

        shared_acts = sorted(fa.act_ids & fb.act_ids, key=lambda x: -self.icf.get(x, 1.0))
        shared_secs = sorted(fa.section_keys & fb.section_keys, key=lambda x: -self.icf.get(x, 1.0))

        acts_out = [{"act_id": aid, "icf": round(self.icf.get(aid, 1.0), 3)} for aid in shared_acts]
        secs_out = [
            {"section_key": s, "number": s.split("::s")[-1], "icf": round(self.icf.get(s, 1.0), 3)}
            for s in shared_secs
        ]
        cocit = len(self._cited_by.get(a, set()) & self._cited_by.get(b, set()))

        return NetworkSimilarityBreakdown(
            doc_id_1=a, doc_id_2=b,
            network_score=round(self._combine(emb, stat, cite), 4),
            embedding_score=round(emb, 4),
            statute_overlap=round(stat, 4),
            citation_overlap=round(cite, 4),
            cocitation_count=cocit,
            shared_acts=acts_out,
            shared_sections=secs_out,
            shared_citation_doc_ids=sorted(fa.cited_doc_ids & fb.cited_doc_ids),
            shared_raw_citations=sorted(fa.raw_citations & fb.raw_citations),
        )

    # -- ad-hoc query (uploaded doc, not in the corpus) -------------
    def query_breakdown(self, qfeat: DocFeatures, target: str, *,
                        embedding_score: float = 0.0) -> NetworkSimilarityBreakdown:
        fb = self.features.get(target, DocFeatures())
        emb = max(0.0, float(embedding_score))
        stat = self._statute_overlap(qfeat, fb)
        cite = self._citation_overlap(qfeat, fb)
        shared_acts = sorted(qfeat.act_ids & fb.act_ids, key=lambda x: -self.icf.get(x, 1.0))
        shared_secs = sorted(qfeat.section_keys & fb.section_keys, key=lambda x: -self.icf.get(x, 1.0))
        return NetworkSimilarityBreakdown(
            doc_id_1="<query>", doc_id_2=target,
            network_score=round(self._combine(emb, stat, cite), 4),
            embedding_score=round(emb, 4),
            statute_overlap=round(stat, 4),
            citation_overlap=round(cite, 4),
            cocitation_count=0,
            shared_acts=[{"act_id": a, "icf": round(self.icf.get(a, 1.0), 3)} for a in shared_acts],
            shared_sections=[
                {"section_key": s, "number": s.split("::s")[-1], "icf": round(self.icf.get(s, 1.0), 3)}
                for s in shared_secs
            ],
            shared_citation_doc_ids=sorted(qfeat.cited_doc_ids & fb.cited_doc_ids),
            shared_raw_citations=sorted(qfeat.raw_citations & fb.raw_citations),
        )

    def query_overlaps(self, qfeat: DocFeatures, target: str) -> tuple[float, float]:
        fb = self.features.get(target, DocFeatures())
        return self._statute_overlap(qfeat, fb), self._citation_overlap(qfeat, fb)

    def query_similarity(self, qfeat: DocFeatures, target: str, *, embedding_score: float = 0.0) -> float:
        stat, cite = self.query_overlaps(qfeat, target)
        return self._combine(max(0.0, embedding_score), stat, cite)

    def most_similar(self, doc_id: str, k: int = 10, *, exclude_self: bool = True) -> list[Neighbour]:
        scores = []
        for other in self.doc_ids:
            if exclude_self and other == doc_id:
                continue
            scores.append(Neighbour(other, self.similarity(doc_id, other)))
        scores.sort(key=lambda n: -n.score)
        return scores[:k]
