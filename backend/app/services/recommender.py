"""RecommenderService - loads the ML pipeline once and serves the routers.

Holds the corpus, the RecommendationEngine, and the Explainer. Built from the
persisted artifacts in data/embeddings/ when present, otherwise rebuilt on the
fly (slower first start; run scripts/build_all.py to pre-build).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from app.config import get_settings
from app.ml.case_treatment import TreatmentSummary, build_treatment_index
from app.ml.citation_extraction import build_citation_edges
from app.ml.evaluation import ReliabilityCalibrator
from app.ml.explainability import Explainer
from app.ml.hybrid_similarity import HybridSimilarityModel, LearnedCombiner, make_combiner
from app.ml.network_embedding import NetworkEmbedder, NetworkEmbeddingIndex
from app.ml.network_model import build_hier_spcnet, from_node_link
from app.ml.network_similarity import NetworkSimilarityIndex
from app.ml.recommendation import Filters, RecommendationEngine
from app.ml.text_similarity import TextSimilarityIndex, TfidfEmbedder
from app.ml.vector_index import VectorIndex, build_from_index
from app.utils.dataset import load_corpus
from app.utils.paths import data_embeddings_dir, data_graphs_dir, data_indexes_dir


@dataclass
class ServiceMeta:
    text_method: str
    network_method: str
    hybrid_method: str
    n_cases: int
    rebuilt: list[str] = field(default_factory=list)
    load_seconds: float = 0.0
    ann_backend: str = "numpy"


class RecommenderService:
    def __init__(self, corpus, engine: RecommendationEngine, explainer: Explainer,
                 meta: ServiceMeta):
        self.corpus = corpus
        self.engine = engine
        self.explainer = explainer
        self.meta = meta

    # -- construction --------------------------------------------
    @classmethod
    def load_default(cls) -> "RecommenderService":
        s = get_settings()
        t0 = time.time()
        rebuilt: list[str] = []
        corpus = load_corpus(verify=True)
        emb_dir = data_embeddings_dir()

        # text index
        tdir = emb_dir / s.text_similarity_method
        text_index = None
        if (tdir / "index.json").is_file():
            text_index = TextSimilarityIndex.load(tdir)
        if text_index is None or text_index.embedder is None:
            rebuilt.append("text_index")
            text_index = TextSimilarityIndex.build(corpus, TfidfEmbedder())

        # network embedding
        ndir = emb_dir / f"network_{s.network_embedding_method}"
        if (ndir / "index.json").is_file():
            net_emb = NetworkEmbeddingIndex.load(ndir)
        else:
            rebuilt.append("network_embedding")
            gpath = data_graphs_dir() / "hier_spcnet.json"
            graph = (from_node_link(__import__("json").loads(gpath.read_text(encoding="utf-8")))
                     if gpath.is_file() else build_hier_spcnet(corpus, min_score=90))
            net_emb = NetworkEmbeddingIndex.build(
                graph, corpus, NetworkEmbedder(method=s.network_embedding_method)
            )

        # network similarity (feature store)
        nsdir = emb_dir / "network_similarity"
        if (nsdir / "feature_store.json").is_file():
            net_sim = NetworkSimilarityIndex.load(nsdir, net_emb, corpus=corpus)
        else:
            rebuilt.append("network_similarity")
            net_sim = NetworkSimilarityIndex.build(corpus, net_emb, method="blend")

        # hybrid
        hdir = emb_dir / "hybrid"
        if (hdir / "config.json").is_file():
            hybrid = HybridSimilarityModel.load(hdir, text_index, net_sim)
            if isinstance(hybrid.combiner, LearnedCombiner) and not hybrid.combiner._fitted:
                hybrid.fit_learned(corpus, split="validation")
        else:
            rebuilt.append("hybrid")
            combiner = (make_combiner("weighted", alpha=s.hybrid_alpha, beta=s.hybrid_beta)
                        if s.hybrid_method == "weighted" else LearnedCombiner(model="linear"))
            hybrid = HybridSimilarityModel(text_index, net_sim, combiner)
            if isinstance(combiner, LearnedCombiner):
                hybrid.fit_learned(corpus, split="validation")

        # ANN indexes for candidate generation (Phase 2, P2). Small corpora fall
        # back to exact flat search; large ones use FAISS-HNSW if faiss-cpu is present.
        idx_dir = data_indexes_dir()
        text_ann = cls._load_or_build_ann(idx_dir / "text", text_index)
        net_ann = cls._load_or_build_ann(idx_dir / "network", net_emb)

        # Reliability calibration (optional - built by scripts/evaluate.py from
        # the 100 validation pairs; absent until that's been run at least once).
        cal_path = hdir / "calibration.joblib"
        reliability = ReliabilityCalibrator.load(cal_path) if cal_path.is_file() else None

        # Citation-treatment signal ("is this case still good law?"). Cheap to
        # compute fresh at startup (only the resolved edges, ~1.2% of raw
        # mentions) rather than coupling to the persisted graph JSON.
        citation_edges = build_citation_edges(corpus, min_score=90)
        treatment_index = build_treatment_index(citation_edges, corpus=corpus)

        engine = RecommendationEngine(corpus, hybrid, text_ann=text_ann, network_ann=net_ann,
                                      reliability=reliability, treatment_index=treatment_index)
        explainer = Explainer(corpus, hybrid, title_index=engine.title_index,
                              reliability=reliability)
        meta = ServiceMeta(
            text_method=text_index.method, network_method=net_emb.method,
            hybrid_method=hybrid.combiner.name, n_cases=len(corpus),
            rebuilt=rebuilt, load_seconds=round(time.time() - t0, 1),
        )
        meta.ann_backend = text_ann.backend
        return cls(corpus, engine, explainer, meta)

    @staticmethod
    def _load_or_build_ann(directory, sim_index) -> VectorIndex:
        if (directory / "meta.json").is_file():
            try:
                loaded = VectorIndex.load(directory)
                if loaded.doc_ids == list(sim_index.doc_ids):
                    return loaded
            except Exception:  # noqa: BLE001 - stale/corrupt index -> rebuild
                pass
        return build_from_index(sim_index)

    # -- operations ----------------------------------------------
    def recommend_case(self, doc_id: str, *, k: int, filters: Optional[Filters] = None):
        return self.engine.recommend_for_case(doc_id, k=k, filters=filters)

    def recommend_text(self, text: str, *, k: int, year: Optional[int] = None,
                       filters: Optional[Filters] = None,
                       category: Optional[str] = None, category_query: Optional[str] = None,
                       details: Optional[dict] = None, detail_bonus: float = 0.0):
        return self.engine.recommend_for_text(text, k=k, year=year, filters=filters,
                                              category=category, category_query=category_query,
                                              details=details, detail_bonus=detail_bonus)

    def explain_pair(self, a: str, b: str):
        return self.explainer.explain_pair(a, b)

    def treatment_for(self, doc_id: str) -> Optional[TreatmentSummary]:
        return self.engine.treatment_index.get(doc_id)

    # -- search --------------------------------------------------
    def search(self, query: str, *, mode: str = "semantic", year_min: Optional[int] = None,
               year_max: Optional[int] = None, split: Optional[str] = None,
               limit: int = 20, offset: int = 0, min_score: float = 0.02) -> dict:
        ti = self.engine.model.text_index
        sem = ti.query_scores(query) if mode in ("semantic", "hybrid") else {}
        ql = query.lower().strip()

        hits: list[tuple[float, str, bool]] = []
        for doc_id in self.corpus.doc_ids:
            doc = self.corpus[doc_id]
            if year_min is not None and (doc.year is None or doc.year < year_min):
                continue
            if year_max is not None and (doc.year is None or doc.year > year_max):
                continue
            if split is not None and split not in getattr(doc, "splits", ()):
                continue

            title_hit = bool(doc.title and ql in doc.title.lower())
            if mode == "title":
                if not title_hit:
                    continue
                score = 1.0
            elif mode == "semantic":
                score = sem.get(doc_id, 0.0)
                if score < min_score:
                    continue
            else:  # hybrid
                score = 0.8 * sem.get(doc_id, 0.0) + (0.2 if title_hit else 0.0)
                if score < min_score and not title_hit:
                    continue
            hits.append((round(score, 4), doc_id, title_hit))

        hits.sort(key=lambda h: -h[0])
        page = hits[offset:offset + limit]
        items = [
            {
                "doc_id": doc_id,
                "title": self.corpus[doc_id].title,
                "year": self.corpus[doc_id].year,
                "splits": list(getattr(self.corpus[doc_id], "splits", ())),
                "score": score,
                "title_match": title_hit,
                "snippet": self._snippet(doc_id, ql),
            }
            for score, doc_id, title_hit in page
        ]
        return {"query": query, "mode": mode, "total": len(hits),
                "limit": limit, "offset": offset, "items": items}

    # -- ego graph (Hier-SPCNet neighbourhood of a case) -----------
    def ego_graph(self, doc_id: str, *, n_similar: int = 6, n_statutes: int = 8) -> dict:
        if doc_id not in self.corpus:
            raise KeyError(doc_id)
        ni = self.engine.model.network_index
        feats = ni.features.get(doc_id)
        center = f"case:{doc_id}"
        nodes = {center: {"id": center, "label": self.corpus[doc_id].title or doc_id,
                          "type": "case", "center": True,
                          "year": self.corpus[doc_id].year}}
        edges = []

        hier = self.explainer.hierarchy

        # statutes this case cites: prefer ones with a curated label, then by ICF
        def _stat_key(k: str):
            act_id = k.split("::s")[0]
            num = k.split("::s")[-1] if "::s" in k else None
            has_label = bool(num and hier.section_title(act_id, num)) or (
                "::s" not in k and hier.has_act(act_id)
            )
            return (not has_label, -ni.icf.get(k, 1.0))

        stat_keys = sorted(
            list((feats.act_ids if feats else set())) +
            list((feats.section_keys if feats else set())),
            key=_stat_key,
        )[:n_statutes]
        for k in stat_keys:
            is_section = "::s" in k
            act_id = k.split("::s")[0]
            nid = f"stat:{k}"
            label = (f"{hier.act_canonical(act_id) or act_id} s.{k.split('::s')[-1]}"
                     if is_section else (hier.act_canonical(act_id) or act_id))
            nodes[nid] = {"id": nid, "label": label,
                          "type": "section" if is_section else "act",
                          "icf": round(ni.icf.get(k, 1.0), 2)}
            edges.append({"source": center, "target": nid, "type": "cites", "weight": ni.icf.get(k, 1.0)})

        # most similar cases, and which of the above statutes they share
        for nb in self.engine.recommend_for_case(doc_id, k=n_similar).results:
            nid = f"case:{nb.doc_id}"
            nodes[nid] = {"id": nid, "label": nb.title or nb.doc_id, "type": "case",
                          "year": nb.year, "score": nb.hybrid_score}
            edges.append({"source": center, "target": nid, "type": "similar",
                          "weight": nb.hybrid_score})
            nbf = ni.features.get(nb.doc_id)
            if nbf:
                shared = (nbf.act_ids | nbf.section_keys)
                for k in stat_keys:
                    if k in shared:
                        edges.append({"source": nid, "target": f"stat:{k}",
                                      "type": "cites", "weight": ni.icf.get(k, 1.0)})

        return {"center": center, "nodes": list(nodes.values()), "edges": edges}

    def _snippet(self, doc_id: str, query_lower: str, *, width: int = 240) -> str:
        import re

        text = re.sub(r"\s+", " ", self.corpus.text(doc_id)).strip()
        terms = [t for t in re.findall(r"[a-z]{3,}", query_lower)]
        best_i, best_hits = 0, -1
        sentences = re.split(r"(?<=[.!?]) ", text[:8000])
        for i, s in enumerate(sentences):
            sl = s.lower()
            h = sum(sl.count(t) for t in terms)
            if h > best_hits:
                best_hits, best_i = h, i
        snip = sentences[best_i] if sentences else text
        return (snip[:width] + "…") if len(snip) > width else snip

    def explain_text(self, text: str, target: str, *, year: Optional[int] = None):
        return self.explainer.explain_text(text, target, year=year)


@lru_cache
def get_service() -> RecommenderService:
    return RecommenderService.load_default()
