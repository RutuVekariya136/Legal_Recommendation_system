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
from app.ml.explainability import Explainer
from app.ml.hybrid_similarity import HybridSimilarityModel, LearnedCombiner, make_combiner
from app.ml.network_embedding import NetworkEmbedder, NetworkEmbeddingIndex
from app.ml.network_model import build_hier_spcnet, from_node_link
from app.ml.network_similarity import NetworkSimilarityIndex
from app.ml.recommendation import Filters, RecommendationEngine
from app.ml.text_similarity import TextSimilarityIndex, TfidfEmbedder
from app.utils.dataset import load_corpus
from app.utils.paths import data_embeddings_dir, data_graphs_dir


@dataclass
class ServiceMeta:
    text_method: str
    network_method: str
    hybrid_method: str
    n_cases: int
    rebuilt: list[str] = field(default_factory=list)
    load_seconds: float = 0.0


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

        engine = RecommendationEngine(corpus, hybrid)
        explainer = Explainer(corpus, hybrid, title_index=engine.title_index)
        meta = ServiceMeta(
            text_method=text_index.method, network_method=net_emb.method,
            hybrid_method=hybrid.combiner.name, n_cases=len(corpus),
            rebuilt=rebuilt, load_seconds=round(time.time() - t0, 1),
        )
        return cls(corpus, engine, explainer, meta)

    # -- operations ----------------------------------------------
    def recommend_case(self, doc_id: str, *, k: int, filters: Optional[Filters] = None):
        return self.engine.recommend_for_case(doc_id, k=k, filters=filters)

    def recommend_text(self, text: str, *, k: int, year: Optional[int] = None,
                       filters: Optional[Filters] = None):
        return self.engine.recommend_for_text(text, k=k, year=year, filters=filters)

    def explain_pair(self, a: str, b: str):
        return self.explainer.explain_pair(a, b)

    def explain_text(self, text: str, target: str, *, year: Optional[int] = None):
        return self.explainer.explain_text(text, target, year=year)


@lru_cache
def get_service() -> RecommenderService:
    return RecommenderService.load_default()
