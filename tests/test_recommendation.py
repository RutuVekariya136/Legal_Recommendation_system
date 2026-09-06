"""Tests for app.ml.recommendation (Step 13)."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.hybrid_similarity import HybridSimilarityModel, WeightedSumCombiner
from app.ml.network_embedding import NetworkEmbeddingIndex
from app.ml.network_similarity import DocFeatures, NetworkSimilarityIndex
from app.ml.recommendation import Filters, RecommendationEngine
from app.ml.text_similarity import TextDoc, TextSimilarityIndex, TfidfEmbedder


class _Doc:
    def __init__(self, doc_id, year, text):
        self.doc_id, self.year, self._text = doc_id, year, text
        self.title = f"{doc_id} v State"


class _Corpus:
    def __init__(self, docs):
        self._d = {d.doc_id: d for d in docs}

    @property
    def doc_ids(self):
        return sorted(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, k):
        return self._d[k]

    def __contains__(self, k):
        return k in self._d

    def text(self, k):
        return self._d[k]._text


@pytest.fixture
def engine():
    docs = [
        _Doc("tax_1980", 1980, "s. 66 of the Indian Income-tax Act, 1922 assessment revenue levy"),
        _Doc("tax_1995", 1995, "s. 66 of the Indian Income-tax Act, 1922 assessment revenue return"),
        _Doc("crime_1975", 1975, "s. 302 of the Indian Penal Code murder accused evidence"),
        _Doc("crime_2005", 2005, "s. 302 of the Indian Penal Code murder witness conviction"),
    ]
    corpus = _Corpus(docs)

    from app.ml.preprocessing import preprocess_text

    te = TfidfEmbedder(min_df=1)
    te.fit([  # MUST be in corpus.doc_ids order to match the index labels
        TextDoc(i, preprocess_text(corpus.text(i)).tokens, corpus.text(i))
        for i in corpus.doc_ids
    ])
    ti = TextSimilarityIndex("tfidf", corpus.doc_ids, te.embed_corpus(), embedder=te)

    ids = corpus.doc_ids
    mat = np.eye(len(ids), dtype=np.float32)  # embeddings not the point here
    ei = NetworkEmbeddingIndex("metapath2vec", ids, mat)
    store = {
        "tax_1980": DocFeatures(act_ids={"income_tax_act_1922"}, section_keys={"income_tax_act_1922::s66"}),
        "tax_1995": DocFeatures(act_ids={"income_tax_act_1922"}, section_keys={"income_tax_act_1922::s66"}),
        "crime_1975": DocFeatures(act_ids={"ipc_1860"}, section_keys={"ipc_1860::s302"}),
        "crime_2005": DocFeatures(act_ids={"ipc_1860"}, section_keys={"ipc_1860::s302"}),
    }
    ni = NetworkSimilarityIndex(ei, store, method="blend")
    model = HybridSimilarityModel(ti, ni, WeightedSumCombiner(0.5, 0.5))
    return RecommendationEngine(corpus, model)


# -- filters --------------------------------------------------------

def test_filters_year_and_exclude():
    f = Filters(year_min=1990, year_max=2000, exclude_ids={"x"})
    assert f.allows("a", 1995)
    assert not f.allows("a", 1980)
    assert not f.allows("a", 2010)
    assert not f.allows("x", 1995)
    assert not f.allows("a", None)


# -- recommend_for_case ------------------------------------------

def test_recommend_for_case_ranks_same_topic_first(engine):
    res = engine.recommend_for_case("tax_1980", k=3)
    assert res.query_in_corpus is True
    assert res.results[0].doc_id == "tax_1995"
    assert "tax_1980" not in [r.doc_id for r in res.results]  # self excluded


def test_recommend_for_case_respects_year_filter(engine):
    res = engine.recommend_for_case("crime_1975", k=5, filters=Filters(year_min=2000))
    assert [r.doc_id for r in res.results] == ["crime_2005"]
    assert res.n_filtered_out >= 2


def test_recommendation_has_breakdown_and_rationale(engine):
    r0 = engine.recommend_for_case("tax_1980", k=1).results[0]
    assert r0.shared_sections and r0.shared_sections[0]["section_key"] == "income_tax_act_1922::s66"
    assert isinstance(r0.rationale, str) and r0.rationale
    assert 0.0 <= r0.hybrid_score <= 1.0


def test_unknown_case_raises(engine):
    with pytest.raises(KeyError):
        engine.recommend_for_case("nope")


def test_ann_candidate_pool_matches_brute_force_top_results(engine):
    from app.ml.vector_index import build_from_index

    engine.text_ann = build_from_index(engine.model.text_index, backend="numpy")
    engine.network_ann = build_from_index(engine.model.network_index.embeddings, backend="numpy")
    engine.candidate_pool = 3
    res = engine.recommend_for_case("tax_1980", k=2)
    assert res.results[0].doc_id == "tax_1995"           # same top result
    assert res.n_candidates <= 3                          # pool, not the whole corpus


# -- recommend_for_text ----------------------------------------

def test_recommend_for_text_uploaded_case(engine):
    res = engine.recommend_for_text(
        "s. 302 of the Indian Penal Code murder trial accused", k=2, year=2011
    )
    assert res.query_in_corpus is False
    assert res.results[0].doc_id in {"crime_1975", "crime_2005"}
    assert res.results[0].embedding_score == 0.0  # no graph node for an upload


def test_to_dict_is_json_safe(engine):
    import json

    res = engine.recommend_for_case("tax_1980", k=2)
    json.dumps(res.to_dict())  # must not raise


# -- real corpus -----------------------------------------------

def test_engine_on_real_corpus():
    from app.ml.network_embedding import NetworkEmbeddingIndex as NEI
    from app.utils.dataset import load_corpus
    from app.utils.paths import data_embeddings_dir

    corpus = load_corpus()
    tdir = data_embeddings_dir() / "tfidf"
    if not (tdir / "index.json").is_file():
        pytest.skip("run scripts/generate_embeddings.py first")
    ti = TextSimilarityIndex.build(corpus, TfidfEmbedder())
    ei = NEI.load(data_embeddings_dir() / "network_metapath2vec")
    ni = NetworkSimilarityIndex.load(data_embeddings_dir() / "network_similarity", ei, corpus=corpus)
    eng = RecommendationEngine(corpus, HybridSimilarityModel(ti, ni, WeightedSumCombiner()))

    res = eng.recommend_for_case("2005_S_628", k=5)  # Jacob Mathew (medical negligence)
    titles = " ".join((r.title or "") for r in res.results).lower()
    # the medical-negligence line should surface
    assert any(w in titles for w in ("kusum sharma", "d'souza", "desai", "batra"))
