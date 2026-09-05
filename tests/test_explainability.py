"""Tests for app.ml.explainability (Step 14)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.ml.explainability import Explainer, _contributions
from app.ml.hybrid_similarity import HybridSimilarityModel, LearnedCombiner, WeightedSumCombiner
from app.ml.network_embedding import NetworkEmbeddingIndex
from app.ml.network_similarity import DocFeatures, NetworkSimilarityIndex
from app.ml.statute_extraction import load_hierarchy
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
def explainer():
    from app.ml.preprocessing import preprocess_text

    docs = [
        _Doc("a", 1990, "s. 302 of the Indian Penal Code murder accused evidence conviction"),
        _Doc("b", 1995, "s. 302 of the Indian Penal Code murder witness sentence appeal"),
        _Doc("c", 2000, "s. 66 of the Indian Income-tax Act, 1922 assessment revenue"),
    ]
    corpus = _Corpus(docs)
    te = TfidfEmbedder(min_df=1)
    te.fit([TextDoc(i, preprocess_text(corpus.text(i)).tokens, corpus.text(i)) for i in corpus.doc_ids])
    ti = TextSimilarityIndex("tfidf", corpus.doc_ids, te.embed_corpus(), embedder=te)

    ids = corpus.doc_ids
    ei = NetworkEmbeddingIndex("metapath2vec", ids, np.eye(3, dtype=np.float32))
    store = {
        "a": DocFeatures(cited_doc_ids={"z"}, act_ids={"ipc_1860"}, section_keys={"ipc_1860::s302"}),
        "b": DocFeatures(cited_doc_ids={"z"}, act_ids={"ipc_1860"}, section_keys={"ipc_1860::s302"}),
        "c": DocFeatures(act_ids={"income_tax_act_1922"}),
    }
    ni = NetworkSimilarityIndex(ei, store, method="blend")
    model = HybridSimilarityModel(ti, ni, WeightedSumCombiner(0.6, 0.4))
    return Explainer(corpus, model, hierarchy=load_hierarchy())


# -- contribution attribution -------------------------------------

def test_weighted_contributions_sum_to_score():
    c = WeightedSumCombiner(0.5, 0.5)
    feats = {"text_score": 0.8, "network_score": 0.4}
    parts = _contributions(c, feats)
    assert {p.component for p in parts} == {"text", "network"}
    assert sum(p.contribution for p in parts) == pytest.approx(0.6, abs=1e-6)
    assert sum(p.pct_of_score for p in parts) == pytest.approx(100.0, abs=0.2)


def test_learned_linear_contributions_include_intercept():
    lc = LearnedCombiner(model="linear", alpha_reg=0.01)
    rng = np.random.default_rng(0)
    feats = [{"text_score": rng.random(), "network_score": rng.random()} for _ in range(80)]
    from app.ml.hybrid_similarity import HybridFeatures

    hf = [HybridFeatures(f["text_score"], f["network_score"], 0, 0, 0, 0) for f in feats]
    lc.fit(hf, [0.5 * f["text_score"] + 0.5 * f["network_score"] for f in feats])
    parts = _contributions(lc, {"text_score": 0.9, "network_score": 0.2})
    assert any(p.component == "intercept" for p in parts)


# -- explain_pair -----------------------------------------------

def test_explain_pair_shape_and_labels(explainer):
    ex = explainer.explain_pair("a", "b")
    assert ex.query_in_corpus is True
    assert ex.hybrid_score > 0
    # shared IPC s.302 with a human label from the curated hierarchy
    assert ex.shared_statutes
    s302 = next(s for s in ex.shared_statutes if s.section_number == "302")
    assert s302.act_name == "Indian Penal Code, 1860"
    assert s302.section_title and "murder" in s302.section_title.lower()
    assert s302.in_curated_hierarchy is True
    # shared cited case 'z' isn't in the corpus -> reported with no title
    assert any(cc.doc_id == "z" and cc.title is None for cc in ex.shared_cited_cases)
    assert isinstance(ex.summary, str) and ex.summary


def test_explain_pair_json_safe(explainer):
    json.dumps(explainer.explain_pair("a", "c").to_dict(), default=str)


def test_unrelated_pair_has_weak_summary(explainer):
    ex = explainer.explain_pair("a", "c")
    assert ex.hybrid_score < explainer.explain_pair("a", "b").hybrid_score


# -- explain_text (upload) ------------------------------------

def test_explain_text_upload(explainer):
    ex = explainer.explain_text(
        "s. 302 of the Indian Penal Code murder trial accused", "a", year=2011
    )
    assert ex.query_in_corpus is False
    assert ex.embedding_score == 0.0
    assert any(s.section_number == "302" for s in ex.shared_statutes)


# -- real corpus ----------------------------------------------

def test_explain_on_real_corpus():
    from app.ml.network_embedding import NetworkEmbeddingIndex as NEI
    from app.utils.dataset import load_corpus
    from app.utils.paths import data_embeddings_dir

    d = data_embeddings_dir()
    if not (d / "network_metapath2vec" / "index.json").is_file():
        pytest.skip("network embeddings not built")
    corpus = load_corpus()
    ti = TextSimilarityIndex.build(corpus, TfidfEmbedder())
    ei = NEI.load(d / "network_metapath2vec")
    ni = NetworkSimilarityIndex.load(d / "network_similarity", ei, corpus=corpus)
    model = HybridSimilarityModel(ti, ni, WeightedSumCombiner(0.5, 0.5))
    ex = Explainer(corpus, model).explain_pair("2005_S_628", "2010_B_40")
    # the two medical-negligence judgments share the classic English authorities
    joined = " ".join(ex.shared_raw_citations).lower()
    assert "bolam" in joined or "hunter v hanley" in joined
    assert 0.0 <= ex.hybrid_score <= 1.0
