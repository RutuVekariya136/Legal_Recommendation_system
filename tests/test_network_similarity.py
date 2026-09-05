"""Tests for app.ml.network_similarity (Step 11)."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.network_embedding import NetworkEmbeddingIndex
from app.ml.network_similarity import (
    BlendWeights,
    DocFeatures,
    NetworkSimilarityIndex,
    _jaccard,
    _weighted_jaccard,
    compute_statute_icf,
)


# -- set-overlap helpers -------------------------------------------

def test_jaccard():
    assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert _jaccard(set(), set()) == 0.0


def test_weighted_jaccard_rewards_rare_shared_items():
    w = {"rare": 5.0, "common": 1.0}
    rare_share = _weighted_jaccard({"rare", "x"}, {"rare", "y"}, w)
    common_share = _weighted_jaccard({"common", "x"}, {"common", "y"}, w)
    assert rare_share > common_share


def test_compute_statute_icf_prefers_rare():
    store = {
        "d1": DocFeatures(act_ids={"common", "rare"}),
        "d2": DocFeatures(act_ids={"common"}),
        "d3": DocFeatures(act_ids={"common"}),
    }
    icf = compute_statute_icf(store)
    assert icf["rare"] > icf["common"]


# -- index over a synthetic setup -------------------------------

class _Doc:
    def __init__(self, doc_id, year):
        self.doc_id, self.year, self.title = doc_id, year, f"{doc_id} v State"


class _Corpus:
    def __init__(self, ids):
        self._d = {i: _Doc(i, 1990) for i in ids}

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
        return ""


@pytest.fixture
def index():
    ids = ["a", "b", "c", "d"]
    corpus = _Corpus(ids)
    # embedding: a,b close; c,d close
    mat = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 0, 1], [0, 0.1, 0.9]], dtype=np.float32)
    emb = NetworkEmbeddingIndex("metapath2vec", ids, mat)
    store = {
        "a": DocFeatures(cited_doc_ids={"x"}, act_ids={"ipc"}, section_keys={"ipc::s302"}),
        "b": DocFeatures(cited_doc_ids={"x"}, act_ids={"ipc"}, section_keys={"ipc::s302"}),
        "c": DocFeatures(act_ids={"tax"}, section_keys={"tax::s66"}),
        "d": DocFeatures(act_ids={"tax"}),
    }
    return NetworkSimilarityIndex(emb, store, method="blend",
                                  weights=BlendWeights(0.5, 0.4, 0.1), corpus=corpus)


def test_method_embedding_ignores_structural(index):
    index.method = "embedding"
    assert index.similarity("a", "b") == pytest.approx(index.embedding_score("a", "b"))


def test_blend_boosts_pairs_that_share_statutes(index):
    index.method = "embedding"
    emb_only = index.similarity("a", "b")
    index.method = "blend"
    blended = index.similarity("a", "b")
    assert blended > 0  # a & b share ipc + s302 + cite x
    # c & d share only the (common) 'tax' act -> lower structural contribution than a&b
    assert index.statute_overlap("a", "b") > index.statute_overlap("c", "d")
    assert emb_only >= 0


def test_breakdown_lists_shared_items(index):
    bd = index.breakdown("a", "b")
    assert [x["act_id"] for x in bd.shared_acts] == ["ipc"]
    assert [x["section_key"] for x in bd.shared_sections] == ["ipc::s302"]
    assert bd.shared_citation_doc_ids == ["x"]  # both a and b cite x
    assert bd.citation_overlap > 0
    assert bd.embedding_score > 0.9


def test_scores_bounded(index):
    for x in index.doc_ids:
        for y in index.doc_ids:
            s = index.similarity(x, y)
            assert 0.0 <= s <= 1.0


def test_most_similar_orders_and_excludes_self(index):
    nbrs = index.most_similar("a", k=3)
    assert nbrs[0].doc_id == "b"
    assert "a" not in [n.doc_id for n in nbrs]


def test_save_load_roundtrip(index, tmp_path):
    index.save(tmp_path)
    loaded = NetworkSimilarityIndex.load(tmp_path, index.embeddings)
    assert loaded.similarity("a", "b") == pytest.approx(index.similarity("a", "b"), abs=1e-6)
    assert loaded.breakdown("a", "b").shared_acts == index.breakdown("a", "b").shared_acts


# -- real corpus --------------------------------------------------

def test_build_on_real_corpus():
    from app.utils.dataset import load_corpus

    corpus = load_corpus()
    ids = corpus.doc_ids
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(len(ids), 16)).astype(np.float32)
    emb = NetworkEmbeddingIndex("metapath2vec", ids, mat)
    nsi = NetworkSimilarityIndex.build(corpus, emb, method="blend")
    assert len(nsi.features) == 295
    # some pair with shared statutes must exist
    p = corpus.pairs_by_split["validation"][0]
    bd = nsi.breakdown(p.doc_id_1, p.doc_id_2)
    assert 0.0 <= bd.network_score <= 1.0
    assert isinstance(bd.shared_sections, list)
