"""Tests for app.ml.network_embedding (Step 10)."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.network_embedding import (
    METHODS,
    NetworkEmbedder,
    NetworkEmbeddingIndex,
    metapath_walks,
    node2vec_walks,
)
from app.ml.network_model import build_hier_spcnet


class _Doc:
    def __init__(self, doc_id, year, text):
        self.doc_id, self.year, self.text_ = doc_id, year, text
        self.title = f"{doc_id} title"


class _Corpus:
    def __init__(self, docs):
        self._d = {d.doc_id: d for d in docs}

    def __len__(self):
        return len(self._d)

    @property
    def doc_ids(self):
        return sorted(self._d)

    def __getitem__(self, k):
        return self._d[k]

    def __contains__(self, k):
        return k in self._d

    def text(self, k):
        return self._d[k].text_


@pytest.fixture(scope="module")
def graph_and_corpus():
    # tax cluster (cite Income-tax Act s.66) vs crime cluster (cite IPC s.302),
    # plus one isolated case with no statutes.
    docs = [
        _Doc("t1", 1980, "s. 66 of the Indian Income-tax Act, 1922 and s. 4 of the Indian Income-tax Act, 1922"),
        _Doc("t2", 1985, "s. 66 of the Indian Income-tax Act, 1922; s. 22 of the Indian Income-tax Act, 1922"),
        _Doc("t3", 1990, "assessment under s. 66 of the Indian Income-tax Act, 1922"),
        _Doc("c1", 1975, "s. 302 of the Indian Penal Code and s. 34 of the Indian Penal Code"),
        _Doc("c2", 1979, "conviction under s. 302 of the Indian Penal Code; s. 149 of the Indian Penal Code"),
        _Doc("iso", 2001, "a case that cites nothing resolvable at all"),
    ]
    corpus = _Corpus(docs)
    graph = build_hier_spcnet(corpus, citation_edges=[])
    return graph, corpus


# -- walks ----------------------------------------------------------

def test_node2vec_walks_are_reproducible(graph_and_corpus):
    g, _ = graph_and_corpus
    u = g.to_undirected()
    w1 = node2vec_walks(u, num_walks=3, walk_length=10, seed=7)
    w2 = node2vec_walks(u, num_walks=3, walk_length=10, seed=7)
    assert w1 == w2
    assert all(len(w) >= 1 for w in w1)


def test_metapath_walks_follow_schema(graph_and_corpus):
    g, _ = graph_and_corpus
    u = g.to_undirected()
    walks = metapath_walks(u, metapaths=[("case", "section", "case")],
                           num_walks=2, walk_length=6, seed=1)
    assert walks
    for w in walks:
        types = [u.nodes[n]["ntype"] for n in w]
        # schema case-section-case-section-...
        assert types[0] == "case"
        assert all(t in {"case", "section"} for t in types)


# -- embedder / index ---------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_build_index_shapes_and_missing(method, graph_and_corpus):
    g, corpus = graph_and_corpus
    emb = NetworkEmbedder(method=method, dimensions=32, num_walks=10, walk_length=15, epochs=3)
    idx = NetworkEmbeddingIndex.build(g, corpus, emb)
    assert idx.matrix.shape == (6, 32)
    assert "iso" in idx.missing            # isolated case -> zero vector
    assert not idx.has("iso")
    # rows are unit-norm except the missing one
    norms = np.linalg.norm(idx.matrix, axis=1)
    assert np.isclose(norms[idx.doc_ids.index("iso")], 0.0)


def test_pair_similarity_zero_for_isolated(graph_and_corpus):
    g, corpus = graph_and_corpus
    idx = NetworkEmbeddingIndex.build(
        g, corpus, NetworkEmbedder(method="metapath2vec", dimensions=32, epochs=3)
    )
    assert idx.pair_similarity("t1", "iso") == 0.0


def test_clusters_separate(graph_and_corpus):
    g, corpus = graph_and_corpus
    idx = NetworkEmbeddingIndex.build(
        g, corpus, NetworkEmbedder(method="metapath2vec", dimensions=48,
                                   num_walks=30, walk_length=20, epochs=15, seed=3)
    )
    # a tax case should look more like the other tax cases than the crime cases
    tax = np.mean([idx.pair_similarity("t1", d) for d in ("t2", "t3")])
    crime = np.mean([idx.pair_similarity("t1", d) for d in ("c1", "c2")])
    assert tax > crime


def test_most_similar_excludes_self_and_missing(graph_and_corpus):
    g, corpus = graph_and_corpus
    idx = NetworkEmbeddingIndex.build(
        g, corpus, NetworkEmbedder(method="metapath2vec", dimensions=32, epochs=3)
    )
    nbrs = [n.doc_id for n in idx.most_similar("t1", k=10)]
    assert "t1" not in nbrs and "iso" not in nbrs


def test_save_load_roundtrip(graph_and_corpus, tmp_path):
    g, corpus = graph_and_corpus
    idx = NetworkEmbeddingIndex.build(
        g, corpus, NetworkEmbedder(method="metapath2vec", dimensions=32, epochs=3)
    )
    idx.save(tmp_path)
    loaded = NetworkEmbeddingIndex.load(tmp_path)
    assert loaded.doc_ids == idx.doc_ids
    assert loaded.missing == idx.missing
    assert loaded.pair_similarity("t1", "t2") == pytest.approx(idx.pair_similarity("t1", "t2"), abs=1e-6)
