"""Tests for app.ml.text_similarity (Step 8)."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.text_similarity import (
    METHODS,
    Doc2VecEmbedder,
    TextDoc,
    TextSimilarityIndex,
    TfidfEmbedder,
    l2_normalize,
    make_embedder,
)

# small synthetic corpus: two clear topic clusters
_TOY = [
    TextDoc("tax_1", "income tax assessment revenue levy return assessee".split(), "income tax assessment revenue levy return assessee"),
    TextDoc("tax_2", "tax revenue assessment levy income assessee deduction".split(), "tax revenue assessment levy income assessee deduction"),
    TextDoc("crim_1", "murder accused evidence witness sentence culpable homicide".split(), "murder accused evidence witness sentence culpable homicide"),
    TextDoc("crim_2", "accused murder witness evidence conviction sentence appeal".split(), "accused murder witness evidence conviction sentence appeal"),
]


class _ToyCorpus:
    doc_ids = [d.doc_id for d in _TOY]
    _by = {d.doc_id: d for d in _TOY}

    def __getitem__(self, k):  # not used by build() path we exercise
        raise KeyError(k)

    def text(self, k):
        return self._by[k].text


def _fit_index(embedder):
    embedder.fit(_TOY)
    return TextSimilarityIndex(embedder.name, [d.doc_id for d in _TOY],
                               embedder.embed_corpus(), embedder=embedder)


# -- helpers ----------------------------------------------------------

def test_l2_normalize_rows_unit_norm():
    X = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = l2_normalize(X)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)
    assert np.isclose(np.linalg.norm(out[1]), 0.0)  # zero row stays zero


def test_make_embedder_unknown():
    with pytest.raises(ValueError):
        make_embedder("word2vec")


@pytest.mark.parametrize("method", METHODS)
def test_embedder_declares_provenance(method):
    e = make_embedder(method)
    assert e.provenance and e.name == method


# -- TF-IDF ---------------------------------------------------------

def test_tfidf_clusters_toy_corpus():
    idx = _fit_index(TfidfEmbedder(min_df=1, ngram_range=(1, 1)))
    assert idx.pair_similarity("tax_1", "tax_2") > idx.pair_similarity("tax_1", "crim_1")
    top = idx.most_similar("crim_1", k=1)[0]
    assert top.doc_id == "crim_2"


def test_tfidf_self_similarity_is_one():
    idx = _fit_index(TfidfEmbedder(min_df=1))
    assert idx.pair_similarity("tax_1", "tax_1") == pytest.approx(1.0, abs=1e-5)


def test_most_similar_excludes_self_by_default():
    idx = _fit_index(TfidfEmbedder(min_df=1))
    ids = [n.doc_id for n in idx.most_similar("tax_1", k=3)]
    assert "tax_1" not in ids


def test_query_vector_ranks():
    idx = _fit_index(TfidfEmbedder(min_df=1))
    q = idx.vector("tax_2")
    assert idx.query_vector(q, k=1)[0].doc_id == "tax_2"


# -- Doc2Vec (base paper) - tiny + few epochs so it's fast --------

def test_doc2vec_builds_and_ranks():
    idx = _fit_index(Doc2VecEmbedder(vector_size=16, epochs=20, min_count=1, window=3))
    assert idx.matrix.shape == (4, 16)
    assert np.allclose(np.linalg.norm(idx.matrix, axis=1), 1.0, atol=1e-4)
    # infer_vector on unseen tokens returns a unit vector of the right size
    v = idx.embedder.embed_text("tax revenue assessment".split(), "tax revenue assessment")
    assert v.shape == (16,)


# -- persistence --------------------------------------------------

def test_save_and_load_roundtrip(tmp_path):
    idx = _fit_index(TfidfEmbedder(min_df=1))
    idx.save(tmp_path)
    loaded = TextSimilarityIndex.load(tmp_path)
    assert loaded.doc_ids == idx.doc_ids
    assert loaded.pair_similarity("tax_1", "tax_2") == pytest.approx(
        idx.pair_similarity("tax_1", "tax_2"), abs=1e-6
    )
    with pytest.raises(RuntimeError):
        loaded.query_text("some text")  # no embedder after load


# -- real corpus (TF-IDF only; doc2vec/sbert are slow/optional) ---

@pytest.mark.parametrize("split,expect_positive", [("validation", True), ("test", False)])
def test_tfidf_real_corpus_correlation_direction(split, expect_positive):
    """Sanity, not a benchmark: on the real pairs TF-IDF cosine correlates with
    the expert score on validation; on test the correlation is known to be weak
    (documented in CHANGE_LOG - text alone is not enough, hence the network)."""
    pytest.importorskip("scipy")
    from scipy.stats import spearmanr

    from app.utils.dataset import load_corpus

    corpus = load_corpus()
    idx = TextSimilarityIndex.build(corpus, TfidfEmbedder())
    pairs = corpus.pairs_by_split[split]
    rho = spearmanr(
        [idx.pair_similarity(p.doc_id_1, p.doc_id_2) for p in pairs],
        [p.score for p in pairs],
    ).statistic
    if expect_positive:
        assert rho > 0.4
    else:
        assert -0.3 < rho < 0.3  # near zero - the point of the paper
