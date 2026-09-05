"""Tests for app.ml.hybrid_similarity (Step 12)."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.hybrid_similarity import (
    HybridFeatures,
    HybridSimilarityModel,
    LearnedCombiner,
    WeightedSumCombiner,
    make_combiner,
)
from app.ml.network_embedding import NetworkEmbeddingIndex
from app.ml.network_similarity import DocFeatures, NetworkSimilarityIndex
from app.ml.text_similarity import TextSimilarityIndex, TfidfEmbedder, TextDoc


def _feats(text, net, **kw):
    return HybridFeatures(text, net, kw.get("emb", net), kw.get("stat", 0.0),
                          kw.get("cite", 0.0), kw.get("cocit", 0))


# -- weighted --------------------------------------------------------

def test_weighted_normalises_alpha_beta():
    c = WeightedSumCombiner(alpha=3, beta=1)
    assert c.alpha == pytest.approx(0.75) and c.beta == pytest.approx(0.25)


def test_weighted_score_is_convex_combo_and_clipped():
    c = WeightedSumCombiner(0.5, 0.5)
    assert c.score(_feats(0.8, 0.4)) == pytest.approx(0.6)
    assert 0.0 <= c.score(_feats(1.0, 1.0)) <= 1.0


# -- learned --------------------------------------------------------

def test_learned_requires_fit():
    with pytest.raises(RuntimeError):
        LearnedCombiner().score(_feats(0.5, 0.5))


def test_learned_recovers_a_linear_relationship():
    # target = 0.7*text + 0.3*network exactly -> Ridge should get close
    rng = np.random.default_rng(0)
    feats, targets = [], []
    for _ in range(120):
        t, n = rng.random(), rng.random()
        feats.append(_feats(t, n))
        targets.append(0.7 * t + 0.3 * n)
    c = LearnedCombiner(model="linear", alpha_reg=0.01).fit(feats, targets)
    pred = c.score(_feats(0.9, 0.1))
    assert pred == pytest.approx(0.7 * 0.9 + 0.3 * 0.1, abs=0.05)


def test_learned_cross_val_reports_metrics():
    rng = np.random.default_rng(1)
    feats = [_feats(rng.random(), rng.random()) for _ in range(60)]
    targets = [0.5 * f.text_score + 0.5 * f.network_score for f in feats]
    cv = LearnedCombiner().cross_val_pearson(feats, targets, folds=4)
    assert cv["folds"] == 4
    assert cv["cv_pearson"] > 0.8  # near-perfect linear target
    assert cv["cv_mse"] >= 0.0


def test_make_combiner_unknown():
    with pytest.raises(ValueError):
        make_combiner("stacking")


# -- model over a tiny real-ish setup -----------------------------

class _Doc:
    def __init__(self, doc_id):
        self.doc_id, self.year, self.title = doc_id, 1990, f"{doc_id} v State"


class _Corpus:
    def __init__(self, ids, pairs):
        self._d = {i: _Doc(i) for i in ids}
        self.pairs_by_split = {"validation": pairs}

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


class _Pair:
    def __init__(self, a, b, s):
        self.doc_id_1, self.doc_id_2, self.score = a, b, s


@pytest.fixture
def model():
    ids = ["a", "b", "c"]
    # text index from toy tokens
    te = TfidfEmbedder(min_df=1)
    te.fit([
        TextDoc("a", "tax revenue assessment".split(), ""),
        TextDoc("b", "tax revenue levy".split(), ""),
        TextDoc("c", "murder evidence witness".split(), ""),
    ])
    ti = TextSimilarityIndex("tfidf", ids, te.embed_corpus(), embedder=te)

    mat = np.array([[1, 0], [0.8, 0.2], [0, 1]], dtype=np.float32)
    ei = NetworkEmbeddingIndex("metapath2vec", ids, mat)
    store = {
        "a": DocFeatures(act_ids={"tax"}, section_keys={"tax::s66"}),
        "b": DocFeatures(act_ids={"tax"}, section_keys={"tax::s66"}),
        "c": DocFeatures(act_ids={"ipc"}),
    }
    ni = NetworkSimilarityIndex(ei, store, method="blend")

    corpus = _Corpus(ids, [_Pair("a", "b", 0.9), _Pair("a", "c", 0.1), _Pair("b", "c", 0.1)])
    return HybridSimilarityModel(ti, ni, WeightedSumCombiner(0.5, 0.5)), corpus


def test_features_and_score(model):
    m, _ = model
    f = m.features("a", "b")
    assert f.text_score > 0 and f.network_score > 0
    assert 0.0 <= m.score("a", "b") <= 1.0
    assert m.score("a", "b") > m.score("a", "c")


def test_explain_shape(model):
    m, _ = model
    ex = m.explain("a", "b")
    assert ex.method == "weighted"
    assert set(ex.network_breakdown) >= {"shared_acts", "embedding_score"}
    assert ex.hybrid_score == pytest.approx(
        0.5 * ex.text_score + 0.5 * ex.network_score, abs=1e-3
    )


def test_most_similar_excludes_self(model):
    m, _ = model
    nbrs = m.most_similar("a", k=2)
    assert nbrs[0].doc_id == "b" and "a" not in [n.doc_id for n in nbrs]


def test_fit_learned_wrong_combiner_raises(model):
    m, corpus = model
    with pytest.raises(TypeError):
        m.fit_learned(corpus)


def test_fit_learned_runs(model):
    m, corpus = model
    m.combiner = LearnedCombiner(model="linear")
    info = m.fit_learned(corpus, split="validation", folds=3)
    assert info["n_pairs"] == 3 and "cv_pearson" in info
    assert 0.0 <= m.score("a", "b") <= 1.0


def test_save_writes_config(model, tmp_path):
    m, _ = model
    out = m.save(tmp_path)
    import json

    cfg = json.loads((out / "config.json").read_text())
    assert cfg["combiner"]["name"] == "weighted"
