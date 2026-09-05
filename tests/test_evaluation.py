"""Tests for app.ml.evaluation (Step 22)."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.evaluation import (
    correlation_metrics,
    diagnose_split,
    evaluate_pairwise,
    ranking_sanity,
)


class _Pair:
    def __init__(self, a, b, s):
        self.doc_id_1, self.doc_id_2, self.score = a, b, s


class _Doc:
    def __init__(self, doc_id, year):
        self.doc_id, self.year = doc_id, year


class _Corpus:
    def __init__(self, ids):
        self._d = {i: _Doc(i, 2000) for i in ids}

    @property
    def doc_ids(self):
        return sorted(self._d)

    def __contains__(self, k):
        return k in self._d

    def __getitem__(self, k):
        return self._d[k]


# -- correlation --------------------------------------------------

def test_correlation_perfect():
    m = correlation_metrics([0.1, 0.4, 0.6, 0.9], [0.1, 0.4, 0.6, 0.9])
    assert m.pearson == pytest.approx(1.0) and m.spearman == pytest.approx(1.0)
    assert m.mse == pytest.approx(0.0)


def test_correlation_constant_prediction_is_nan_not_crash():
    m = correlation_metrics([0.5, 0.5, 0.5, 0.5], [0.1, 0.2, 0.3, 0.4])
    assert np.isnan(m.pearson) and m.mse > 0


def test_evaluate_pairwise_uses_score_fn():
    pairs = [_Pair("a", "b", 0.8), _Pair("a", "c", 0.2),
             _Pair("a", "d", 0.5), _Pair("a", "e", 0.9)]
    m = evaluate_pairwise(lambda x, y: {"b": 0.8, "c": 0.2, "d": 0.5, "e": 0.9}[y], pairs)
    assert m.n == 4 and m.pearson == pytest.approx(1.0)


# -- ranking ----------------------------------------------------

def test_ranking_sanity_finds_partner_at_top():
    corpus = _Corpus(["q", "good", "bad1", "bad2"])
    pairs = [_Pair("q", "good", 0.9)]
    # rank_fn puts 'good' first
    rm = ranking_sanity(lambda q: ["good", "bad1", "bad2"], pairs, corpus, min_gold=0.5)
    assert rm.n_queries == 1 and rm.mrr == pytest.approx(1.0)
    assert rm.hit_at[1] == 1.0


def test_ranking_sanity_ignores_low_gold_pairs():
    corpus = _Corpus(["q", "x"])
    rm = ranking_sanity(lambda q: ["x"], [_Pair("q", "x", 0.2)], corpus, min_gold=0.5)
    assert rm.n_queries == 0


# -- diagnosis -------------------------------------------------

def test_diagnose_split_flags_range_restriction_and_isolation():
    pairs = [_Pair("a", "b", 0.3), _Pair("a", "iso", 0.7), _Pair("c", "iso", 0.6)]
    d = diagnose_split("test", pairs, isolated_case_ids={"iso"},
                       docs_with_statutes={"a", "b", "c"})
    assert d.score_min == 0.3
    assert d.pairs_with_isolated_case == 2
    assert d.pairs_both_have_statutes == 1
    assert any("restricted" in n for n in d.notes)
    assert any("no Hier-SPCNet node" in n for n in d.notes)


# -- against the real pairs ---------------------------------

def test_real_validation_correlation_is_positive_for_tfidf():
    from app.ml.text_similarity import TextSimilarityIndex, TfidfEmbedder
    from app.utils.dataset import load_corpus

    corpus = load_corpus()
    ti = TextSimilarityIndex.build(corpus, TfidfEmbedder())
    m = evaluate_pairwise(lambda a, b: ti.pair_similarity(a, b),
                          corpus.pairs_by_split["validation"])
    assert m.n == 100 and m.spearman > 0.5  # documented: ~0.76
