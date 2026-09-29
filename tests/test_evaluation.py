"""Tests for app.ml.evaluation (Step 22)."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.evaluation import (
    ReliabilityCalibrator,
    classification_metrics,
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


# -- classification -------------------------------------------------

def test_classification_perfect_separation():
    # gold >= 0.5 -> similar; predictions perfectly agree at the same threshold
    pred = [0.1, 0.9, 0.2, 0.8]
    gold = [0.1, 0.9, 0.3, 0.7]
    m = classification_metrics(pred, gold, gold_threshold=0.5)
    assert m.accuracy == pytest.approx(1.0)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)
    assert m.roc_auc == pytest.approx(1.0)
    assert m.positives == 2


def test_classification_worst_case_inverted():
    pred = [0.9, 0.1]  # inverted relative to gold
    gold = [0.1, 0.9]
    m = classification_metrics(pred, gold, gold_threshold=0.5)
    assert m.accuracy == pytest.approx(0.0)
    assert m.roc_auc == pytest.approx(0.0)


def test_classification_single_class_gold_gives_nan_auc_not_crash():
    # every gold label below threshold -> only one class in y_true
    m = classification_metrics([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], gold_threshold=0.9)
    assert m.positives == 0
    assert np.isnan(m.roc_auc)
    assert m.accuracy == pytest.approx(1.0)  # all correctly predicted "not similar"


def test_classification_custom_pred_threshold():
    pred = [0.6, 0.4]
    gold = [0.9, 0.1]
    # at pred_threshold=0.5 this would be perfect; at 0.7 both are classified "not similar"
    m = classification_metrics(pred, gold, gold_threshold=0.5, pred_threshold=0.7)
    assert m.pred_threshold == 0.7
    assert m.recall == pytest.approx(0.0)


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


# -- reliability calibration ----------------------------------

def test_reliability_calibrator_monotonic_and_bounded():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 1, 200)
    gold = np.clip(pred + rng.normal(0, 0.1, 200), 0, 1)  # noisy but correlated
    cal = ReliabilityCalibrator(gold_threshold=0.5).fit(pred, gold)
    lo, hi = cal.reliability(0.05), cal.reliability(0.95)
    assert 0.0 <= lo <= hi <= 1.0
    assert cal.n_samples == 200


def test_reliability_calibrator_separates_low_from_high():
    # logistic (not isotonic) scaling: won't hit exact 0/1, but must clearly
    # separate a low-scoring region from a high-scoring one, monotonically.
    pred = [0.1, 0.15, 0.2, 0.8, 0.85, 0.9]
    gold = [0.1, 0.15, 0.2, 0.8, 0.85, 0.9]  # low half never similar, high half always
    cal = ReliabilityCalibrator(gold_threshold=0.5).fit(pred, gold)
    lo, hi = cal.reliability(0.1), cal.reliability(0.9)
    assert lo < 0.3
    assert hi > 0.7
    assert lo < hi


def test_reliability_calibrator_distinguishes_nearby_scores():
    # the whole point of switching off isotonic: no flat plateaus - two
    # different scores in the same rough band still get different values.
    rng = np.random.default_rng(1)
    pred = rng.uniform(0, 1, 100)
    gold = np.clip(pred + rng.normal(0, 0.15, 100), 0, 1)
    cal = ReliabilityCalibrator(gold_threshold=0.5).fit(pred, gold)
    assert cal.reliability(0.40) != cal.reliability(0.42)


def test_reliability_calibrator_save_load_roundtrip(tmp_path):
    cal = ReliabilityCalibrator().fit([0.1, 0.5, 0.9], [0.1, 0.5, 0.9])
    path = tmp_path / "calibration.joblib"
    cal.save(path)
    loaded = ReliabilityCalibrator.load(path)
    assert loaded.n_samples == 3
    assert loaded.reliability(0.9) == pytest.approx(cal.reliability(0.9))


def test_reliability_curve_has_requested_points():
    cal = ReliabilityCalibrator().fit([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    curve = cal.curve(n_points=5)
    assert len(curve) == 5
    assert curve[0]["score"] == 0.0 and curve[-1]["score"] == 1.0
