"""Evaluation metrics (pipeline stage: Evaluation).

Everything here operates on the REAL expert-scored pairs in
``data/raw/<split>/similarity_scores.csv`` (100 validation + 90 test). No number
this module produces is hard-coded - callers pass a scoring function and get back
metrics computed live.

Primary metric (the base paper's): correlation of the model's pairwise similarity
with the expert pairwise similarity (Pearson + Spearman), plus MSE / MAE.

Secondary: a pairwise ranking sanity - for each doc that appears as a query in
the pairs, rank the whole corpus by the model and report where its highest-rated
expert partner lands (MRR, hit@k).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class CorrelationMetrics:
    n: int
    pearson: float
    pearson_p: float
    spearman: float
    spearman_p: float
    mse: float
    mae: float

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def correlation_metrics(pred: Sequence[float], gold: Sequence[float]) -> CorrelationMetrics:
    from scipy.stats import pearsonr, spearmanr

    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gold, dtype=np.float64)
    if len(p) < 3 or np.allclose(p, p[0]):
        return CorrelationMetrics(len(p), float("nan"), float("nan"),
                                  float("nan"), float("nan"),
                                  float(np.mean((p - g) ** 2)), float(np.mean(np.abs(p - g))))
    pr = pearsonr(p, g)
    sr = spearmanr(p, g)
    return CorrelationMetrics(
        n=len(p), pearson=float(pr.statistic), pearson_p=float(pr.pvalue),
        spearman=float(sr.statistic), spearman_p=float(sr.pvalue),
        mse=float(np.mean((p - g) ** 2)), mae=float(np.mean(np.abs(p - g))),
    )


def evaluate_pairwise(score_fn: Callable[[str, str], float], pairs) -> CorrelationMetrics:
    """`pairs` is a list of app.utils.dataset.ExpertPair."""
    pred = [float(score_fn(p.doc_id_1, p.doc_id_2)) for p in pairs]
    gold = [float(p.score) for p in pairs]
    return correlation_metrics(pred, gold)


# --------------------------------------------------------------------------- #
# Classification metrics (binarized view of the continuous similarity task)   #
# --------------------------------------------------------------------------- #

@dataclass
class ClassificationMetrics:
    n: int
    gold_threshold: float
    pred_threshold: float
    positives: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def classification_metrics(
    pred: Sequence[float], gold: Sequence[float], *,
    gold_threshold: float = 0.5, pred_threshold: Optional[float] = None,
) -> ClassificationMetrics:
    """Accuracy / precision / recall / F1 / ROC-AUC for the "is this pair
    similar?" yes-or-no question, binarizing the expert score at
    `gold_threshold` and the model's score at `pred_threshold` (default: the
    same value - a principled a-priori choice, not tuned on this data, since
    a threshold picked to maximise F1 on the same set it's scored on would be
    a form of leakage). ROC-AUC needs no threshold at all and is the more
    honest single number for a continuous score - report it alongside these,
    not instead of the correlation metrics above, which are what this task
    actually optimizes for.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gold, dtype=np.float64)
    y_true = (g >= gold_threshold).astype(int)
    thr = gold_threshold if pred_threshold is None else pred_threshold
    y_pred = (p >= thr).astype(int)

    try:
        auc = float(roc_auc_score(y_true, p))
    except ValueError:
        auc = float("nan")  # only one class present in y_true

    return ClassificationMetrics(
        n=len(p), gold_threshold=gold_threshold, pred_threshold=thr,
        positives=int(y_true.sum()),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=auc,
    )


# --------------------------------------------------------------------------- #
# Reliability calibration (per-result "how often has a score like this been    #
# right historically")                                                       #
# --------------------------------------------------------------------------- #

class ReliabilityCalibrator:
    """Maps a raw hybrid score to "of past expert-scored pairs the model rated
    around this level, what fraction did legal experts actually judge similar
    (gold >= threshold)?" - via logistic (Platt) scaling on out-of-sample
    predictions, so a *specific* recommended case can carry a real,
    data-backed reliability figure instead of a fabricated per-pair number
    (there is no ground truth for any single new pair).

    Logistic rather than isotonic regression deliberately: isotonic is a step
    function, and with only ~100 calibration points it produces wide flat
    plateaus - many visibly different scores landing on the exact same
    percentage, which reads as a bug even though each value is real. A
    2-parameter sigmoid is the standard, more data-efficient choice for small
    samples (see e.g. Platt 1999; scikit-learn's own calibration guidance
    recommends sigmoid over isotonic below roughly 1,000 samples): every
    distinct score gets its own smoothly-varying value, monotonically
    increasing, with no plateaus.

    Deliberately not a claim of statistical precision: n is small (currently
    100 validation pairs), so the number is reported alongside its sample
    size everywhere it's shown, never alone.
    """

    def __init__(self, gold_threshold: float = 0.5):
        self.gold_threshold = gold_threshold
        self.n_samples = 0
        self._model = None

    def fit(self, pred: Sequence[float], gold: Sequence[float]) -> "ReliabilityCalibrator":
        from sklearn.linear_model import LogisticRegression

        p = np.asarray(pred, dtype=np.float64).reshape(-1, 1)
        y = (np.asarray(gold, dtype=np.float64) >= self.gold_threshold).astype(int)
        self.n_samples = len(p)
        # Light regularization (C=10, i.e. weak): with n~100 the default C=1.0
        # over-shrinks toward 0.5, muting the very separation we're fitting
        # this for. Still regularized enough not to blow up on a handful of
        # points during tests.
        self._model = LogisticRegression(C=10.0)
        self._model.fit(p, y)
        return self

    def reliability(self, score: float) -> float:
        """P(this score's pair would be judged "similar") per validation calibration."""
        if self._model is None:
            raise RuntimeError("ReliabilityCalibrator.reliability() before fit()/load()")
        return float(np.clip(self._model.predict_proba([[score]])[0, 1], 0.0, 1.0))

    def curve(self, n_points: int = 11) -> list[dict]:
        """Sampled (score -> reliability) points, for display as a small chart/table."""
        xs = np.linspace(0.0, 1.0, n_points)
        return [{"score": round(float(x), 2), "reliability": round(self.reliability(x), 4)} for x in xs]

    def save(self, path) -> None:
        import joblib

        joblib.dump({"gold_threshold": self.gold_threshold, "n_samples": self.n_samples,
                     "model": self._model}, path)

    @classmethod
    def load(cls, path) -> "ReliabilityCalibrator":
        import joblib

        d = joblib.load(path)
        obj = cls(gold_threshold=d["gold_threshold"])
        obj.n_samples = d["n_samples"]
        obj._model = d["model"]
        return obj


# --------------------------------------------------------------------------- #
# Pairwise ranking sanity                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class RankingMetrics:
    n_queries: int
    mrr: float
    hit_at: dict[int, float]
    median_rank: float
    mean_gold_score: float

    def to_dict(self) -> dict:
        return {
            "n_queries": self.n_queries, "mrr": round(self.mrr, 4),
            "hit_at": {str(k): round(v, 4) for k, v in self.hit_at.items()},
            "median_rank": self.median_rank, "mean_gold_score": round(self.mean_gold_score, 4),
        }


def ranking_sanity(
    rank_fn: Callable[[str], list[str]],
    pairs,
    corpus,
    *,
    min_gold: float = 0.5,
    ks: Sequence[int] = (1, 5, 10, 20),
) -> RankingMetrics:
    """For each query doc with >=1 expert partner scored >= `min_gold`, rank the
    whole corpus with `rank_fn(query)` and locate that partner."""
    partners: dict[str, list[tuple[str, float]]] = {}
    for p in pairs:
        if p.score >= min_gold:
            partners.setdefault(p.doc_id_1, []).append((p.doc_id_2, p.score))
            partners.setdefault(p.doc_id_2, []).append((p.doc_id_1, p.score))

    ranks: list[int] = []
    gold_scores: list[float] = []
    for q, plist in partners.items():
        if q not in corpus:
            continue
        ordered = rank_fn(q)
        pos = {d: i for i, d in enumerate(ordered)}
        best_partner, best_score = max(plist, key=lambda t: t[1])
        r = pos.get(best_partner)
        if r is None:
            continue
        ranks.append(r + 1)  # 1-indexed
        gold_scores.append(best_score)

    if not ranks:
        return RankingMetrics(0, float("nan"), {k: float("nan") for k in ks}, float("nan"), float("nan"))
    ranks_arr = np.array(ranks)
    return RankingMetrics(
        n_queries=len(ranks),
        mrr=float(np.mean(1.0 / ranks_arr)),
        hit_at={k: float(np.mean(ranks_arr <= k)) for k in ks},
        median_rank=float(np.median(ranks_arr)),
        mean_gold_score=float(np.mean(gold_scores)),
    )


# --------------------------------------------------------------------------- #
# Split diagnostics (the test-split investigation)                            #
# --------------------------------------------------------------------------- #

@dataclass
class SplitDiagnosis:
    split: str
    n_pairs: int
    score_min: float
    score_max: float
    score_mean: float
    score_std: float
    distinct_scores: int
    pairs_with_isolated_case: int
    pairs_both_have_statutes: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def diagnose_split(split: str, pairs, *, isolated_case_ids: set[str],
                   docs_with_statutes: set[str]) -> SplitDiagnosis:
    scores = np.array([p.score for p in pairs], dtype=np.float64)
    iso = sum(
        1 for p in pairs
        if p.doc_id_1 in isolated_case_ids or p.doc_id_2 in isolated_case_ids
    )
    both_stat = sum(
        1 for p in pairs
        if p.doc_id_1 in docs_with_statutes and p.doc_id_2 in docs_with_statutes
    )
    notes: list[str] = []
    if scores.min() > 0.05:
        notes.append(f"score range is restricted to [{scores.min():.2f}, {scores.max():.2f}] "
                     "- no clearly-dissimilar pairs, which caps achievable correlation")
    if iso / max(1, len(pairs)) > 0.2:
        notes.append(f"{iso}/{len(pairs)} pairs involve a case with no Hier-SPCNet node "
                     "(network similarity is 0 for those by construction)")
    return SplitDiagnosis(
        split=split, n_pairs=len(pairs),
        score_min=float(scores.min()), score_max=float(scores.max()),
        score_mean=float(scores.mean()), score_std=float(scores.std()),
        distinct_scores=len(set(p.score for p in pairs)),
        pairs_with_isolated_case=iso, pairs_both_have_statutes=both_stat, notes=notes,
    )
