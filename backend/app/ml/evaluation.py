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
