"""Hybrid similarity (pipeline stage: Hybrid Similarity).

    Text Similarity + Network Similarity -> [Hybrid Similarity] -> Case Ranking

Combines the text score (Step 8) and the network score (Step 11) into one
hybrid score. Two combiners:

  * ``weighted``  - ``clip(alpha * text + beta * network)``.  Configurable alpha/beta.
                    OUR simple hybrid.
  * ``learned``   - a small regressor (Ridge by default, optional MLP) trained on
                    the expert-scored pairs to map features -> expert similarity.
                    This follows the *idea* of the base paper's **NN-Map+Conc**
                    learned combination. The paper's reported gains (+11.8% over
                    best text-only, +20.6% over best network-only) are THEIR
                    result on THEIR 53k-case corpus - never claimed as ours.

Everything is pairwise and plain-data; no FastAPI/DB. The learned combiner must
be fit on a split (``validation``) before use; k-fold CV is provided for an
honest in-sample estimate.
"""

from __future__ import annotations

import abc
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from app.ml.network_similarity import NetworkSimilarityIndex
from app.ml.text_similarity import Neighbour, TextSimilarityIndex
from app.utils.paths import data_embeddings_dir, ensure_dir

METHODS = ("weighted", "learned")

# feature name -> extractor from a HybridFeatures instance
FEATURE_NAMES = (
    "text_score", "network_score", "embedding_score",
    "statute_overlap", "citation_overlap", "cocitation_count",
)


@dataclass
class HybridFeatures:
    text_score: float
    network_score: float
    embedding_score: float
    statute_overlap: float
    citation_overlap: float
    cocitation_count: int

    def vector(self, names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
        return np.array([getattr(self, n) for n in names], dtype=np.float64)


@dataclass
class HybridExplanation:
    doc_id_1: str
    doc_id_2: str
    hybrid_score: float
    text_score: float
    network_score: float
    method: str
    network_breakdown: dict


# --------------------------------------------------------------------------- #
# Combiners                                                                   #
# --------------------------------------------------------------------------- #

class BaseCombiner(abc.ABC):
    name: str

    @abc.abstractmethod
    def score(self, feats: HybridFeatures) -> float: ...

    def config(self) -> dict:
        return {"name": self.name}


class WeightedSumCombiner(BaseCombiner):
    name = "weighted"

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, *, normalize: bool = True):
        if normalize and (alpha + beta) > 0:
            s = alpha + beta
            alpha, beta = alpha / s, beta / s
        self.alpha, self.beta = float(alpha), float(beta)

    def score(self, feats: HybridFeatures) -> float:
        return float(np.clip(self.alpha * feats.text_score + self.beta * feats.network_score, 0.0, 1.0))

    def config(self) -> dict:
        return {"name": self.name, "alpha": self.alpha, "beta": self.beta}


class LearnedCombiner(BaseCombiner):
    name = "learned"

    def __init__(self, *, model: str = "linear",
                 features: Sequence[str] = ("text_score", "network_score"),
                 alpha_reg: float = 1.0, seed: int = 42):
        self.model_kind = model
        self.features = tuple(features)
        self.alpha_reg = alpha_reg
        self.seed = seed
        self._model = None
        self._fitted = False

    def _new_model(self):
        if self.model_kind == "linear":
            from sklearn.linear_model import Ridge

            return Ridge(alpha=self.alpha_reg, random_state=self.seed)
        if self.model_kind == "mlp":
            from sklearn.neural_network import MLPRegressor

            return MLPRegressor(hidden_layer_sizes=(8,), max_iter=2000,
                                random_state=self.seed, alpha=self.alpha_reg)
        raise ValueError(f"unknown learned model {self.model_kind!r}; expected 'linear' or 'mlp'")

    def _matrix(self, feats_list: Sequence[HybridFeatures]) -> np.ndarray:
        return np.vstack([f.vector(self.features) for f in feats_list])

    def fit(self, feats_list: Sequence[HybridFeatures], targets: Sequence[float]) -> "LearnedCombiner":
        self._model = self._new_model()
        self._model.fit(self._matrix(feats_list), np.asarray(targets, dtype=np.float64))
        self._fitted = True
        return self

    def cross_val_predict(self, feats_list: Sequence[HybridFeatures],
                          targets: Sequence[float], *, folds: int = 5) -> np.ndarray:
        from sklearn.model_selection import KFold

        X = self._matrix(feats_list)
        y = np.asarray(targets, dtype=np.float64)
        preds = np.zeros_like(y)
        kf = KFold(n_splits=folds, shuffle=True, random_state=self.seed)
        for tr, te in kf.split(X):
            m = self._new_model()
            m.fit(X[tr], y[tr])
            preds[te] = np.clip(m.predict(X[te]), 0.0, 1.0)
        return preds

    def cross_val_pearson(self, feats_list: Sequence[HybridFeatures],
                          targets: Sequence[float], *, folds: int = 5) -> dict:
        from scipy.stats import pearsonr, spearmanr

        y = np.asarray(targets, dtype=np.float64)
        preds = self.cross_val_predict(feats_list, targets, folds=folds)
        return {
            "folds": folds,
            "cv_pearson": float(pearsonr(preds, y).statistic),
            "cv_spearman": float(spearmanr(preds, y).statistic),
            "cv_mse": float(np.mean((preds - y) ** 2)),
        }

    def score(self, feats: HybridFeatures) -> float:
        if not self._fitted:
            raise RuntimeError("LearnedCombiner.score() before fit()")
        pred = float(self._model.predict(feats.vector(self.features)[None, :])[0])
        return float(np.clip(pred, 0.0, 1.0))

    def config(self) -> dict:
        cfg = {"name": self.name, "model": self.model_kind, "features": list(self.features),
               "alpha_reg": self.alpha_reg, "fitted": self._fitted}
        if self._fitted and self.model_kind == "linear":
            cfg["coef_"] = [float(c) for c in np.ravel(self._model.coef_)]
            cfg["intercept_"] = float(self._model.intercept_)
        return cfg


def make_combiner(method: str, **kwargs) -> BaseCombiner:
    if method == "weighted":
        return WeightedSumCombiner(**kwargs)
    if method == "learned":
        return LearnedCombiner(**kwargs)
    raise ValueError(f"unknown hybrid method {method!r}; expected one of {METHODS}")


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class HybridSimilarityModel:
    text_index: TextSimilarityIndex
    network_index: NetworkSimilarityIndex
    combiner: BaseCombiner = field(default_factory=WeightedSumCombiner)

    @property
    def doc_ids(self) -> list[str]:
        return list(self.text_index.doc_ids)

    def features(self, a: str, b: str) -> HybridFeatures:
        bd = self.network_index.breakdown(a, b)
        return HybridFeatures(
            text_score=float(self.text_index.pair_similarity(a, b)),
            network_score=float(bd.network_score),
            embedding_score=float(bd.embedding_score),
            statute_overlap=float(bd.statute_overlap),
            citation_overlap=float(bd.citation_overlap),
            cocitation_count=int(bd.cocitation_count),
        )

    def score(self, a: str, b: str) -> float:
        return self.combiner.score(self.features(a, b))

    def explain(self, a: str, b: str) -> HybridExplanation:
        bd = self.network_index.breakdown(a, b)
        feats = HybridFeatures(
            text_score=float(self.text_index.pair_similarity(a, b)),
            network_score=float(bd.network_score),
            embedding_score=float(bd.embedding_score),
            statute_overlap=float(bd.statute_overlap),
            citation_overlap=float(bd.citation_overlap),
            cocitation_count=int(bd.cocitation_count),
        )
        return HybridExplanation(
            doc_id_1=a, doc_id_2=b,
            hybrid_score=round(self.combiner.score(feats), 4),
            text_score=round(feats.text_score, 4),
            network_score=round(feats.network_score, 4),
            method=self.combiner.name,
            network_breakdown=bd.to_dict(),
        )

    def most_similar(self, doc_id: str, k: int = 10, *, exclude_self: bool = True) -> list[Neighbour]:
        out = [
            Neighbour(other, self.score(doc_id, other))
            for other in self.doc_ids
            if not (exclude_self and other == doc_id)
        ]
        out.sort(key=lambda n: -n.score)
        return out[:k]

    # -- fitting the learned combiner -------------------------------
    def _pairs_features(self, corpus, split: str) -> tuple[list[HybridFeatures], list[float]]:
        feats, targets = [], []
        for p in corpus.pairs_by_split[split]:
            feats.append(self.features(p.doc_id_1, p.doc_id_2))
            targets.append(p.score)
        return feats, targets

    def fit_learned(self, corpus, *, split: str = "validation", folds: int = 5) -> dict:
        if not isinstance(self.combiner, LearnedCombiner):
            raise TypeError("fit_learned() requires a LearnedCombiner")
        feats, targets = self._pairs_features(corpus, split)
        cv = self.combiner.cross_val_pearson(feats, targets, folds=folds)
        self.combiner.fit(feats, targets)
        return {"trained_on": split, "n_pairs": len(targets), **cv}

    def evaluate_correlation(self, corpus, split: str) -> dict:
        from scipy.stats import pearsonr, spearmanr

        feats, targets = self._pairs_features(corpus, split)
        preds = [self.combiner.score(f) for f in feats]
        return {
            "split": split, "n_pairs": len(targets),
            "pearson": float(pearsonr(preds, targets).statistic),
            "spearman": float(spearmanr(preds, targets).statistic),
            "mse": float(np.mean((np.array(preds) - np.array(targets)) ** 2)),
        }

    # -- persistence ----------------------------------------------
    @classmethod
    def load(cls, directory: Path, text_index: TextSimilarityIndex,
             network_index: NetworkSimilarityIndex) -> "HybridSimilarityModel":
        directory = Path(directory)
        cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        c = cfg["combiner"]
        if c["name"] == "weighted":
            combiner: BaseCombiner = WeightedSumCombiner(
                c.get("alpha", 0.5), c.get("beta", 0.5), normalize=False
            )
        else:
            combiner = LearnedCombiner(model=c.get("model", "linear"),
                                       features=tuple(c.get("features", ("text_score", "network_score"))))
            mp = directory / "learned_model.joblib"
            if mp.is_file():
                import joblib

                combiner._model = joblib.load(mp)
                combiner._fitted = True
        return cls(text_index, network_index, combiner)

    def save(self, directory: Optional[Path] = None) -> Path:
        directory = ensure_dir(Path(directory) if directory
                               else data_embeddings_dir() / "hybrid")
        (directory / "config.json").write_text(
            json.dumps({
                "combiner": self.combiner.config(),
                "text_method": self.text_index.method,
                "network_method": self.network_index.method,
            }, indent=2), encoding="utf-8",
        )
        if isinstance(self.combiner, LearnedCombiner) and self.combiner._fitted:
            import joblib

            joblib.dump(self.combiner._model, directory / "learned_model.joblib")
        return directory
