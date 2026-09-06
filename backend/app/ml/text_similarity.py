"""Text similarity (pipeline stage: Text Embedding -> Text Similarity).

    ... -> Preprocessing -> [Text Embedding -> Text Similarity] -> Hybrid Similarity

Three interchangeable embedders behind one interface:

  * ``TfidfEmbedder``   - TF-IDF bag of words + cosine.        BASELINE (ours; standard).
  * ``Doc2VecEmbedder`` - gensim Doc2Vec (PV-DM).              BASE PAPER method
                                                               (Bhattacharya et al. 2022).
  * ``SbertEmbedder``   - Sentence-BERT, long-doc mean-pooled. OUR ENHANCEMENT
                                                               (needs requirements-ml.txt).

Every embedder is fit/queried from plain data - no FastAPI, no DB. Vectors are
L2-normalised so cosine similarity is a dot product.

The base paper did not release code or exact hyper-parameters; the Doc2Vec
settings here are our reasonable choices following its described PV-DM method and
are labelled as such.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from app.ml.preprocessing import DEFAULT_CONFIG, PreprocessConfig, preprocess_text
from app.utils.paths import data_embeddings_dir, data_processed_dir, ensure_dir

METHODS = ("tfidf", "doc2vec", "sbert")


@dataclass(frozen=True)
class TextDoc:
    doc_id: str
    tokens: list[str]
    text: str


# --------------------------------------------------------------------------- #
# Corpus preparation (reuses Step 4 artifacts when present)                    #
# --------------------------------------------------------------------------- #

def prepare_documents(
    corpus,
    *,
    config: PreprocessConfig = DEFAULT_CONFIG,
    use_cache: bool = True,
) -> list[TextDoc]:
    """Build (doc_id, tokens, clean_text) for every corpus document.

    Uses data/processed/{tokens,clean}/ from Step 4 when available and complete,
    otherwise runs preprocessing on the fly.
    """
    tok_dir = data_processed_dir() / "tokens"
    clean_dir = data_processed_dir() / "clean"
    out: list[TextDoc] = []
    for doc_id in corpus.doc_ids:
        tok_p, clean_p = tok_dir / f"{doc_id}.json", clean_dir / f"{doc_id}.txt"
        if use_cache and tok_p.is_file() and clean_p.is_file():
            tokens = json.loads(tok_p.read_text(encoding="utf-8"))
            text = clean_p.read_text(encoding="utf-8")
        else:
            pd = preprocess_text(corpus.text(doc_id), doc_id=doc_id, config=config)
            tokens, text = pd.tokens, pd.clean_text
        out.append(TextDoc(doc_id=doc_id, tokens=list(tokens), text=text))
    return out


# --------------------------------------------------------------------------- #
# Embedder interface                                                          #
# --------------------------------------------------------------------------- #

def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class BaseTextEmbedder(abc.ABC):
    name: str = "base"
    provenance: str = ""
    params: dict

    @abc.abstractmethod
    def fit(self, docs: Sequence[TextDoc]) -> "BaseTextEmbedder": ...

    @abc.abstractmethod
    def embed_corpus(self) -> np.ndarray:
        """(n_docs, dim), L2-normalised, row order == fit() order."""

    @abc.abstractmethod
    def embed_text(self, tokens: Sequence[str], text: str) -> np.ndarray:
        """(dim,), L2-normalised - for an unseen document."""

    @property
    @abc.abstractmethod
    def embedding_dim(self) -> int: ...

    def config(self) -> dict:
        return {"name": self.name, "provenance": self.provenance, "params": self.params}


# --------------------------------------------------------------------------- #
# TF-IDF - baseline                                                           #
# --------------------------------------------------------------------------- #

class TfidfEmbedder(BaseTextEmbedder):
    name = "tfidf"
    provenance = "baseline (ours; standard bag-of-words)"

    def __init__(self, *, ngram_range: tuple[int, int] = (1, 2), min_df: int = 2,
                 max_df: float = 0.9, sublinear_tf: bool = True, max_features: Optional[int] = 50000):
        self.params = dict(ngram_range=list(ngram_range), min_df=min_df, max_df=max_df,
                           sublinear_tf=sublinear_tf, max_features=max_features)
        self._vectorizer = None
        self._matrix = None

    def fit(self, docs: Sequence[TextDoc]) -> "TfidfEmbedder":
        from sklearn.feature_extraction.text import TfidfVectorizer

        # tokens from Step 4 are already clean lowercase alpha; join and let the
        # vectorizer re-tokenise so ngram_range / min_df / max_df apply normally.
        self._vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=tuple(self.params["ngram_range"]),
            min_df=self.params["min_df"], max_df=self.params["max_df"],
            sublinear_tf=self.params["sublinear_tf"],
            max_features=self.params["max_features"], norm="l2",
        )
        self._matrix = self._vectorizer.fit_transform(" ".join(d.tokens) for d in docs)
        return self

    def embed_corpus(self) -> np.ndarray:
        return np.asarray(self._matrix.todense(), dtype=np.float32)

    def embed_text(self, tokens: Sequence[str], text: str) -> np.ndarray:
        vec = self._vectorizer.transform([" ".join(tokens)])
        return np.asarray(vec.todense(), dtype=np.float32).ravel()

    @property
    def embedding_dim(self) -> int:
        return int(self._matrix.shape[1]) if self._matrix is not None else 0


# --------------------------------------------------------------------------- #
# Doc2Vec - base paper                                                        #
# --------------------------------------------------------------------------- #

class Doc2VecEmbedder(BaseTextEmbedder):
    name = "doc2vec"
    provenance = "base paper: Doc2Vec / PV-DM (Bhattacharya et al. 2022)"

    def __init__(self, *, vector_size: int = 200, window: int = 15, min_count: int = 5,
                 epochs: int = 40, dm: int = 1, negative: int = 10, seed: int = 42):
        # The paper describes Doc2Vec but did not publish settings - these are ours.
        self.params = dict(vector_size=vector_size, window=window, min_count=min_count,
                           epochs=epochs, dm=dm, negative=negative, seed=seed)
        self._model = None
        self._doc_ids: list[str] = []

    def fit(self, docs: Sequence[TextDoc]) -> "Doc2VecEmbedder":
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument

        tagged = [TaggedDocument(d.tokens, [d.doc_id]) for d in docs]
        self._doc_ids = [d.doc_id for d in docs]
        p = self.params
        self._model = Doc2Vec(
            documents=tagged, vector_size=p["vector_size"], window=p["window"],
            min_count=p["min_count"], epochs=p["epochs"], dm=p["dm"],
            negative=p["negative"], seed=p["seed"], workers=1,  # workers=1 -> reproducible
        )
        return self

    def embed_corpus(self) -> np.ndarray:
        return l2_normalize(np.vstack([self._model.dv[d] for d in self._doc_ids]))

    def embed_text(self, tokens: Sequence[str], text: str) -> np.ndarray:
        vec = self._model.infer_vector(list(tokens), epochs=self.params["epochs"])
        return l2_normalize(vec[None, :])[0]

    @property
    def embedding_dim(self) -> int:
        return int(self.params["vector_size"])


# --------------------------------------------------------------------------- #
# Sentence-BERT - our enhancement                                             #
# --------------------------------------------------------------------------- #

class SbertEmbedder(BaseTextEmbedder):
    """Sentence-Transformer text embeddings, long docs chunked + mean-pooled.

    OUR ENHANCEMENT (not in the base paper). ``model_name`` is any HuggingFace
    model: ``all-MiniLM-L6-v2`` (default, general), or a legal-domain encoder
    such as ``law-ai/InLegalBERT`` / ``law-ai/InCaseLawBERT`` (SentenceTransformer
    wraps a bare BERT with a mean-pooling layer - fine for similarity). Needs
    ``pip install -r backend/requirements-ml.txt`` and, for a corpus of any size,
    a GPU.
    """

    name = "sbert"
    provenance = "our enhancement (not in the base paper)"

    def __init__(self, *, model_name: str = "all-MiniLM-L6-v2", chunk_words: int = 320,
                 max_chunks: int = 40, batch_size: int = 64):
        self.params = dict(model_name=model_name, chunk_words=chunk_words,
                           max_chunks=max_chunks, batch_size=batch_size)
        self._model = None
        self._corpus_vectors: Optional[np.ndarray] = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on optional install
                raise ImportError(
                    "SbertEmbedder needs sentence-transformers: "
                    "pip install -r backend/requirements-ml.txt"
                ) from exc
            self._model = SentenceTransformer(self.params["model_name"])
        return self._model

    def _chunks(self, text: str) -> list[str]:
        words = text.split()
        n, step = self.params["chunk_words"], self.params["chunk_words"]
        out = [" ".join(words[i:i + n]) for i in range(0, len(words), step)]
        return out[: self.params["max_chunks"]] or [""]

    def _embed_one(self, text: str) -> np.ndarray:
        model = self._load_model()
        chunk_vecs = model.encode(self._chunks(text), batch_size=self.params["batch_size"],
                                  show_progress_bar=False, normalize_embeddings=True)
        return l2_normalize(np.asarray(chunk_vecs, dtype=np.float32).mean(axis=0)[None, :])[0]

    def fit(self, docs: Sequence[TextDoc]) -> "SbertEmbedder":
        self._corpus_vectors = np.vstack([self._embed_one(d.text) for d in docs])
        return self

    def embed_corpus(self) -> np.ndarray:
        return self._corpus_vectors

    def embed_text(self, tokens: Sequence[str], text: str) -> np.ndarray:
        return self._embed_one(text)

    @property
    def embedding_dim(self) -> int:
        if self._corpus_vectors is not None:
            return int(self._corpus_vectors.shape[1])
        return int(self._load_model().get_sentence_embedding_dimension())


def make_embedder(method: str, **kwargs) -> BaseTextEmbedder:
    method = method.lower()
    if method == "tfidf":
        return TfidfEmbedder(**kwargs)
    if method == "doc2vec":
        return Doc2VecEmbedder(**kwargs)
    if method == "sbert":
        return SbertEmbedder(**kwargs)
    raise ValueError(f"unknown text-similarity method {method!r}; expected one of {METHODS}")


# --------------------------------------------------------------------------- #
# Similarity index                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class Neighbour:
    doc_id: str
    score: float


class TextSimilarityIndex:
    """Fitted embedder + the (n_docs, dim) corpus matrix + doc-id lookup."""

    def __init__(self, method: str, doc_ids: list[str], matrix: np.ndarray,
                 embedder: Optional[BaseTextEmbedder] = None, embedder_config: Optional[dict] = None):
        self.method = method
        self.doc_ids = list(doc_ids)
        self.matrix = l2_normalize(matrix)
        self._row = {d: i for i, d in enumerate(self.doc_ids)}
        self.embedder = embedder
        self.embedder_config = embedder_config or (embedder.config() if embedder else {})

    # -- build / persist ------------------------------------------------
    @classmethod
    def build(cls, corpus, embedder: BaseTextEmbedder, *,
              config: PreprocessConfig = DEFAULT_CONFIG) -> "TextSimilarityIndex":
        docs = prepare_documents(corpus, config=config)
        embedder.fit(docs)
        matrix = embedder.embed_corpus()
        return cls(embedder.name, [d.doc_id for d in docs], matrix, embedder=embedder)

    def save(self, directory: Optional[Path] = None, *, with_embedder: bool = True) -> Path:
        directory = ensure_dir(Path(directory) if directory else data_embeddings_dir() / self.method)
        np.save(directory / "matrix.npy", self.matrix)
        embedder_saved = False
        if with_embedder and self.embedder is not None:
            try:
                import joblib

                joblib.dump(self.embedder, directory / "embedder.joblib")
                embedder_saved = True
            except Exception:  # noqa: BLE001 - e.g. SBERT model not picklable; not fatal
                embedder_saved = False
        (directory / "index.json").write_text(
            json.dumps({"method": self.method, "doc_ids": self.doc_ids,
                        "embedding_dim": int(self.matrix.shape[1]),
                        "embedder_config": self.embedder_config,
                        "embedder_saved": embedder_saved}, indent=2),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> "TextSimilarityIndex":
        directory = Path(directory)
        meta = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        matrix = np.load(directory / "matrix.npy")
        embedder = None
        emb_path = directory / "embedder.joblib"
        if emb_path.is_file():
            try:
                import joblib

                embedder = joblib.load(emb_path)
            except Exception:  # noqa: BLE001
                embedder = None
        idx = cls(meta["method"], meta["doc_ids"], matrix, embedder=embedder)
        idx.embedder_config = meta.get("embedder_config", {})
        return idx

    # -- queries -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.doc_ids)

    def has(self, doc_id: str) -> bool:
        return doc_id in self._row

    def vector(self, doc_id: str) -> np.ndarray:
        return self.matrix[self._row[doc_id]]

    def pair_similarity(self, a: str, b: str) -> float:
        return float(np.dot(self.matrix[self._row[a]], self.matrix[self._row[b]]))

    def _rank(self, q: np.ndarray, k: int, exclude: set[str]) -> list[Neighbour]:
        scores = self.matrix @ l2_normalize(q[None, :])[0]
        order = np.argsort(-scores)
        out: list[Neighbour] = []
        for i in order:
            did = self.doc_ids[i]
            if did in exclude:
                continue
            out.append(Neighbour(did, float(scores[i])))
            if len(out) >= k:
                break
        return out

    def most_similar(self, doc_id: str, k: int = 10, *, exclude_self: bool = True) -> list[Neighbour]:
        q = self.matrix[self._row[doc_id]]
        return self._rank(q, k, {doc_id} if exclude_self else set())

    def query_vector(self, vector: np.ndarray, k: int = 10, *, exclude: Optional[set[str]] = None) -> list[Neighbour]:
        return self._rank(np.asarray(vector, dtype=np.float32), k, exclude or set())

    def query_text(self, text: str, k: int = 10, *,
                   config: PreprocessConfig = DEFAULT_CONFIG) -> list[Neighbour]:
        if self.embedder is None:
            raise RuntimeError("this index was loaded without an embedder; rebuild to query raw text")
        pd = preprocess_text(text, config=config)
        vec = self.embedder.embed_text(pd.tokens, pd.clean_text)
        return self._rank(np.asarray(vec, dtype=np.float32), k, set())

    def query_scores(self, text: str, *, config: PreprocessConfig = DEFAULT_CONFIG) -> dict[str, float]:
        """Cosine of `text` against every corpus doc, as {doc_id: score}."""
        if self.embedder is None:
            raise RuntimeError("this index was loaded without an embedder; rebuild to query raw text")
        pd = preprocess_text(text, config=config)
        vec = l2_normalize(np.asarray(self.embedder.embed_text(pd.tokens, pd.clean_text),
                                     dtype=np.float32)[None, :])[0]
        scores = self.matrix @ vec
        return {d: float(s) for d, s in zip(self.doc_ids, scores)}

    def scores_for(self, doc_id: str) -> dict[str, float]:
        """Cosine of an indexed doc against every corpus doc."""
        scores = self.matrix @ self.matrix[self._row[doc_id]]
        return {d: float(s) for d, s in zip(self.doc_ids, scores)}
