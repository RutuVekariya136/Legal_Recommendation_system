"""Approximate nearest-neighbour index over an embedding matrix (Phase 2, P2).

Used to turn O(N) "score every corpus document" into O(candidate_pool) - the
recommendation engine ANN-retrieves a candidate pool with text + network
embeddings, then hybrid-re-ranks only that pool.

Backend: FAISS if installed (`pip install faiss-cpu`), otherwise a numpy
brute-force fallback (exact, fine up to ~100k rows, just not sub-linear). All
vectors are treated as **cosine** similarity (rows are L2-normalised on build).

No FastAPI / DB dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import sparse

from app.ml.text_similarity import Neighbour, l2_normalize

try:  # pragma: no cover - depends on optional install
    import faiss

    _HAVE_FAISS = True
except ImportError:  # pragma: no cover
    faiss = None
    _HAVE_FAISS = False

# below this many rows an exact flat index is cheap and best
_FLAT_THRESHOLD = 12_000


class VectorIndex:
    def __init__(self, doc_ids: list[str], matrix, *, backend: str = "auto",
                 hnsw_m: int = 32):
        self.doc_ids = list(doc_ids)
        self._row = {d: i for i, d in enumerate(self.doc_ids)}
        if sparse.issparse(matrix):
            self.matrix = l2_normalize(matrix)          # stays sparse
        else:
            self.matrix = l2_normalize(np.ascontiguousarray(matrix, dtype=np.float32))
        self.is_sparse = sparse.issparse(self.matrix)
        self.dim = int(self.matrix.shape[1])
        self.hnsw_m = hnsw_m
        self.backend = self._resolve_backend(backend)
        self._faiss = None
        if self.backend == "faiss":
            self._build_faiss()

    # -- build --------------------------------------------------------
    def _resolve_backend(self, backend: str) -> str:
        # FAISS needs dense low-dim vectors; a high-dim sparse TF-IDF matrix
        # uses the exact sparse mat-vec path instead (fast enough to ~100k rows).
        if getattr(self, "is_sparse", False) or backend == "numpy":
            return "numpy"
        if backend == "faiss" and not _HAVE_FAISS:
            raise RuntimeError("faiss requested but faiss-cpu is not installed")
        return "faiss" if (_HAVE_FAISS and backend in ("auto", "faiss")) else "numpy"

    def _build_faiss(self) -> None:
        n = self.matrix.shape[0]
        if n <= _FLAT_THRESHOLD:
            index = faiss.IndexFlatIP(self.dim)
        else:
            index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
        index.add(self.matrix)
        self._faiss = index

    # -- search ------------------------------------------------------
    def _search_vec(self, q, k: int) -> list[Neighbour]:
        if sparse.issparse(q):
            q = q.toarray()
        q = np.asarray(q, dtype=np.float32).ravel()
        n = np.linalg.norm(q)
        if not n:
            return []
        q = q / n
        if self.backend == "faiss":
            scores, idx = self._faiss.search(q.reshape(1, -1), min(k, len(self.doc_ids)))
            return [Neighbour(self.doc_ids[i], float(s))
                    for s, i in zip(scores[0], idx[0]) if i != -1]
        sims = np.asarray(self.matrix @ q).ravel()
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [Neighbour(self.doc_ids[i], float(sims[i])) for i in top]

    def search(self, query_vec, k: int = 20) -> list[Neighbour]:
        return self._search_vec(query_vec, k)

    def most_similar(self, doc_id: str, k: int = 20, *, exclude_self: bool = True) -> list[Neighbour]:
        hits = self._search_vec(self.matrix[self._row[doc_id]], k + (1 if exclude_self else 0))
        if exclude_self:
            hits = [h for h in hits if h.doc_id != doc_id]
        return hits[:k]

    def has(self, doc_id: str) -> bool:
        return doc_id in self._row

    # -- persistence ----------------------------------------------
    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.is_sparse:
            sparse.save_npz(directory / "matrix.npz", self.matrix)
        else:
            np.save(directory / "matrix.npy", self.matrix)
        (directory / "meta.json").write_text(
            json.dumps({"doc_ids": self.doc_ids, "dim": self.dim, "sparse": self.is_sparse,
                        "backend": self.backend, "hnsw_m": self.hnsw_m}), encoding="utf-8")
        if self.backend == "faiss":
            faiss.write_index(self._faiss, str(directory / "faiss.index"))
        return directory

    @classmethod
    def load(cls, directory: Path, *, backend: str = "auto") -> "VectorIndex":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        npz = directory / "matrix.npz"
        matrix = sparse.load_npz(npz) if npz.is_file() else np.load(directory / "matrix.npy")
        idx = cls.__new__(cls)
        idx.doc_ids = meta["doc_ids"]
        idx._row = {d: i for i, d in enumerate(idx.doc_ids)}
        idx.matrix = matrix
        idx.is_sparse = sparse.issparse(matrix)
        idx.dim = meta["dim"]
        idx.hnsw_m = meta.get("hnsw_m", 32)
        want = backend if backend != "auto" else meta.get("backend", "auto")
        idx.backend = idx._resolve_backend(want)
        idx._faiss = None
        fp = directory / "faiss.index"
        if idx.backend == "faiss" and fp.is_file():
            idx._faiss = faiss.read_index(str(fp))
        elif idx.backend == "faiss":
            idx._build_faiss()
        return idx


def build_from_index(sim_index, *, backend: str = "auto") -> VectorIndex:
    """Build a VectorIndex from a TextSimilarityIndex / NetworkEmbeddingIndex."""
    return VectorIndex(list(sim_index.doc_ids), sim_index.matrix, backend=backend)
