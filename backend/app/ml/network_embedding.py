"""Network embedding (pipeline stage: Network Embedding).

    Hier-SPCNet -> [Network Embedding] -> Network Similarity -> Hybrid Similarity

Random-walk node embeddings over Hier-SPCNet, following the base paper's approach
(Bhattacharya et al. 2022 use node2vec / metapath2vec style embeddings on the
heterogeneous graph). Implemented with gensim ``Word2Vec`` on walk "sentences" -
no extra dependency:

  * ``node2vec``     - biased random walks (p, q), edge weights = ICF.  BASELINE.
  * ``metapath2vec`` - walks constrained to type schemas over the case/section/
    chapter/act graph (e.g. case-act-case, case-section-chapter-section-case).
    BASE-PAPER-style heterogeneous embedding.

Output: one vector per ``case:`` node -> a ``(n_cases, dim)`` matrix used for
case-to-case network similarity in Step 11.

Walks and Word2Vec are seeded and single-threaded for reproducibility (still not
bit-identical across gensim/OS builds - noted).

No FastAPI / DB dependency.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import networkx as nx
import numpy as np

from app.ml.network_model import case_node
from app.ml.text_similarity import Neighbour, l2_normalize
from app.utils.paths import data_embeddings_dir, ensure_dir

METHODS = ("node2vec", "metapath2vec")

# Undirected type schemas (first == last). Used by metapath2vec.
DEFAULT_METAPATHS: tuple[tuple[str, ...], ...] = (
    ("case", "act", "case"),
    ("case", "section", "case"),
    ("case", "section", "chapter", "section", "case"),
    ("case", "section", "chapter", "act", "chapter", "section", "case"),
    ("case", "case"),  # direct precedent citation
)


# --------------------------------------------------------------------------- #
# Random walks                                                                #
# --------------------------------------------------------------------------- #

def _undirected(graph: nx.DiGraph) -> nx.Graph:
    u = graph.to_undirected(as_view=False)
    return u


def node2vec_walks(
    graph: nx.Graph,
    *,
    num_walks: int = 10,
    walk_length: int = 40,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 42,
) -> list[list[str]]:
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    # precompute neighbour lists + weights + neighbour sets (fast triangle test)
    nbr_list: dict[str, list[str]] = {}
    nbr_w: dict[str, list[float]] = {}
    nbr_set: dict[str, set[str]] = {}
    for n in nodes:
        ns = list(graph.neighbors(n))
        nbr_list[n] = ns
        nbr_w[n] = [float(graph[n][m].get("weight", 1.0)) or 1.0 for m in ns]
        nbr_set[n] = set(ns)

    inv_p, inv_q = 1.0 / p, 1.0 / q
    walks: list[list[str]] = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            while len(walk) < walk_length:
                cur = walk[-1]
                nbrs = nbr_list[cur]
                if not nbrs:
                    break
                if len(walk) == 1:
                    walk.append(rng.choices(nbrs, weights=nbr_w[cur])[0])
                    continue
                prev = walk[-2]
                prev_nbrs = nbr_set[prev]
                weights = [
                    w * (inv_p if n == prev else 1.0 if n in prev_nbrs else inv_q)
                    for n, w in zip(nbrs, nbr_w[cur])
                ]
                walk.append(rng.choices(nbrs, weights=weights)[0])
            if len(walk) >= 2:  # a lone node carries no co-occurrence signal
                walks.append(walk)
    return walks


def metapath_walks(
    graph: nx.Graph,
    *,
    metapaths: Sequence[Sequence[str]] = DEFAULT_METAPATHS,
    num_walks: int = 10,
    walk_length: int = 40,
    seed: int = 42,
) -> list[list[str]]:
    rng = random.Random(seed)
    ntype = {n: d.get("ntype", "?") for n, d in graph.nodes(data=True)}
    by_type: dict[str, list[str]] = defaultdict(list)
    for n, t in ntype.items():
        by_type[t].append(n)

    # precompute per-node {neighbour type -> ([nodes], [weights])} once, so each
    # walk step is an O(1) lookup + weighted choice rather than an O(degree) scan.
    typed_nbrs: dict[str, dict[str, tuple[list, list]]] = {}
    for n in graph.nodes():
        buckets: dict[str, tuple[list, list]] = {}
        for m in graph.neighbors(n):
            nodes_w = buckets.setdefault(ntype[m], ([], []))
            nodes_w[0].append(m)
            nodes_w[1].append(float(graph[n][m].get("weight", 1.0)) or 1.0)
        typed_nbrs[n] = buckets

    walks: list[list[str]] = []
    for _ in range(num_walks):
        for schema in metapaths:
            schema = list(schema)
            unit = len(schema) - 1
            if unit < 1:
                continue
            starts = list(by_type.get(schema[0], []))
            rng.shuffle(starts)
            for start in starts:
                walk = [start]
                cur = start
                for step in range(walk_length - 1):
                    want = schema[(step % unit) + 1]
                    bucket = typed_nbrs[cur].get(want)
                    if not bucket:
                        break
                    cur = rng.choices(bucket[0], weights=bucket[1])[0]
                    walk.append(cur)
                if len(walk) > 2:
                    walks.append(walk)
    return walks


# --------------------------------------------------------------------------- #
# Embedder                                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class NetworkEmbedder:
    method: str = "metapath2vec"
    dimensions: int = 128
    walk_length: int = 40
    num_walks: int = 10
    window: int = 5
    epochs: int = 5
    p: float = 1.0
    q: float = 1.0
    seed: int = 42
    metapaths: tuple = DEFAULT_METAPATHS
    _wv: dict = field(default_factory=dict, repr=False)

    @property
    def provenance(self) -> str:
        return {
            "node2vec": "baseline (ours; node2vec biased walks + skip-gram)",
            "metapath2vec": "base-paper-style heterogeneous embedding "
                            "(Bhattacharya et al. 2022 use metapath2vec-style walks)",
        }[self.method]

    def params(self) -> dict:
        d = {"method": self.method, "dimensions": self.dimensions,
             "walk_length": self.walk_length, "num_walks": self.num_walks,
             "window": self.window, "epochs": self.epochs, "seed": self.seed}
        if self.method == "node2vec":
            d |= {"p": self.p, "q": self.q}
        else:
            d["metapaths"] = ["-".join(m) for m in self.metapaths]
        return d

    def fit(self, graph: nx.DiGraph) -> "NetworkEmbedder":
        from gensim.models import Word2Vec

        g = _undirected(graph)
        if self.method == "node2vec":
            walks = node2vec_walks(g, num_walks=self.num_walks, walk_length=self.walk_length,
                                   p=self.p, q=self.q, seed=self.seed)
        elif self.method == "metapath2vec":
            walks = metapath_walks(g, metapaths=self.metapaths, num_walks=self.num_walks,
                                   walk_length=self.walk_length, seed=self.seed)
        else:
            raise ValueError(f"unknown method {self.method!r}; expected one of {METHODS}")

        model = Word2Vec(
            sentences=walks, vector_size=self.dimensions, window=self.window,
            min_count=0, sg=1, workers=1, seed=self.seed, epochs=self.epochs,
        )
        self._wv = {n: model.wv[n].astype(np.float32) for n in model.wv.index_to_key}
        self._n_walks = len(walks)
        return self

    def case_matrix(self, doc_ids: Sequence[str]) -> tuple[np.ndarray, list[str]]:
        dim = self.dimensions
        rows = []
        missing = []
        for d in doc_ids:
            v = self._wv.get(case_node(d))
            if v is None:
                rows.append(np.zeros(dim, dtype=np.float32))
                missing.append(d)
            else:
                rows.append(v)
        return np.vstack(rows), missing


# --------------------------------------------------------------------------- #
# Index                                                                       #
# --------------------------------------------------------------------------- #

class NetworkEmbeddingIndex:
    def __init__(self, method: str, doc_ids: list[str], matrix: np.ndarray,
                 missing: Optional[list[str]] = None, params: Optional[dict] = None):
        self.method = method
        self.doc_ids = list(doc_ids)
        self.matrix = l2_normalize(matrix)
        self.missing = list(missing or [])
        self.params = params or {}
        self._row = {d: i for i, d in enumerate(self.doc_ids)}

    @classmethod
    def build(cls, graph: nx.DiGraph, corpus, embedder: NetworkEmbedder) -> "NetworkEmbeddingIndex":
        embedder.fit(graph)
        matrix, missing = embedder.case_matrix(corpus.doc_ids)
        return cls(embedder.method, list(corpus.doc_ids), matrix, missing, embedder.params())

    def __len__(self) -> int:
        return len(self.doc_ids)

    def has(self, doc_id: str) -> bool:
        return doc_id in self._row and self.doc_ids[self._row[doc_id]] not in self.missing

    def pair_similarity(self, a: str, b: str) -> float:
        va, vb = self.matrix[self._row[a]], self.matrix[self._row[b]]
        if not va.any() or not vb.any():
            return 0.0
        return float(np.dot(va, vb))

    def most_similar(self, doc_id: str, k: int = 10, *, exclude_self: bool = True) -> list[Neighbour]:
        q = self.matrix[self._row[doc_id]]
        scores = self.matrix @ q
        out: list[Neighbour] = []
        for i in np.argsort(-scores):
            did = self.doc_ids[i]
            if exclude_self and did == doc_id:
                continue
            if did in self.missing:
                continue
            out.append(Neighbour(did, float(scores[i])))
            if len(out) >= k:
                break
        return out

    def save(self, directory: Optional[Path] = None) -> Path:
        directory = ensure_dir(Path(directory) if directory
                               else data_embeddings_dir() / f"network_{self.method}")
        np.save(directory / "matrix.npy", self.matrix)
        (directory / "index.json").write_text(
            json.dumps({"method": self.method, "doc_ids": self.doc_ids,
                        "missing": self.missing, "params": self.params,
                        "embedding_dim": int(self.matrix.shape[1])}, indent=2),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> "NetworkEmbeddingIndex":
        directory = Path(directory)
        meta = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        matrix = np.load(directory / "matrix.npy")
        return cls(meta["method"], meta["doc_ids"], matrix, meta.get("missing"), meta.get("params"))
