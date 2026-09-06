"""build_index.py - Phase 2 P2: build ANN indexes for candidate retrieval.

Reads the text + network embedding matrices and writes FAISS/numpy indexes to
data/indexes/{text,network}/. The recommendation engine uses these to retrieve a
candidate pool instead of scoring the whole corpus.

    python scripts/build_index.py
    CORPUS_SOURCE=db python scripts/build_index.py --backend faiss

Small corpora use an exact flat index; large ones use FAISS-HNSW when faiss-cpu
is installed (otherwise exact numpy - correct, just linear).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.ml.network_embedding import NetworkEmbeddingIndex  # noqa: E402
from app.ml.text_similarity import TextSimilarityIndex, TfidfEmbedder  # noqa: E402
from app.ml.vector_index import VectorIndex  # noqa: E402
from app.utils.dataset import load_corpus  # noqa: E402
from app.utils.paths import data_embeddings_dir, data_indexes_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["auto", "faiss", "numpy"], default="auto")
    p.add_argument("--text-method", default=None)
    p.add_argument("--emb-method", default=None)
    args = p.parse_args(argv)

    s = get_settings()
    emb = data_embeddings_dir()
    out = data_indexes_dir()
    corpus = load_corpus(verify=False)

    tdir = emb / (args.text_method or s.text_similarity_method)
    ti = TextSimilarityIndex.load(tdir) if (tdir / "index.json").is_file() else \
        TextSimilarityIndex.build(corpus, TfidfEmbedder())
    ei = NetworkEmbeddingIndex.load(emb / f"network_{args.emb_method or s.network_embedding_method}")

    for name, idx in (("text", ti), ("network", ei)):
        t0 = time.time()
        vi = VectorIndex(list(idx.doc_ids), idx.matrix, backend=args.backend)
        vi.save(out / name)
        print(f"  {name:<8} {len(vi.doc_ids)} x {vi.dim}  backend={vi.backend}  "
              f"({time.time() - t0:.1f}s)")

    print(f"\n[ OK ] wrote {out.relative_to(_REPO_ROOT).as_posix()}/{{text,network}}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
