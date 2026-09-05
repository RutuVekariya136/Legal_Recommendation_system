"""generate_network_embeddings.py - Step 10: random-walk embeddings over Hier-SPCNet.

    python scripts/generate_network_embeddings.py --method metapath2vec   # base-paper style
    python scripts/generate_network_embeddings.py --method node2vec       # baseline

Loads data/graphs/hier_spcnet.json if present (run scripts/build_graph.py first),
otherwise builds it. Writes data/embeddings/network_<method>/.

--evaluate prints a NETWORK-ONLY sanity correlation of the case-node cosine vs the
expert scores on the real 190 pairs. Not the hybrid evaluation (Step 22); just a
wiring check. Printed exactly as computed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.ml.network_embedding import METHODS, NetworkEmbedder, NetworkEmbeddingIndex  # noqa: E402
from app.ml.network_model import build_hier_spcnet, build_pcnet, from_node_link  # noqa: E402
from app.utils.dataset import load_corpus  # noqa: E402
from app.utils.paths import data_graphs_dir  # noqa: E402


def _load_or_build_graph(which: str, corpus):
    path = data_graphs_dir() / (f"{which}.json")
    if path.is_file():
        print(f"loading {path.relative_to(_REPO_ROOT).as_posix()}")
        return from_node_link(json.loads(path.read_text(encoding="utf-8")))
    print(f"{path.name} not found - building {which}")
    return (build_hier_spcnet if which == "hier_spcnet" else build_pcnet)(corpus, min_score=90)


def _evaluate(index, corpus) -> None:
    from scipy.stats import pearsonr, spearmanr

    print("\n-- NETWORK-ONLY sanity correlation (case cosine vs expert score) --")
    print("   NOT the final hybrid evaluation; see Step 22.")
    for split in ("validation", "test"):
        pairs = corpus.pairs_by_split[split]
        sims = [index.pair_similarity(p.doc_id_1, p.doc_id_2) for p in pairs]
        gold = [p.score for p in pairs]
        pr, sr = pearsonr(sims, gold), spearmanr(sims, gold)
        n_zero = sum(1 for s in sims if s == 0.0)
        print(f"   {split:<11} n={len(pairs):<3} Pearson r={pr.statistic:+.3f}  "
              f"Spearman rho={sr.statistic:+.3f}   ({n_zero} pairs with an isolated case)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="metapath2vec")
    parser.add_argument("--graph", choices=["hier_spcnet", "pcnet"], default="hier_spcnet")
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--num-walks", type=int, default=10)
    parser.add_argument("--walk-length", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    corpus = load_corpus(verify=True)
    graph = _load_or_build_graph(args.graph, corpus)

    embedder = NetworkEmbedder(
        method=args.method, dimensions=args.dimensions, num_walks=args.num_walks,
        walk_length=args.walk_length, epochs=args.epochs,
    )
    print(f"method:     {embedder.method}")
    print(f"provenance: {embedder.provenance}")
    print(f"params:     {embedder.params()}")

    t0 = time.time()
    index = NetworkEmbeddingIndex.build(graph, corpus, embedder)
    print(f"\nbuilt {len(index)} x {index.matrix.shape[1]} case matrix in {time.time() - t0:.1f}s")
    print(f"case nodes with no embedding (isolated in graph): {len(index.missing)}")

    sample = next(d for d in corpus.doc_ids if index.has(d))
    print(f"\nnearest to {sample} ({corpus[sample].title or 'untitled'}):")
    for nb in index.most_similar(sample, 5):
        print(f"   {nb.score:.3f}  {nb.doc_id}  {corpus[nb.doc_id].title or ''}")

    if args.evaluate:
        _evaluate(index, corpus)

    if not args.no_save:
        out = index.save()
        print(f"\n[ OK ] saved -> {out.relative_to(_REPO_ROOT).as_posix()}/")
    print("       Proceed to Step 11 (Network Similarity).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
