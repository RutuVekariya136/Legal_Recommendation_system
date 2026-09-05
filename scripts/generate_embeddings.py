"""generate_embeddings.py - Step 8: fit a text embedder and save the doc matrix.

    python scripts/generate_embeddings.py --method tfidf     # baseline (ours)
    python scripts/generate_embeddings.py --method doc2vec   # base paper (Bhattacharya 2022)
    python scripts/generate_embeddings.py --method sbert      # our enhancement (needs requirements-ml.txt)

Writes data/embeddings/<method>/ (matrix.npy + index.json).

--evaluate runs a TEXT-ONLY sanity correlation of cosine vs the expert scores on
the real 190 pairs. This is NOT the final evaluation (Step 22 does the hybrid
model); it just shows the embedder is wired correctly. Numbers are printed
exactly as computed - never rounded up or invented.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.ml.text_similarity import METHODS, TextSimilarityIndex, make_embedder  # noqa: E402
from app.utils.dataset import load_corpus  # noqa: E402


def _evaluate(index: TextSimilarityIndex, corpus) -> None:
    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        print("  (scipy missing - skipping correlation)")
        return
    print("\n-- TEXT-ONLY sanity correlation (cosine vs expert score) --")
    print("   NOT the final hybrid evaluation; see Step 22.")
    for split in ("validation", "test"):
        pairs = corpus.pairs_by_split[split]
        sims = [index.pair_similarity(p.doc_id_1, p.doc_id_2) for p in pairs]
        gold = [p.score for p in pairs]
        pr = pearsonr(sims, gold)
        sr = spearmanr(sims, gold)
        print(f"   {split:<11} n={len(pairs):<3} "
              f"Pearson r={pr.statistic:+.3f} (p={pr.pvalue:.3f})  "
              f"Spearman rho={sr.statistic:+.3f} (p={sr.pvalue:.3f})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="tfidf")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--neighbours", type=int, default=5, help="print N nearest for a sample doc")
    args = parser.parse_args(argv)

    corpus = load_corpus(verify=True)
    embedder = make_embedder(args.method)
    print(f"method:     {embedder.name}")
    print(f"provenance: {embedder.provenance}")
    print(f"params:     {embedder.params}")

    t0 = time.time()
    index = TextSimilarityIndex.build(corpus, embedder)
    print(f"\nbuilt {len(index)} x {index.matrix.shape[1]} matrix in {time.time() - t0:.1f}s")

    sample = corpus.doc_ids[len(corpus) // 2]
    print(f"\nnearest to {sample} ({corpus[sample].title or 'untitled'}):")
    for nb in index.most_similar(sample, args.neighbours):
        print(f"   {nb.score:.3f}  {nb.doc_id}  {corpus[nb.doc_id].title or ''}")

    if args.evaluate:
        _evaluate(index, corpus)

    if not args.no_save:
        out = index.save()
        print(f"\n[ OK ] saved -> {out.relative_to(_REPO_ROOT).as_posix()}/")
    print("       (Step 8 provides the embedders; Step 12 combines text + network.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
