"""build_hybrid.py - Step 12: combine text + network into the hybrid score.

    python scripts/build_hybrid.py --method weighted --alpha 0.6 --beta 0.4
    python scripts/build_hybrid.py --method learned --model linear --evaluate

Loads:
    data/embeddings/<text-method>/           (Step 8)
    data/embeddings/network_<emb-method>/     (Step 10)
    data/embeddings/network_similarity/       (Step 11)

For --method learned the combiner is trained on the validation expert pairs
(with k-fold CV for an honest in-sample estimate) and evaluated on test.
All correlation / MSE numbers are printed exactly as computed - Step 22 does the
full benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.ml.hybrid_similarity import (  # noqa: E402
    METHODS,
    HybridSimilarityModel,
    LearnedCombiner,
    make_combiner,
)
from app.ml.network_embedding import NetworkEmbeddingIndex  # noqa: E402
from app.ml.network_similarity import NetworkSimilarityIndex  # noqa: E402
from app.ml.text_similarity import TextSimilarityIndex  # noqa: E402
from app.utils.dataset import load_corpus  # noqa: E402
from app.utils.paths import data_embeddings_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="learned")
    parser.add_argument("--text-method", default="tfidf")
    parser.add_argument("--emb-method", default="metapath2vec")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--model", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--net-method", default="blend", choices=["blend", "embedding"])
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    corpus = load_corpus(verify=True)
    ti = TextSimilarityIndex.load(data_embeddings_dir() / args.text_method)
    ei = NetworkEmbeddingIndex.load(data_embeddings_dir() / f"network_{args.emb_method}")
    ni = NetworkSimilarityIndex.load(
        data_embeddings_dir() / "network_similarity", ei, corpus=corpus
    )
    ni.method = args.net_method

    if args.method == "weighted":
        combiner = make_combiner("weighted", alpha=args.alpha, beta=args.beta)
    else:
        combiner = make_combiner("learned", model=args.model)
    model = HybridSimilarityModel(ti, ni, combiner)

    print(f"text={args.text_method}  network={args.emb_method}/{args.net_method}  "
          f"hybrid={args.method}")

    if isinstance(combiner, LearnedCombiner):
        info = model.fit_learned(corpus, split="validation", folds=5)
        print(f"\nlearned combiner ({args.model}) - trained on {info['n_pairs']} validation pairs")
        print(f"  {info['folds']}-fold CV:  Pearson {info['cv_pearson']:+.3f}  "
              f"Spearman {info['cv_spearman']:+.3f}  MSE {info['cv_mse']:.4f}")
        cfg = combiner.config()
        if "coef_" in cfg:
            print(f"  weights: {dict(zip(combiner.features, cfg['coef_']))}  "
                  f"intercept {cfg['intercept_']:+.3f}")
    else:
        print(f"  alpha={combiner.alpha:.3f}  beta={combiner.beta:.3f}")

    if args.evaluate:
        print("\n-- hybrid correlation vs expert score --")
        print("   (validation is in-sample for --method learned; see CV above and Step 22)")
        for split in ("validation", "test"):
            r = model.evaluate_correlation(corpus, split)
            print(f"   {split:<11} n={r['n_pairs']:<3} Pearson {r['pearson']:+.3f}  "
                  f"Spearman {r['spearman']:+.3f}  MSE {r['mse']:.4f}")

    sample = corpus.doc_ids[len(corpus) // 2]
    print(f"\ntop-5 hybrid neighbours of {sample} ({corpus[sample].title or 'untitled'}):")
    for nb in model.most_similar(sample, 5):
        ex = model.explain(sample, nb.doc_id)
        print(f"   {nb.score:.3f}  {nb.doc_id}  (text {ex.text_score:.2f} / net {ex.network_score:.2f})  "
              f"{corpus[nb.doc_id].title or ''}")

    if not args.no_save:
        out = model.save()
        print(f"\n[ OK ] saved -> {out.relative_to(_REPO_ROOT).as_posix()}/")
    print("       Proceed to Step 13 (Recommendation Engine).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
