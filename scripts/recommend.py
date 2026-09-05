"""recommend.py - Step 13: Top-K hybrid case recommendation (CLI demo).

    python scripts/recommend.py --case 1955_S_34 --k 5
    python scripts/recommend.py --file path/to/case.pdf --k 10 --year-min 1990
    python scripts/recommend.py --case 2005_S_628 --k 5 --explain 2010_B_40

Loads the Step 8/10/11/12 artifacts from data/embeddings/. The text embedder is
reloaded if it was saved, otherwise rebuilt from the corpus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.ml.hybrid_similarity import HybridSimilarityModel, LearnedCombiner, make_combiner  # noqa: E402
from app.ml.network_embedding import NetworkEmbeddingIndex  # noqa: E402
from app.ml.network_similarity import NetworkSimilarityIndex  # noqa: E402
from app.ml.recommendation import Filters, RecommendationEngine  # noqa: E402
from app.ml.text_similarity import TextSimilarityIndex, TfidfEmbedder  # noqa: E402
from app.utils.dataset import load_corpus  # noqa: E402
from app.utils.paths import data_embeddings_dir  # noqa: E402


def _load_engine(text_method: str, emb_method: str, hybrid_method: str, corpus):
    tdir = data_embeddings_dir() / text_method
    ti = TextSimilarityIndex.load(tdir) if (tdir / "index.json").is_file() else None
    if ti is None or ti.embedder is None:
        print("(rebuilding text index - embedder not persisted)")
        ti = TextSimilarityIndex.build(corpus, TfidfEmbedder())
    ei = NetworkEmbeddingIndex.load(data_embeddings_dir() / f"network_{emb_method}")
    ni = NetworkSimilarityIndex.load(data_embeddings_dir() / "network_similarity", ei, corpus=corpus)

    combiner = make_combiner("weighted") if hybrid_method == "weighted" else LearnedCombiner(model="linear")
    model = HybridSimilarityModel(ti, ni, combiner)
    if isinstance(combiner, LearnedCombiner):
        model.fit_learned(corpus, split="validation")
    return RecommendationEngine(corpus, model)


def _print(result, corpus) -> None:
    print(f"\nquery: {result.query}   method: {result.method}   "
          f"candidates: {result.n_candidates}   filtered out: {result.n_filtered_out}")
    if not result.query_in_corpus:
        print("(uploaded case - network score is structural overlap only, no graph node)")
    for r in result.results:
        print(f"\n #{r.rank}  hybrid={r.hybrid_score:.3f}   {r.doc_id}   ({r.year})")
        print(f"     {r.title or '(untitled)'}")
        print(f"     text={r.text_score:.3f}  network={r.network_score:.3f} "
              f"(emb {r.embedding_score:.2f} / statute {r.statute_overlap:.2f} / cite {r.citation_overlap:.2f})")
        print(f"     {r.rationale}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--case", help="corpus doc_id to find similar cases for")
    src.add_argument("--file", type=Path, help="PDF/txt of an uploaded case")
    src.add_argument("--text", help="raw case text")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--year-min", type=int, default=None)
    p.add_argument("--year-max", type=int, default=None)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--text-method", default="tfidf")
    p.add_argument("--emb-method", default="metapath2vec")
    p.add_argument("--hybrid", choices=["weighted", "learned"], default="learned")
    p.add_argument("--explain", metavar="TARGET_DOC_ID", default=None)
    args = p.parse_args(argv)

    corpus = load_corpus(verify=True)
    engine = _load_engine(args.text_method, args.emb_method, args.hybrid, corpus)
    filters = Filters(year_min=args.year_min, year_max=args.year_max,
                      min_hybrid_score=args.min_score)

    if args.case:
        if args.case not in corpus:
            print(f"[FAIL] {args.case!r} not in corpus", file=sys.stderr)
            return 1
        result = engine.recommend_for_case(args.case, k=args.k, filters=filters)
    else:
        if args.file:
            from app.ml.text_extraction import extract_text

            text = extract_text(args.file).raw_text
        else:
            text = args.text
        result = engine.recommend_for_text(text, k=args.k, filters=filters)

    _print(result, corpus)

    if args.explain and args.case:
        ex = engine.explain_pair(args.case, args.explain)
        print(f"\n-- pairwise explanation {args.case} vs {args.explain} --")
        for key in ("hybrid_score", "text_score", "network_score", "method"):
            print(f"   {key}: {ex[key]}")
        nb = ex["network_breakdown"]
        print(f"   shared acts: {[a['act_id'] for a in nb['shared_acts']]}")
        print(f"   shared sections: {[s['section_key'] for s in nb['shared_sections']]}")
        print(f"   shared cited cases: {nb['shared_citation_doc_ids']}")

    print("\n[ OK ] Step 13 recommendation engine works. Proceed to Step 17 (Database).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
