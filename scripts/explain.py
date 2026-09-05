"""explain.py - Step 14: full explainability for one case pair.

    python scripts/explain.py --query 2005_S_628 --target 2010_B_40
    python scripts/explain.py --file some_case.pdf --target 2009_M_311

Prints the score breakdown, per-component contribution attribution, shared
statutes (with labels), shared cited cases, and a natural-language summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.ml.explainability import Explainer  # noqa: E402
from app.ml.hybrid_similarity import HybridSimilarityModel, LearnedCombiner, make_combiner  # noqa: E402
from app.ml.network_embedding import NetworkEmbeddingIndex  # noqa: E402
from app.ml.network_similarity import NetworkSimilarityIndex  # noqa: E402
from app.ml.text_similarity import TextSimilarityIndex, TfidfEmbedder  # noqa: E402
from app.utils.dataset import load_corpus  # noqa: E402
from app.utils.paths import data_embeddings_dir  # noqa: E402


def _load_model(corpus, hybrid: str):
    tdir = data_embeddings_dir() / "tfidf"
    ti = TextSimilarityIndex.load(tdir) if (tdir / "index.json").is_file() else None
    if ti is None or ti.embedder is None:
        ti = TextSimilarityIndex.build(corpus, TfidfEmbedder())
    ei = NetworkEmbeddingIndex.load(data_embeddings_dir() / "network_metapath2vec")
    ni = NetworkSimilarityIndex.load(data_embeddings_dir() / "network_similarity", ei, corpus=corpus)
    combiner = make_combiner("weighted") if hybrid == "weighted" else LearnedCombiner(model="linear")
    model = HybridSimilarityModel(ti, ni, combiner)
    if isinstance(combiner, LearnedCombiner):
        model.fit_learned(corpus, split="validation")
    return model


def _print(ex) -> None:
    print(f"\n{ex.query_id}  vs  {ex.target_id}     method: {ex.method}")
    print("=" * 66)
    print(f"  hybrid   {ex.hybrid_score:.3f}")
    print(f"  text     {ex.text_score:.3f}")
    print(f"  network  {ex.network_score:.3f}   "
          f"(embedding {ex.embedding_score:.2f} / statute {ex.statute_overlap:.2f} "
          f"/ citation {ex.citation_overlap:.2f})")
    print("\n  contribution attribution:")
    for c in ex.contributions:
        v = "  -  " if c.value is None else f"{c.value:.3f}"
        print(f"    {c.component:<16} value={v}  weight={c.weight:+.3f}  "
              f"contribution={c.contribution:+.3f}  ({c.pct_of_score:+.0f}%)")
    if ex.shared_statutes:
        print("\n  shared statutes:")
        for s in ex.shared_statutes:
            label = s.act_name or s.act_id
            if s.kind == "section":
                label += f"  s.{s.section_number}"
                if s.section_title:
                    label += f"  ({s.section_title})"
            tag = "" if s.in_curated_hierarchy else "  [bare Act node]"
            print(f"    ICF {s.icf:5.2f}  {label}{tag}")
    if ex.shared_cited_cases:
        print("\n  shared cited cases (resolved to corpus):")
        for cc in ex.shared_cited_cases:
            print(f"    {cc.doc_id}  {cc.title or ''} ({cc.year})")
    if ex.shared_raw_citations:
        print(f"\n  shared citation strings ({len(ex.shared_raw_citations)}):")
        for r in ex.shared_raw_citations[:12]:
            print(f"    {r}")
    if ex.cocitation_count:
        print(f"\n  co-cited by {ex.cocitation_count} other case(s)")
    print(f"\n  summary: {ex.summary}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--query", help="corpus doc_id of the query case")
    src.add_argument("--file", type=Path, help="PDF/txt of an uploaded query case")
    p.add_argument("--target", required=True, help="corpus doc_id to explain against")
    p.add_argument("--year", type=int, default=None, help="year of the uploaded case")
    p.add_argument("--hybrid", choices=["weighted", "learned"], default="learned")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    corpus = load_corpus(verify=True)
    if args.target not in corpus:
        print(f"[FAIL] target {args.target!r} not in corpus", file=sys.stderr)
        return 1
    explainer = Explainer(corpus, _load_model(corpus, args.hybrid))

    if args.query:
        if args.query not in corpus:
            print(f"[FAIL] query {args.query!r} not in corpus", file=sys.stderr)
            return 1
        ex = explainer.explain_pair(args.query, args.target)
    else:
        from app.ml.text_extraction import extract_text

        ex = explainer.explain_text(extract_text(args.file).raw_text, args.target, year=args.year)

    if args.json:
        print(json.dumps(ex.to_dict(), indent=2, default=str))
    else:
        _print(ex)
    print("\n[ OK ] Step 14 explainability works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
