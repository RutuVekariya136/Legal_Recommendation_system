"""build_network_similarity.py - Step 11: assemble the network-similarity index.

Combines the Hier-SPCNet embedding (Step 10) with a structural feature store
(resolved citations, raw citations, Acts, sections) into one network_score plus
the explanation breakdown Step 14 needs.

    python scripts/build_network_similarity.py --method blend --evaluate

Loads data/embeddings/network_<embmethod>/ (run generate_network_embeddings.py
first). Writes data/embeddings/network_similarity/feature_store.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.ml.network_embedding import NetworkEmbeddingIndex  # noqa: E402
from app.ml.network_similarity import METHODS, NetworkSimilarityIndex  # noqa: E402
from app.utils.dataset import load_corpus  # noqa: E402
from app.utils.paths import data_embeddings_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="blend")
    parser.add_argument("--emb-method", default="metapath2vec",
                        choices=["metapath2vec", "node2vec"])
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    corpus = load_corpus(verify=True)
    emb_dir = data_embeddings_dir() / f"network_{args.emb_method}"
    if not (emb_dir / "index.json").is_file():
        print(f"[FAIL] {emb_dir} missing - run scripts/generate_network_embeddings.py first",
              file=sys.stderr)
        return 1
    embeddings = NetworkEmbeddingIndex.load(emb_dir)

    print(f"embedding: {args.emb_method}   scoring method: {args.method}")
    t0 = time.time()
    nsi = NetworkSimilarityIndex.build(corpus, embeddings, method=args.method)
    print(f"feature store built in {time.time() - t0:.1f}s")

    n_cited = sum(1 for f in nsi.features.values() if f.cited_doc_ids)
    n_acts = sum(len(f.act_ids) for f in nsi.features.values())
    print(f"  docs with >=1 resolved citation: {n_cited}/{len(nsi.features)}")
    print(f"  total (doc, Act) pairs:          {n_acts}")

    sample = corpus.pairs_by_split["validation"][0]
    bd = nsi.breakdown(sample.doc_id_1, sample.doc_id_2)
    print(f"\nexample breakdown  {bd.doc_id_1} vs {bd.doc_id_2}  (expert={sample.score}):")
    print(f"  network_score={bd.network_score}  embedding={bd.embedding_score}  "
          f"statute_overlap={bd.statute_overlap}  citation_overlap={bd.citation_overlap}")
    print(f"  shared Acts: {[a['act_id'] for a in bd.shared_acts]}")
    print(f"  shared sections: {[s['section_key'] for s in bd.shared_sections][:6]}")

    if args.evaluate:
        from scipy.stats import pearsonr, spearmanr

        print("\n-- NETWORK sanity correlation (NOT the hybrid eval; Step 22) --")
        for m in ("embedding", "blend"):
            nsi.method = m
            for split in ("validation", "test"):
                ps = corpus.pairs_by_split[split]
                sims = [nsi.similarity(p.doc_id_1, p.doc_id_2) for p in ps]
                gold = [p.score for p in ps]
                print(f"   {m:<9} {split:<11} Pearson {pearsonr(sims, gold)[0]:+.3f}  "
                      f"Spearman {spearmanr(sims, gold)[0]:+.3f}")
        nsi.method = args.method

    if not args.no_save:
        out = nsi.save()
        print(f"\n[ OK ] saved -> {out.relative_to(_REPO_ROOT).as_posix()}/feature_store.json")
    print("       Proceed to Step 12 (Hybrid Similarity).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
