"""extract_citations.py - Step 6: extract & resolve case->case citations.

For every document it:
  1. regex-extracts "A v B" case mentions (+ any trailing reporter year),
  2. fuzzy-resolves each against the corpus cause-titles (rapidfuzz), with a
     temporal guard (a case cannot cite a later case) and an optional
     reporter-year window.

Writes per-doc JSON to data/processed/citations/<doc_id>.json and a summary to
data/processed/citation_report.json. These feed PCNet construction in Step 9.

The resolution rate is REPORTED honestly - most mentions stay unresolved because
many cited cases are English/foreign or lower-court decisions absent from our
~295-doc demo corpus (CLAUDE.md §3).

Usage:
    python scripts/extract_citations.py
    python scripts/extract_citations.py --min-score 92
    python scripts/extract_citations.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.ml.citation_extraction import (  # noqa: E402
    DEFAULT_MIN_SCORE,
    CorpusTitleIndex,
    extract_case_citations,
)
from app.utils.dataset import load_corpus  # noqa: E402
from app.utils.paths import data_processed_dir, ensure_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    corpus = load_corpus(verify=True)
    index = CorpusTitleIndex.from_corpus(corpus)
    out_dir = args.out_dir or data_processed_dir()
    cite_dir = ensure_dir(out_dir / "citations")

    doc_ids = corpus.doc_ids[: args.limit] if args.limit else corpus.doc_ids
    print(f"resolving citations for {len(doc_ids)} docs "
          f"(title index: {len(index)}/{len(corpus)} docs, min_score={args.min_score})")

    total_mentions = 0
    total_unique = 0
    total_resolved_unique = 0
    per_doc = []
    in_degree: Counter[str] = Counter()
    out_degree: dict[str, int] = {}
    resolved_examples: list[str] = []
    unresolved_examples: list[str] = []

    for i, src in enumerate(doc_ids, 1):
        src_year = corpus[src].year
        mentions = extract_case_citations(corpus.text(src))
        uniq: dict[str, dict] = {}
        for m in mentions:
            slot = uniq.setdefault(m.raw, {"raw": m.raw, "reporter_year": m.reporter_year, "count": 0})
            slot["count"] += 1

        resolved = []
        for raw, slot in uniq.items():
            from app.ml.citation_extraction import RawCaseCitation

            cit = RawCaseCitation(
                raw=raw, party_a=raw.split(" v ", 1)[0], party_b=raw.split(" v ", 1)[-1],
                reporter_year=slot["reporter_year"],
            )
            r = index.resolve(cit, exclude_doc_id=src, min_score=args.min_score,
                              not_after_year=src_year)
            if r is not None:
                resolved.append({"to": r.to_doc_id, "to_title": r.to_title,
                                 "score": round(r.score, 1), "raw": raw})
                in_degree[r.to_doc_id] += 1
                if len(resolved_examples) < 25:
                    resolved_examples.append(f"{src}: {raw!r} -> {r.to_doc_id} {r.to_title!r} @{r.score:.0f}")
            elif len(unresolved_examples) < 25:
                unresolved_examples.append(f"{src}: {raw!r}")

        total_mentions += len(mentions)
        total_unique += len(uniq)
        total_resolved_unique += len(resolved)
        out_degree[src] = len(resolved)

        (cite_dir / f"{src}.json").write_text(
            json.dumps(
                {
                    "doc_id": src,
                    "title": corpus[src].title,
                    "year": src_year,
                    "n_mentions": len(mentions),
                    "n_unique_mentions": len(uniq),
                    "n_resolved": len(resolved),
                    "mentions": list(uniq.values()),
                    "resolved": resolved,
                },
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        per_doc.append({"doc_id": src, "n_mentions": len(mentions),
                        "n_unique_mentions": len(uniq), "n_resolved": len(resolved)})
        if i % 50 == 0 or i == len(doc_ids):
            print(f"  {i}/{len(doc_ids)}")

    n_edges = sum(out_degree.values())
    rate = (total_resolved_unique / total_unique * 100) if total_unique else 0.0
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_score": args.min_score,
        "title_index_size": len(index),
        "n_documents": len(per_doc),
        "total_mentions": total_mentions,
        "total_unique_mentions": total_unique,
        "total_resolved_edges": n_edges,
        "unique_mentions_resolved": total_resolved_unique,
        "resolution_rate_pct": round(rate, 1),
        "docs_with_out_edge": sum(1 for v in out_degree.values() if v),
        "distinct_cited_docs": len(in_degree),
        "mean_out_degree": round(statistics.fmean(out_degree.values()), 2) if out_degree else 0,
        "top_cited": in_degree.most_common(15),
        "resolved_examples": resolved_examples,
        "unresolved_examples": unresolved_examples,
        "documents": per_doc,
    }
    (out_dir / "citation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n-- summary --")
    print(f"  mentions:              {total_mentions} ({total_unique} unique per-doc)")
    print(f"  resolved case->case edges: {n_edges}")
    print(f"  resolution rate:       {rate:.1f}%  (low is expected - many cited cases are")
    print(f"                         foreign / lower-court, not in a 295-doc corpus)")
    print(f"  docs citing >=1 corpus case: {report['docs_with_out_edge']}/{len(per_doc)}")
    print(f"  distinct cited corpus docs:  {len(in_degree)}")
    if in_degree:
        print("  most-cited within corpus:")
        for did, cnt in in_degree.most_common(5):
            print(f"    {did}  x{cnt}  {corpus[did].title!r}")
    print(f"\n[ OK ] wrote citations/ ({len(per_doc)} files) and citation_report.json")
    print("       Proceed to Step 7 (Statute Extraction).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
