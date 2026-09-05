"""ingest_data.py - Step 3: load the raw corpus, validate it, write a manifest.

This does NOT transform any text (that is Step 4). It:
  1. Loads data/raw/ via app.utils.dataset.load_corpus().
  2. Runs referential-integrity + structural checks.
  3. Prints a human-readable report (counts, year spread, title coverage,
     expert-pair score stats).
  4. Writes data/processed/corpus_manifest.json — one entry per unique document
     plus split/pair summaries — for later steps to consume.

All numbers are computed live from data/raw/. None are hard-coded as results.

Usage:
    python scripts/ingest_data.py               # report + write manifest
    python scripts/ingest_data.py --check       # validate only, no manifest (exit code)
    python scripts/ingest_data.py --out PATH    # custom manifest path
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Make `import app.*` work when run as a plain script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.utils.dataset import (  # noqa: E402
    EXPECTED_DOCUMENTS,
    EXPECTED_PAIRS,
    SPLITS,
    Corpus,
    DatasetError,
    load_corpus,
)
from app.utils.paths import data_processed_dir, ensure_dir  # noqa: E402


def _score_stats(scores: list[float]) -> dict:
    if not scores:
        return {}
    return {
        "count": len(scores),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "mean": round(statistics.fmean(scores), 4),
        "stdev": round(statistics.pstdev(scores), 4),
        "distinct_values": len(set(scores)),
    }


def build_manifest(corpus: Corpus) -> dict:
    documents = []
    for doc_id in corpus.doc_ids:
        doc = corpus[doc_id]
        text = doc.text
        documents.append(
            {
                "doc_id": doc_id,
                "splits": list(doc.splits),
                "year": doc.year,
                "path": str(doc.path.relative_to(_REPO_ROOT).as_posix()),
                "title": doc.title,
                "has_title": doc.has_title,
                "char_len": len(text),
                "line_count": text.count("\n") + 1,
                "sha1": doc.sha1(),
            }
        )

    splits = {}
    for split in SPLITS:
        pairs = corpus.pairs_by_split.get(split, [])
        splits[split] = {
            "documents": len(corpus.split_doc_ids(split)),
            "documents_expected": EXPECTED_DOCUMENTS[split],
            "expert_pairs": len(pairs),
            "expert_pairs_expected": EXPECTED_PAIRS[split],
            "score_stats": _score_stats([p.score for p in pairs]),
        }

    overlap = [d for d in documents if len(d["splits"]) > 1]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(corpus.raw_dir.relative_to(_REPO_ROOT).as_posix()),
        "unique_documents": len(corpus),
        "documents_in_multiple_splits": [d["doc_id"] for d in overlap],
        "documents_without_title": sum(1 for d in documents if not d["has_title"]),
        "splits": splits,
        "documents": documents,
    }


def print_report(corpus: Corpus, manifest: dict) -> None:
    print("=" * 70)
    print("Step 3 - Dataset ingestion report")
    print("=" * 70)

    print(f"\nunique documents: {manifest['unique_documents']}")
    overlap = manifest["documents_in_multiple_splits"]
    print(f"documents in BOTH splits: {len(overlap)}  {overlap or ''}")

    print("\n-- per split --")
    for split, info in manifest["splits"].items():
        d_ok = "OK" if info["documents"] == info["documents_expected"] else "MISMATCH"
        p_ok = "OK" if info["expert_pairs"] == info["expert_pairs_expected"] else "MISMATCH"
        print(
            f"  {split:<11} documents={info['documents']:>3} [{d_ok}]   "
            f"expert_pairs={info['expert_pairs']:>3} [{p_ok}]"
        )
        s = info["score_stats"]
        print(
            f"              scores: min={s['min']} max={s['max']} mean={s['mean']} "
            f"stdev={s['stdev']} distinct={s['distinct_values']}"
        )

    years = [d["year"] for d in manifest["documents"] if d["year"]]
    print("\n-- documents by decade --")
    by_decade = Counter((y // 10) * 10 for y in years)
    for decade in sorted(by_decade):
        bar = "#" * by_decade[decade]
        print(f"  {decade}s  {by_decade[decade]:>3}  {bar}")
    print(f"  year range: {min(years)}-{max(years)}")

    n_no_title = manifest["documents_without_title"]
    total = manifest["unique_documents"]
    print(
        f"\n-- title coverage --\n  {total - n_no_title}/{total} documents have a "
        f"party-style title line; {n_no_title} do not (title=None)."
    )
    print(
        "  (Documented limitation: Step 6 citation matching must handle title=None "
        "docs; they are never dropped.)"
    )

    sizes = [d["char_len"] for d in manifest["documents"]]
    print(
        f"\n-- document length (chars) --\n  min={min(sizes)} "
        f"median={int(statistics.median(sizes))} max={max(sizes)} "
        f"total={sum(sizes):,}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="validate only; do not write the manifest"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="manifest output path (default: data/processed/corpus_manifest.json)",
    )
    args = parser.parse_args(argv)

    try:
        corpus = load_corpus(verify=True)
    except DatasetError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    problems = corpus.check_integrity()
    if problems:
        print("[FAIL] integrity problems:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    manifest = build_manifest(corpus)
    print_report(corpus, manifest)

    if args.check:
        print("\n[ OK ] dataset valid (--check: manifest not written).")
        return 0

    out_path = args.out or (data_processed_dir() / "corpus_manifest.json")
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ OK ] wrote manifest: {out_path.relative_to(_REPO_ROOT).as_posix()}")
    print("       Proceed to Step 4 (Data Preprocessing).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
