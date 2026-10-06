"""export_corpus.py - write every case in the database to one gzipped JSONL file.

Each line is one judgment: doc_id, corpus, title, year, category, splits, text.
The export is what gets published as a GitHub Release asset, so a fresh clone
can rebuild the full corpus with scripts/import_corpus_export.py.

Usage:
  python scripts/export_corpus.py [--out data/export/legal_corpus.jsonl.gz]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "legal_case_similarity.db"
DEFAULT_OUT = REPO_ROOT / "data" / "export" / "legal_corpus.jsonl.gz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # read-only connection: the export never modifies the database
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT doc_id, corpus, title, year, category, splits, text FROM cases ORDER BY id"
    )
    n = 0
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for doc_id, corpus, title, year, category, splits, text in rows:
            record = {
                "doc_id": doc_id,
                "corpus": corpus,
                "title": title,
                "year": year,
                "category": category,
                "splits": json.loads(splits) if splits else [],
                "text": text or "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    conn.close()
    print(f"[ OK ] wrote {n} cases to {args.out}")


if __name__ == "__main__":
    main()
