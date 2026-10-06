"""import_corpus_export.py - load a corpus export (from scripts/export_corpus.py)
into the local database.

Download the export from the project's GitHub Release, then:
  python scripts/import_corpus_export.py --file legal_corpus.jsonl.gz

Cases whose doc_id already exists are skipped, so the import can be re-run.
After importing, rebuild the search indexes over the full corpus:
  CORPUS_SOURCE=db python scripts/build_all.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "legal_case_similarity.db"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, required=True, help="the .jsonl.gz export")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    existing = {r[0] for r in conn.execute("SELECT doc_id FROM cases")}
    added = skipped = 0
    with gzip.open(args.file, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["doc_id"] in existing:
                skipped += 1
                continue
            text = rec["text"]
            conn.execute(
                "INSERT INTO cases (doc_id, corpus, title, year, splits, has_title, char_len, "
                "source_path, sha1, text, category) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rec["doc_id"], rec["corpus"], rec["title"], rec["year"],
                    json.dumps(rec.get("splits") or []), bool(rec["title"]), len(text), None,
                    hashlib.sha1(text.encode("utf-8")).hexdigest(), text, rec.get("category"),
                ),
            )
            added += 1
    conn.commit()
    conn.close()
    print(f"[ OK ] added {added} cases, skipped {skipped} already present")


if __name__ == "__main__":
    main()
