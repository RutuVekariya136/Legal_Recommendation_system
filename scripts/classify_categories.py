"""Backfill Case.category for every row using the keyword classifier in
app.ml.case_category. Idempotent - safe to re-run after editing categories."""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.ml.case_category import CATEGORIES, classify  # noqa: E402

DB_PATH = _REPO_ROOT / "legal_case_similarity.db"


def main() -> int:
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()
    cur.execute("SELECT id, text FROM cases")
    rows = cur.fetchall()
    print(f"classifying {len(rows)} cases...")

    t0 = time.time()
    counts: dict[str, int] = {}
    updates = []
    for i, (case_id, text) in enumerate(rows):
        cat = classify(text or "")
        counts[cat or "uncategorised"] = counts.get(cat or "uncategorised", 0) + 1
        updates.append((cat, case_id))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)")

    cur.executemany("UPDATE cases SET category = ? WHERE id = ?", updates)
    con.commit()
    con.close()

    print(f"\ndone in {time.time() - t0:.0f}s")
    for cat_id, _, _ in CATEGORIES:
        print(f"  {cat_id:22s} {counts.get(cat_id, 0)}")
    print(f"  {'uncategorised':22s} {counts.get('uncategorised', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
