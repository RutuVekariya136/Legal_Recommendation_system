"""init_db.py - Step 17: create the SQLite schema and load the corpus.

    python scripts/init_db.py            # create tables + upsert the 295 cases
    python scripts/init_db.py --drop     # drop everything first (destructive)

The DB file location comes from backend/.env (DATABASE_URL), default
./legal_case_similarity.db at the repo root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.database import engine, init_db  # noqa: E402
from app.models import Base  # noqa: E402
from app.services.repository import case_stats  # noqa: E402
from app.services.repository import upsert_cases_from_corpus  # noqa: E402
from app.database import session_scope  # noqa: E402
from app.utils.dataset import load_corpus  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop", action="store_true", help="drop all tables first")
    args = parser.parse_args(argv)

    settings = get_settings()
    print(f"database: {settings.database_url_resolved}")

    if args.drop:
        print("dropping all tables ...")
        Base.metadata.drop_all(engine)

    init_db()
    print(f"tables: {', '.join(sorted(Base.metadata.tables))}")

    corpus = load_corpus(verify=True)
    with session_scope() as db:
        result = upsert_cases_from_corpus(db, corpus)
        stats = case_stats(db)

    print(f"cases upserted: +{result['created']} new / {result['updated']} updated "
          f"/ {result['total']} total")
    print(f"stats: {stats}")
    print("\n[ OK ] database ready. Proceed to Step 18 (FastAPI Backend).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
