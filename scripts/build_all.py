"""build_all.py - run the whole artifact pipeline end to end.

    python scripts/build_all.py

Order: preprocess -> citations -> statutes -> graph -> text embeddings ->
network embeddings -> network similarity -> hybrid -> DB init. Idempotent;
safe to re-run. ~2-3 minutes cold (Doc2Vec/SBERT are NOT run here - only the
defaults the API uses: TF-IDF text + metapath2vec network).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

STEPS = [
    ("preprocess",              ["--"]),
    ("extract_citations",       []),
    ("extract_statutes",        []),
    ("build_graph",             []),
    ("generate_embeddings",     ["--method", "tfidf"]),
    ("generate_network_embeddings", ["--method", "metapath2vec"]),
    ("build_network_similarity", ["--method", "blend"]),
    ("build_hybrid",            ["--method", "learned", "--model", "linear"]),
    ("init_db",                 []),
]


def main() -> int:
    import importlib

    total0 = time.time()
    for name, args in STEPS:
        mod = importlib.import_module(name)
        print(f"\n{'=' * 70}\n>>> {name} {' '.join(a for a in args if a != '--')}\n{'=' * 70}")
        t0 = time.time()
        rc = mod.main([a for a in args if a != "--"])
        if rc != 0:
            print(f"[FAIL] {name} exited {rc}", file=sys.stderr)
            return rc
        print(f"    ({time.time() - t0:.1f}s)")
    print(f"\n[ OK ] all artifacts built in {time.time() - total0:.0f}s. "
          f"Start the API:  uvicorn app.main:app --app-dir backend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
