"""Central filesystem path resolution.

Every script / ML module resolves data locations through here so that a larger
corpus can be swapped in later by changing env vars (or backend/.env), never by
editing code. Paths default to the layout in PROJECT_NOTES.md §4 and are resolved
relative to the repo root unless an absolute path is given.
"""

from __future__ import annotations

import os
from pathlib import Path

# paths.py -> utils -> app -> backend -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(env_var: str, default_rel: str) -> Path:
    raw = os.environ.get(env_var, default_rel)
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p)


def data_raw_dir() -> Path:
    return _resolve("DATA_RAW_DIR", "data/raw")


def data_processed_dir() -> Path:
    return _resolve("DATA_PROCESSED_DIR", "data/processed")


def data_embeddings_dir() -> Path:
    return _resolve("DATA_EMBEDDINGS_DIR", "data/embeddings")


def data_graphs_dir() -> Path:
    return _resolve("DATA_GRAPHS_DIR", "data/graphs")


def data_indexes_dir() -> Path:
    return _resolve("DATA_INDEXES_DIR", "data/indexes")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
