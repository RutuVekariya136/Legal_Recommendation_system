"""Repository-setup smoke tests (Step 2).

These assert only that the skeleton is wired correctly — the backend package
imports, the ML subpackage is reachable without FastAPI, and the real dataset is
in place with the counts documented in the base paper. No pipeline logic yet.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# Counts from PROJECT_NOTES.md §3 / the base paper. Read live from disk below.
EXPECTED = {
    "validation": {"documents": 143, "pairs": 100},
    "test": {"documents": 153, "pairs": 90},
}


def test_backend_package_imports():
    app = importlib.import_module("app")
    assert app.__version__


def test_ml_subpackage_imports_without_fastapi():
    # app.ml must stay decoupled from FastAPI (PROJECT_NOTES.md §4). Checked in a fresh
    # subprocess so an earlier test importing app.main can't pollute the result.
    import subprocess
    import sys

    code = (
        "import sys, importlib;"
        "[importlib.import_module(m) for m in ("
        "'app.ml.text_similarity','app.ml.network_model','app.ml.network_embedding',"
        "'app.ml.network_similarity','app.ml.hybrid_similarity','app.ml.recommendation',"
        "'app.ml.explainability','app.ml.citation_extraction','app.ml.statute_extraction',"
        "'app.ml.preprocessing','app.ml.text_extraction')];"
        "sys.exit(1 if 'fastapi' in sys.modules else 0)"
    )
    env = {"PYTHONPATH": str((REPO_ROOT := __import__('pathlib').Path(__file__).resolve().parents[1]) / 'backend')}
    r = subprocess.run([sys.executable, "-c", code], env={**__import__("os").environ, **env})
    assert r.returncode == 0, "importing app.ml.* pulled in fastapi"


def test_core_dependencies_present():
    for mod in ("numpy", "sklearn", "gensim", "networkx", "rapidfuzz", "pymupdf"):
        importlib.import_module(mod)


def test_numpy_below_2(  ):
    import numpy as np

    assert int(np.__version__.split(".")[0]) < 2, "gensim 4.3.3 requires numpy<2"


@pytest.mark.parametrize("split", list(EXPECTED))
def test_dataset_documents_present(raw_data_dir: Path, split: str):
    docs = list((raw_data_dir / split / "documents").glob("*.txt"))
    assert len(docs) == EXPECTED[split]["documents"]


@pytest.mark.parametrize("split", list(EXPECTED))
def test_dataset_expert_pairs_present(raw_data_dir: Path, split: str):
    csv_path = raw_data_dir / split / "similarity_scores.csv"
    assert csv_path.is_file()
    rows = 0
    for raw in csv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.split(",")[0].strip().lower() in {"doc1", "doc_1", "query", "case1"}:
            continue
        rows += 1
    assert rows == EXPECTED[split]["pairs"]
