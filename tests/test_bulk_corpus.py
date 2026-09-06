"""Tests for the scale-mode DB-backed corpus (Phase 2, P1)."""

from __future__ import annotations

import json

import pytest
from sqlalchemy.orm import Session

from app.database import bind, init_db, make_engine
from app.services import repository as repo
from app.utils.dataset import CaseDocument, _load_corpus_from_db, load_corpus


@pytest.fixture
def db_engine(tmp_path):
    import app.database as database

    original = database.engine
    engine = make_engine(f"sqlite:///{tmp_path / 'bulk.db'}")
    init_db(engine)
    bind(engine)  # point SessionLocal at it so load_corpus(source="db") works
    try:
        yield engine
    finally:
        bind(original)


# -- CaseDocument in-memory mode ---------------------------------

def test_case_document_from_text_value():
    d = CaseDocument("x_1", path=None, splits=("bulk",), year=2001,
                     text_value="A v State. Held: appeal allowed.", title_value="A v State")
    assert d.text.startswith("A v State")
    assert d.title == "A v State"
    assert len(d.sha1()) == 40


def test_case_document_empty_title_value_is_none():
    d = CaseDocument("x", path=None, splits=(), year=None, text_value="text", title_value="")
    assert d.title is None


# -- bulk insert ------------------------------------------------

def test_bulk_insert_dedups_on_id_and_sha(db_engine):
    with Session(db_engine) as db:
        rows = [
            {"doc_id": "a", "text": "first document text", "year": 2000},
            {"doc_id": "a", "text": "different text same id", "year": 2000},   # dup id
            {"doc_id": "b", "text": "first document text"},                    # dup sha
            {"doc_id": "c", "text": "genuinely new content here", "title": "C v D"},
            {"doc_id": "d", "text": ""},                                       # empty
        ]
        r = repo.bulk_insert_documents(db, rows, corpus_name="ildc")
        assert r["inserted"] == 2 and r["skipped"] == 3
        assert r["by_corpus"] == {"ildc": 2}


# -- load_corpus(source="db") ---------------------------------

@pytest.mark.slow
def test_load_corpus_from_db_merges_base_and_bulk(db_engine):
    base = load_corpus(source="files")
    with Session(db_engine) as db:
        repo.upsert_cases_from_corpus(db, base, corpus_name="base")
        repo.bulk_insert_documents(
            db,
            [{"doc_id": "ildc_1", "text": "s. 302 of the Indian Penal Code murder appeal", "year": 2015}],
            corpus_name="ildc",
        )

    corpus = load_corpus(source="db", verify=True)  # expert-pair integrity must hold
    assert "ildc_1" in corpus and "1955_S_34" in corpus
    assert len(corpus) == len(base) + 1
    assert corpus.text("ildc_1").startswith("s. 302")
    assert len(corpus.pairs_by_split["validation"]) == 100
    assert corpus["1955_S_34"].title == base["1955_S_34"].title


def test_load_corpus_from_db_empty_raises(db_engine):
    with pytest.raises(Exception):
        _load_corpus_from_db(verify=False)


# -- jsonl ingestion (script-level) -------------------------

def test_ingest_jsonl(tmp_path):
    p = tmp_path / "docs.jsonl"
    p.write_text("\n".join(json.dumps(o) for o in [
        {"id": "j1", "text": "First judgment about tax assessment.", "year": 1999},
        {"id": "j2", "text": "Second judgment about criminal appeal.", "year": 2004},
    ]), encoding="utf-8")

    import subprocess
    import sys

    import os

    from app.utils.paths import REPO_ROOT

    r = subprocess.run(
        [sys.executable, "scripts/ingest_bulk.py", "--source", "jsonl",
         "--path", str(p), "--corpus", "testset"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "backend"),
             "DATABASE_URL": f"sqlite:///{tmp_path / 'jsonl.db'}"},
    )
    assert r.returncode == 0, r.stderr
    assert "inserted 2" in r.stdout
