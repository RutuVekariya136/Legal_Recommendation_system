"""Tests for the database layer (Step 17): models + repository helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from sqlalchemy.orm import Session

from app.database import init_db, make_engine
from app.models import Base, Case, RecommendationRun, UploadedDocument
from app.services import repository as repo


@pytest.fixture
def db(tmp_path) -> Session:
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        yield session


# -- schema -------------------------------------------------------

def test_tables_created(db):
    names = set(Base.metadata.tables)
    assert {"cases", "uploaded_documents", "recommendation_runs",
            "recommendation_results"} <= names


# -- corpus upsert ---------------------------------------------

class _Doc:
    def __init__(self, doc_id, year, title):
        self.doc_id, self.year, self.title = doc_id, year, title
        self.path = f"/x/{doc_id}.txt"
        self.splits = ("validation",)

    def sha1(self):
        return "0" * 40


class _Corpus:
    def __init__(self, docs):
        self._d = {d.doc_id: d for d in docs}

    @property
    def doc_ids(self):
        return sorted(self._d)

    def __getitem__(self, k):
        return self._d[k]

    def text(self, k):
        return "some text " * 10


def test_upsert_cases_is_idempotent(db):
    corpus = _Corpus([_Doc("1990_A_1", 1990, "A v State"),
                      _Doc("1995_B_2", 1995, None)])
    r1 = repo.upsert_cases_from_corpus(db, corpus)
    assert r1 == {"created": 2, "updated": 0, "total": 2}
    r2 = repo.upsert_cases_from_corpus(db, corpus)
    assert r2["created"] == 0 and r2["updated"] == 2 and r2["total"] == 2

    c = repo.get_case(db, "1990_A_1")
    assert c.year == 1990 and c.has_title is True and c.splits == ["validation"]
    assert repo.get_case(db, "1995_B_2").has_title is False


def test_list_cases_filters(db):
    repo.upsert_cases_from_corpus(db, _Corpus([
        _Doc("1990_A_1", 1990, "Tax matter"),
        _Doc("2001_B_2", 2001, "Criminal appeal"),
        _Doc("2010_C_3", 2010, "Tax reassessment"),
    ]))
    rows, total = repo.list_cases(db, q="tax")
    assert total == 2 and {r.doc_id for r in rows} == {"1990_A_1", "2010_C_3"}
    rows, total = repo.list_cases(db, year_min=2005)
    assert [r.doc_id for r in rows] == ["2010_C_3"]


def test_case_stats(db):
    repo.upsert_cases_from_corpus(db, _Corpus([
        _Doc("1990_A_1", 1990, "T"), _Doc("2001_B_2", 2001, None),
    ]))
    s = repo.case_stats(db)
    assert s == {"cases": 2, "with_title": 1, "year_min": 1990, "year_max": 2001}


# -- uploads --------------------------------------------------

def test_create_and_get_upload_with_json_warnings(db):
    up = repo.create_upload(db, filename="case.pdf", source_type="pdf",
                            raw_text="hello world", page_count=3,
                            warnings=["removed 2 header lines"], year_hint=2015)
    assert len(up.id) == 32 and up.char_count == 11
    fetched = repo.get_upload(db, up.id)
    assert fetched.warnings == ["removed 2 header lines"] and fetched.year_hint == 2015
    assert fetched.created_at is not None


# -- recommendation runs ------------------------------------

@dataclass
class _Reco:
    rank: int
    doc_id: str
    hybrid_score: float
    text_score: float
    network_score: float
    rationale: str

    def to_dict(self):
        return {"rank": self.rank, "doc_id": self.doc_id, "hybrid_score": self.hybrid_score}


def test_record_run_stores_ordered_results_and_explanation(db):
    recos = [
        _Reco(2, "y", 0.5, 0.4, 0.3, "second"),
        _Reco(1, "x", 0.9, 0.8, 0.6, "first"),
    ]
    run = repo.record_run(db, query_type="case", method="learned", top_k=2,
                          params={"year_min": 1990}, results=recos,
                          query_case_doc_id="q1")
    assert run.query_case_doc_id == "q1"
    # relationship is ordered by rank
    assert [r.rank for r in run.results] == [1, 2]
    assert run.results[0].target_doc_id == "x"
    assert run.results[0].explanation["hybrid_score"] == 0.9
    assert run.params == {"year_min": 1990}


def test_recent_runs_and_cascade_delete(db):
    for i in range(3):
        repo.record_run(db, query_type="case", method="weighted", top_k=1,
                        params={}, results=[_Reco(1, f"t{i}", 0.5, 0.5, 0.5, "r")],
                        query_case_doc_id=f"q{i}")
    recent = repo.list_recent_runs(db, limit=2)
    assert len(recent) == 2

    from app.models import RecommendationResult

    run = recent[0]
    db.delete(run)
    db.commit()
    assert db.query(RecommendationResult).filter_by(run_id=run.id).count() == 0
