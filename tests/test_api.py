"""Tests for the FastAPI app (Step 18), using an injected lightweight service."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.database import get_db, init_db, make_engine
from app.ml.explainability import Explainer
from app.ml.hybrid_similarity import HybridSimilarityModel, WeightedSumCombiner
from app.ml.network_embedding import NetworkEmbeddingIndex
from app.ml.network_similarity import DocFeatures, NetworkSimilarityIndex
from app.ml.preprocessing import preprocess_text
from app.ml.recommendation import RecommendationEngine
from app.ml.text_similarity import TextDoc, TextSimilarityIndex, TfidfEmbedder
from app.main import app
from app.services import repository as repo
from app.services.recommender import RecommenderService, ServiceMeta, get_service


class _Doc:
    def __init__(self, doc_id, year, title, text):
        self.doc_id, self.year, self.title, self._text = doc_id, year, title, text
        self.path = f"/x/{doc_id}.txt"
        self.splits = ("validation",)

    def sha1(self):
        return "a" * 40


class _Corpus:
    def __init__(self, docs):
        self._d = {d.doc_id: d for d in docs}

    def __len__(self):
        return len(self._d)

    @property
    def doc_ids(self):
        return sorted(self._d)

    def __getitem__(self, k):
        return self._d[k]

    def __contains__(self, k):
        return k in self._d

    def text(self, k):
        return self._d[k]._text


DOCS = [
    _Doc("a", 1990, "A v State", "s. 302 of the Indian Penal Code murder accused evidence"),
    _Doc("b", 1995, "B v State", "s. 302 of the Indian Penal Code murder witness sentence"),
    _Doc("c", 2001, "C v Union", "s. 66 of the Indian Income-tax Act, 1922 assessment revenue"),
    _Doc("d", 2011, "D v Union", "s. 66 of the Indian Income-tax Act, 1922 assessment return"),
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    corpus = _Corpus(DOCS)
    te = TfidfEmbedder(min_df=1)
    te.fit([TextDoc(i, preprocess_text(corpus.text(i)).tokens, corpus.text(i))
            for i in corpus.doc_ids])
    ti = TextSimilarityIndex("tfidf", corpus.doc_ids, te.embed_corpus(), embedder=te)
    ei = NetworkEmbeddingIndex("metapath2vec", corpus.doc_ids, np.eye(4, dtype=np.float32))
    store = {
        "a": DocFeatures(act_ids={"ipc_1860"}, section_keys={"ipc_1860::s302"}),
        "b": DocFeatures(act_ids={"ipc_1860"}, section_keys={"ipc_1860::s302"}),
        "c": DocFeatures(act_ids={"income_tax_act_1922"}, section_keys={"income_tax_act_1922::s66"}),
        "d": DocFeatures(act_ids={"income_tax_act_1922"}, section_keys={"income_tax_act_1922::s66"}),
    }
    ni = NetworkSimilarityIndex(ei, store, method="blend")
    hybrid = HybridSimilarityModel(ti, ni, WeightedSumCombiner(0.5, 0.5))
    engine = RecommendationEngine(corpus, hybrid)
    svc = RecommenderService(
        corpus, engine, Explainer(corpus, hybrid, title_index=engine.title_index),
        ServiceMeta("tfidf", "metapath2vec", "weighted", len(corpus), [], 0.0),
    )

    db_engine = make_engine(f"sqlite:///{tmp_path_factory.mktemp('db') / 'api.db'}")
    init_db(db_engine)
    with Session(db_engine) as s:
        repo.upsert_cases_from_corpus(s, corpus)

    def _get_db():
        s = Session(db_engine)
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_service] = lambda: svc
    app.dependency_overrides[get_db] = _get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# -- health / stats ---------------------------------------------

def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_ready(client):
    j = client.get("/api/health/ready").json()
    assert j["ready"] is True and j["n_cases"] == 4


def test_stats(client):
    j = client.get("/api/stats").json()
    assert j["cases"] == 4 and j["model"]["text_method"] == "tfidf"


# -- cases -----------------------------------------------------

def test_list_cases_and_filter(client):
    assert client.get("/api/cases").json()["total"] == 4
    j = client.get("/api/cases", params={"year_min": 2000}).json()
    assert {i["doc_id"] for i in j["items"]} == {"c", "d"}


def test_get_case_detail_and_text(client):
    d = client.get("/api/cases/a").json()
    assert d["title"] == "A v State" and d["preview"].startswith("s. 302")
    assert client.get("/api/cases/a/text").json()["text"] == DOCS[0]._text
    assert client.get("/api/cases/zzz").status_code == 404


# -- recommend ------------------------------------------------

def test_recommend_case_ranks_and_records_run(client):
    j = client.get("/api/recommend/case/a", params={"k": 2}).json()
    assert j["results"][0]["doc_id"] == "b"          # same statute + topic
    assert j["run_id"] and j["method"] == "weighted"
    assert j["results"][0]["shared_sections"][0]["section_key"] == "ipc_1860::s302"
    # the run is persisted
    recent = client.get("/api/recommend/case/a", params={"k": 1, "save": False}).json()
    assert recent["run_id"] is None


def test_recommend_unknown_case_404(client):
    assert client.get("/api/recommend/case/nope").status_code == 404


def test_recommend_respects_year_filter(client):
    j = client.get("/api/recommend/case/c", params={"k": 5, "year_min": 2005}).json()
    assert [r["doc_id"] for r in j["results"]] == ["d"]


# -- compare -------------------------------------------------

def test_compare_returns_explanation(client):
    j = client.get("/api/compare", params={"a": "a", "b": "b"}).json()
    assert j["method"] == "weighted"
    assert any(s["section_number"] == "302" for s in j["shared_statutes"])
    assert sum(c["pct_of_score"] for c in j["contributions"]) == pytest.approx(100, abs=1)


def test_compare_unknown_404(client):
    assert client.get("/api/compare", params={"a": "a", "b": "zzz"}).status_code == 404


# -- graph ---------------------------------------------------

def test_ego_graph(client):
    j = client.get("/api/graph/case/a", params={"similar": 2, "statutes": 3}).json()
    assert j["center"] == "case:a"
    types = {n["type"] for n in j["nodes"]}
    assert "case" in types and ("section" in types or "act" in types)
    # the center case cites IPC 302 -> a stat node and a cites edge
    assert any(e["type"] == "cites" for e in j["edges"])
    assert any(e["type"] == "similar" for e in j["edges"])
    assert client.get("/api/graph/case/zzz").status_code == 404


# -- search -------------------------------------------------

def test_search_semantic_ranks_by_content(client):
    j = client.get("/api/search", params={"q": "murder trial evidence", "mode": "semantic"}).json()
    assert j["mode"] == "semantic"
    assert [i["doc_id"] for i in j["items"][:2]] == ["a", "b"] or \
           {i["doc_id"] for i in j["items"][:2]} == {"a", "b"}
    assert j["items"][0]["snippet"]


def test_search_title_mode_requires_title_hit(client):
    j = client.get("/api/search", params={"q": "Union", "mode": "title"}).json()
    assert {i["doc_id"] for i in j["items"]} == {"c", "d"}
    assert all(i["title_match"] for i in j["items"])
    assert j["total"] == 2


def test_search_year_filter_and_facets(client):
    j = client.get("/api/search", params={"q": "assessment", "mode": "semantic",
                                          "year_min": 2005}).json()
    assert all(i["year"] >= 2005 for i in j["items"])
    assert j["facets"]["total"] == 4
    assert j["facets"]["splits"] == {"validation": 4}


def test_search_facets_endpoint(client):
    j = client.get("/api/search/facets").json()
    assert j["total"] == 4 and "years" in j


def test_search_pagination(client):
    p1 = client.get("/api/search", params={"q": "the", "mode": "hybrid", "limit": 1, "offset": 0}).json()
    p2 = client.get("/api/search", params={"q": "the", "mode": "hybrid", "limit": 1, "offset": 1}).json()
    if p1["total"] >= 2:
        assert p1["items"][0]["doc_id"] != p2["items"][0]["doc_id"]


def test_search_invalid_mode_422(client):
    assert client.get("/api/search", params={"q": "x", "mode": "fuzzy"}).status_code == 422


# -- upload workflow --------------------------------------

_UPLOAD_TEXT = (
    "IN THE SUPREME COURT OF INDIA\n"
    "E. F. Rao v State\n\n"
    "1. This appeal concerns s. 302 of the Indian Penal Code and the appreciation "
    "of circumstantial evidence in a murder trial before the Sessions Court.\n"
    "2. The accused was convicted; the question is whether the chain of evidence is complete.\n"
)


def _pdf_bytes(text: str) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    rect = pymupdf.Rect(40, 40, 555, 800)
    for i in range(0, len(text), 1500):
        doc.new_page().insert_textbox(rect, text[i:i + 1500], fontsize=10)
    out = doc.tobytes()
    doc.close()
    return out


def test_upload_txt_then_recommend_and_compare(client):
    up = client.post("/api/upload",
                     files={"file": ("efrao.txt", _UPLOAD_TEXT.encode(), "text/plain")},
                     data={"year": "2015"})
    assert up.status_code == 201
    j = up.json()
    assert j["source_type"] == "txt" and j["year_hint"] == 2015 and j["char_count"] > 0
    uid = j["id"]

    assert client.get(f"/api/upload/{uid}").json()["id"] == uid
    assert _UPLOAD_TEXT[:20] in client.get(f"/api/upload/{uid}/text").json()["text"]

    rec = client.post(f"/api/recommend/upload/{uid}", params={"k": 2}).json()
    assert rec["query_in_corpus"] is False and rec["run_id"]
    assert rec["results"][0]["doc_id"] in {"a", "b"}  # the two IPC-302 cases

    cmp = client.get(f"/api/compare/upload/{uid}", params={"target": "a"}).json()
    assert cmp["query_in_corpus"] is False
    assert any(s["section_number"] == "302" for s in cmp["shared_statutes"])


def test_upload_pdf(client):
    up = client.post("/api/upload",
                     files={"file": ("efrao.pdf", _pdf_bytes(_UPLOAD_TEXT), "application/pdf")})
    assert up.status_code == 201 and up.json()["source_type"] == "pdf"
    assert up.json()["page_count"] >= 1


def test_upload_rejects_bad_extension(client):
    r = client.post("/api/upload",
                    files={"file": ("x.docx", b"PK\x03\x04stuff", "application/octet-stream")})
    assert r.status_code == 422


def test_upload_rejects_fake_pdf(client):
    r = client.post("/api/upload",
                    files={"file": ("x.pdf", b"not really a pdf", "application/pdf")})
    assert r.status_code == 422


def test_upload_rejects_empty(client):
    r = client.post("/api/upload", files={"file": ("x.txt", b"", "text/plain")})
    assert r.status_code == 422


def test_recommend_upload_unknown_404(client):
    assert client.post("/api/recommend/upload/nope").status_code == 404


def test_compare_upload_unknown_target_404(client):
    up = client.post("/api/upload",
                     files={"file": ("z.txt", _UPLOAD_TEXT.encode(), "text/plain")}).json()
    assert client.get(f"/api/compare/upload/{up['id']}", params={"target": "zzz"}).status_code == 404


# -- misc ----------------------------------------------------

def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/docs"


def test_openapi_schema_available(client):
    j = client.get("/openapi.json").json()
    assert "/api/recommend/case/{doc_id}" in j["paths"]


def test_evaluate_endpoint(client, tmp_path_factory, monkeypatch):
    import json as _json

    from app.routers import evaluate as ev

    report = {"generated_at": "now", "pairs": {"validation": 100, "test": 90},
              "protocol": "x",
              "results": [{"config": "text: tfidf", "kind": "text",
                           "splits": {"validation": {"correlation": {"spearman": 0.76, "pearson": 0.66, "mse": 0.12}},
                                      "test": {"correlation": {"spearman": -0.02, "pearson": -0.01, "mse": 0.19}}}}],
              "diagnosis": {}}
    d = tmp_path_factory.mktemp("proc")
    (d / "evaluation_report.json").write_text(_json.dumps(report))
    monkeypatch.setattr(ev, "data_processed_dir", lambda: d)

    assert client.get("/api/evaluate").json()["pairs"]["test"] == 90
    s = client.get("/api/evaluate/summary").json()
    assert s["rows"][0]["config"] == "text: tfidf"
    assert s["rows"][0]["validation"]["spearman"] == 0.76
