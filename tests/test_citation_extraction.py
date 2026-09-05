"""Tests for app.ml.citation_extraction (Step 6)."""

from __future__ import annotations

import pytest

from app.ml.citation_extraction import (
    CitationEdge,
    CorpusTitleIndex,
    RawCaseCitation,
    build_citation_edges,
    extract_case_citations,
    normalize_citation,
    normalize_party,
)


# -- extraction ---------------------------------------------------------

def test_extract_simple_party_citation():
    cs = extract_case_citations("The rule in Croft v. Dunphy was applied by the Board.")
    assert len(cs) == 1
    assert cs[0].party_a == "Croft" and cs[0].party_b == "Dunphy"


def test_extract_strips_leadin_words_and_judge_names():
    cs = extract_case_citations("As Lord Cranworth observed in Whicker v. Hume, domicile matters.")
    assert cs and cs[0].party_a == "Whicker" and cs[0].party_b == "Hume"


def test_extract_does_not_cross_sentence_boundary():
    cs = extract_case_citations("This was decided in Madras. See Badische Anilin v. Hickson here.")
    assert cs and cs[0].party_a.startswith("Badische")


def test_extract_captures_reporter_year():
    cs = extract_case_citations("relied on Qasim Razvi v. The State of Hyderabad [1953] S.C.R. 589 for this")
    assert cs and cs[0].reporter_year == 1953


def test_extract_rejects_generic_only_parties():
    # "State v. It" / bare pronouns are not usable mentions
    assert extract_case_citations("the State v. it was never argued") == []


# -- normalisation ---------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected_contains",
    [
        ("The State of Travancore-Cochin & Others", "travancore"),
        ("D. P. Joshi", "joshi"),
        ("M/s. Bombay Salt and Chemical Industries", "bombay salt"),
    ],
)
def test_normalize_party_drops_noise(raw, expected_contains):
    assert expected_contains in normalize_party(raw)


def test_normalize_citation_shape():
    assert normalize_citation("Dalip Singh", "State of Punjab") == "dalip singh v state punjab"


# -- resolution against a synthetic index ---------------------------

class _FakeDoc:
    def __init__(self, doc_id, title, year):
        self.doc_id, self.title, self.year = doc_id, title, year


class _FakeCorpus:
    def __init__(self, docs):
        self._d = {d.doc_id: d for d in docs}

    @property
    def doc_ids(self):
        return sorted(self._d)

    def __getitem__(self, k):
        return self._d[k]

    def text(self, k):
        return self._d[k]._text  # set in tests that need it


@pytest.fixture
def index():
    docs = [
        _FakeDoc("1960_D_4", "Dalip Singh v State of Punjab", 1960),
        _FakeDoc("1955_S_34", "D. P. Joshi v State of Madhya Bharat and Another", 1955),
        _FakeDoc("2005_S_628", "Jacob Mathew v State of Punjab and another", 2005),
        _FakeDoc("1953_24", "Supreme Court of India", 1953),  # no party title -> skipped
    ]
    return CorpusTitleIndex.from_corpus(_FakeCorpus(docs))


def test_index_skips_titleless_documents(index):
    assert "1953_24" not in index.normalized
    assert len(index) == 3


def test_resolve_exact(index):
    cit = RawCaseCitation(raw="Dalip Singh v State of Punjab", party_a="Dalip Singh",
                          party_b="State of Punjab")
    r = index.resolve(cit, min_score=90)
    assert r is not None and r.to_doc_id == "1960_D_4"


def test_resolve_fuzzy_variant(index):
    cit = RawCaseCitation(raw="Joshi v State of Madhya Bharat", party_a="Joshi",
                          party_b="State of Madhya Bharat")
    r = index.resolve(cit, min_score=88)
    assert r is not None and r.to_doc_id == "1955_S_34"


def test_resolve_respects_temporal_guard(index):
    cit = RawCaseCitation(raw="Jacob Mathew v State of Punjab", party_a="Jacob Mathew",
                          party_b="State of Punjab")
    assert index.resolve(cit, min_score=88, not_after_year=2000) is None
    assert index.resolve(cit, min_score=88, not_after_year=2010) is not None


def test_resolve_excludes_self(index):
    cit = RawCaseCitation(raw="Dalip Singh v State of Punjab", party_a="Dalip Singh",
                          party_b="State of Punjab")
    assert index.resolve(cit, exclude_doc_id="1960_D_4", min_score=90) is None


def test_resolve_rejects_debris_query(index):
    # "ltd v workmen" is a subset of many real titles but not a usable citation
    cit = RawCaseCitation(raw="Ltd v Their Workmen", party_a="Ltd", party_b="Their Workmen")
    assert index.resolve(cit, min_score=85) is None


# -- end to end on the real corpus --------------------------------

@pytest.fixture(scope="module")
def real_corpus():
    from app.utils.dataset import load_corpus

    return load_corpus()


def test_build_edges_on_real_corpus(real_corpus):
    edges = build_citation_edges(real_corpus, min_score=90)
    assert edges, "expected at least some resolved case->case edges"
    assert all(isinstance(e, CitationEdge) for e in edges)
    # honest expectation: sparse, not everything resolves
    assert len(edges) < 400
    # structural guarantees
    assert all(e.src_doc_id != e.dst_doc_id for e in edges)
    assert all(e.src_doc_id in real_corpus and e.dst_doc_id in real_corpus for e in edges)
    # temporal sanity: never cite a later case
    for e in edges:
        sy, dy = real_corpus[e.src_doc_id].year, real_corpus[e.dst_doc_id].year
        if sy and dy:
            assert dy <= sy
