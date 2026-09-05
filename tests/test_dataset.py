"""Tests for the dataset loader (Step 3) against the real data/raw/ corpus."""

from __future__ import annotations

import pytest

from app.utils.dataset import (
    SPLITS,
    extract_title,
    load_corpus,
    load_expert_pairs,
    parse_year,
)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(verify=True)


def test_parse_year():
    assert parse_year("1971_138") == 1971
    assert parse_year("1955_S_34") == 1955
    assert parse_year("not_a_case") is None


def test_extract_title_party_line():
    assert extract_title("D. P. Joshi v State of Madhya Bharat\nSupreme Court of India") == (
        "D. P. Joshi v State of Madhya Bharat"
    )


def test_extract_title_none_when_court_header_first():
    assert extract_title("Supreme Court of India\n\n30 March 1953\n") is None


def test_corpus_has_295_unique_documents(corpus):
    # 143 + 153 with exactly one byte-identical overlap (2011_I_16).
    assert len(corpus) == 295


def test_overlap_document_belongs_to_both_splits(corpus):
    assert "2011_I_16" in corpus
    assert set(corpus["2011_I_16"].splits) == {"validation", "test"}


@pytest.mark.parametrize("split,n_docs", [("validation", 143), ("test", 153)])
def test_per_split_document_counts(corpus, split, n_docs):
    assert len(corpus.split_doc_ids(split)) == n_docs


@pytest.mark.parametrize("split,n_pairs", [("validation", 100), ("test", 90)])
def test_expert_pair_counts(split, n_pairs):
    assert len(load_expert_pairs(split)) == n_pairs


def test_expert_pairs_have_full_referential_integrity(corpus):
    for split in SPLITS:
        for p in corpus.pairs_by_split[split]:
            assert p.doc_id_1 in corpus and p.doc_id_2 in corpus
            assert split in corpus[p.doc_id_1].splits
            assert split in corpus[p.doc_id_2].splits
            assert 0.0 <= p.score <= 1.0


def test_integrity_check_is_clean(corpus):
    assert corpus.check_integrity() == []


def test_title_coverage_is_reported_not_hidden(corpus):
    # ~60 documents legitimately have no party-style title; just assert the
    # loader exposes this rather than silently substituting something.
    missing = [d for d in corpus.documents.values() if not d.has_title]
    assert 0 < len(missing) < len(corpus)
