"""Tests for app.ml.case_treatment ("is this case still good law?")."""

from __future__ import annotations

from app.ml.case_treatment import (
    TreatmentSignal,
    build_treatment_index,
    classify_signal,
)
from app.ml.citation_extraction import CitationEdge


# -- classify_signal -----------------------------------------------------

def test_classify_signal_negative():
    assert classify_signal("...this view is no longer good law and was overruled...") == "negative"
    assert classify_signal("the earlier decision was set aside by this court.") == "negative"


def test_classify_signal_positive():
    assert classify_signal("the principle laid down was affirmed and followed.") == "positive"


def test_classify_signal_distinguished():
    assert classify_signal("that case is clearly distinguishable on facts.") == "distinguished"


def test_classify_signal_neutral_when_no_keyword():
    assert classify_signal("the court referred to this case in passing.") == "neutral"


def test_classify_signal_negative_takes_priority_over_positive():
    # a snippet that mentions both should never under-report the negative signal
    text = "while earlier followed, it is now overruled and no longer good law."
    assert classify_signal(text) == "negative"


# -- build_treatment_index -----------------------------------------------

def test_build_treatment_index_aggregates_by_cited_doc():
    edges = [
        CitationEdge("new1", "old1", 95.0, context="this is no longer good law."),
        CitationEdge("new2", "old1", 92.0, context="the principle was followed."),
        CitationEdge("new3", "old2", 90.0, context="nothing relevant here."),
    ]
    index = build_treatment_index(edges)
    assert index["old1"].overall_status == "negative"  # negative beats positive
    assert index["old1"].n_resolved_citations_checked == 2
    assert index["old2"].overall_status == "neutral"
    assert "old3" not in index  # no incoming edges -> absent, caller treats as no_data


def test_build_treatment_index_priority_distinguished_over_positive():
    edges = [
        CitationEdge("new1", "old1", 95.0, context="affirmed and followed."),
        CitationEdge("new2", "old1", 92.0, context="distinguishable on facts."),
    ]
    index = build_treatment_index(edges)
    assert index["old1"].overall_status == "distinguished"
    assert all(isinstance(s, TreatmentSignal) for s in index["old1"].signals)
