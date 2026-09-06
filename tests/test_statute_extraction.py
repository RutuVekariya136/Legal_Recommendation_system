"""Tests for app.ml.statute_extraction (Step 7)."""

from __future__ import annotations

import pytest

from app.ml.statute_extraction import (
    extract_statute_references,
    find_primary_act_definition,
    load_hierarchy,
    resolve_document_statutes,
)


@pytest.fixture(scope="module")
def h():
    return load_hierarchy()


# -- Act resolution ---------------------------------------------------

def test_resolve_curated_alias(h):
    m = h.resolve_act("Indian Penal Code")
    assert m and m.act_id == "ipc_1860" and m.in_sample_hierarchy


def test_resolve_year_disambiguation(h):
    assert h.resolve_act("Income-tax Act", year=1922).act_id == "income_tax_act_1922"
    assert h.resolve_act("Income-tax Act", year=1961).act_id == "income_tax_act_1961"


def test_resolve_abbreviation(h):
    assert h.resolve_act("Cr.P.C").act_id == "crpc_1973"
    assert h.resolve_act("I.P.C").act_id == "ipc_1860"


def test_non_curated_act_gets_bare_node_never_dropped(h):
    m = h.resolve_act("Widget Regulation Act", year=1975)
    assert m is not None
    assert m.act_id == "widget_regulation_act_1975"
    assert m.in_sample_hierarchy is False


def test_curated_sample_covers_common_criminal_and_commercial_acts(h):
    assert len(h.curated_act_ids) >= 40
    for aid in ("ndps_act_1985", "pmla_2002", "pocso_act_2012", "ibc_2016",
                "dv_act_2005", "arbitration_conciliation_act_1996"):
        assert h.has_act(aid) and h.act_canonical(aid)
    assert h.resolve_act("NDPS Act").act_id == "ndps_act_1985"
    assert h.section_title("pocso_act_2012", "4")


def test_external_hierarchy_file_merges_over_the_sample(tmp_path, monkeypatch):
    import json

    from app.ml import statute_extraction as se

    ext = tmp_path / "extra.json"
    ext.write_text(json.dumps({
        "aliases": {"my special act": "my_act_2020"},
        "acts": {"my_act_2020": {"canonical": "My Special Act, 2020", "year": 2020,
                                 "chapters": {"I": {"title": "Preliminary",
                                                    "sections": {"5": "Key provision"}}}}},
    }), encoding="utf-8")
    monkeypatch.setenv("STATUTE_HIERARCHY_PATH", str(ext))
    se.load_hierarchy.cache_clear()
    try:
        h2 = se.load_hierarchy()
        assert h2.has_act("my_act_2020") and h2.has_act("ipc_1860")  # merged, not replaced
        assert h2.resolve_act("My Special Act").act_id == "my_act_2020"
        assert h2.section_title("my_act_2020", "5") == "Key provision"
    finally:
        se.load_hierarchy.cache_clear()


def test_anaphoric_surface_returns_none(h):
    assert h.resolve_act("the Act") is None
    assert h.resolve_act("said Code") is None


# -- reference extraction ------------------------------------------

def test_extract_section_with_full_act_and_number(h):
    (ref,) = extract_statute_references(
        "Section 66(1) of the Indian Income-tax Act (XI of 1922) applies.", hierarchy=h
    )
    assert ref.kind == "section" and ref.number == "66" and ref.subsections == ["1"]
    assert ref.act_id == "income_tax_act_1922" and ref.act_year == 1922
    assert ref.in_sample_hierarchy


def test_extract_article_defaults_to_constitution(h):
    refs = extract_statute_references("A petition under article 226 was filed.", hierarchy=h)
    assert refs and refs[0].kind == "article" and refs[0].number == "226"
    assert refs[0].act_id == "constitution_india"


def test_extract_anaphoric_the_act(h):
    (ref,) = extract_statute_references("Reference to s. 45 of the Act here.", hierarchy=h)
    assert ref.anaphoric is True and ref.act_id is None


def test_extract_multiple_numbers_expand_to_separate_refs(h):
    refs = extract_statute_references("under ss. 5 and 10 of the Act", hierarchy=h)
    assert sorted(r.number for r in refs) == ["10", "5"]


def test_extract_government_of_india_act(h):
    (ref,) = extract_statute_references(
        "s. 298(1) of the Government of India Act, 1935", hierarchy=h
    )
    assert ref.act_id == "goi_act_1935" and ref.number == "298"


# -- document-level resolution --------------------------------------

def test_primary_act_from_hereinafter_definition(h):
    text = (
        "The State enacted the Bombay Sales Tax Act, 1952, (hereinafter referred to "
        "as 'the Act'). Under s. 5 of the Act and s. 12 of the Act the levy was made."
    )
    pm = find_primary_act_definition(text, hierarchy=h)
    assert pm is not None and pm.act_id == "bombay_sales_tax_act_1952"

    ds = resolve_document_statutes(text, hierarchy=h)
    assert ds.primary_act_id == "bombay_sales_tax_act_1952"
    # both anaphoric "s. N of the Act" now point at the primary Act
    assert ds.n_anaphoric_resolved >= 2
    assert all(r.act_id == "bombay_sales_tax_act_1952" for r in ds.references if r.anaphoric)


def test_primary_act_prefers_substantive_over_constitution(h):
    text = (
        "The challenge under article 14 and article 19 of the Constitution concerns "
        "s. 302 of the Indian Penal Code and s. 304 of the Indian Penal Code."
    )
    ds = resolve_document_statutes(text, hierarchy=h)
    assert ds.primary_act_id == "ipc_1860"


def test_hierarchy_path_for_curated_section(h):
    ds = resolve_document_statutes("s. 302 of the Indian Penal Code", hierarchy=h)
    paths = ds.hierarchy_paths(h)
    key = "ipc_1860::s302"
    assert key in paths
    levels = [lvl for lvl, _, _ in paths[key]]
    assert levels == ["act", "chapter"]


# -- real corpus ----------------------------------------------------

def test_statutes_on_real_corpus():
    from app.utils.dataset import load_corpus

    corpus = load_corpus()
    hier = load_hierarchy()
    docs_with = 0
    total = 0
    for doc_id in corpus.doc_ids:
        ds = resolve_document_statutes(corpus.text(doc_id), hierarchy=hier)
        total += len(ds.references)
        if ds.act_ids:
            docs_with += 1
    assert total > 5000  # statutes are dense in these judgments
    assert docs_with > 200  # most docs cite at least one Act
