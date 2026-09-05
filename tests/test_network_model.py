"""Tests for app.ml.network_model (Step 9)."""

from __future__ import annotations

import networkx as nx
import pytest

from app.ml.citation_extraction import CitationEdge
from app.ml.network_model import (
    build_hier_spcnet,
    build_pcnet,
    case_node,
    compute_icf,
    from_node_link,
    graph_summary,
    to_node_link,
)
from app.ml.statute_extraction import load_hierarchy


class _Doc:
    def __init__(self, doc_id, title, year, text):
        self.doc_id, self.title, self.year, self._text = doc_id, title, year, text


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


@pytest.fixture
def corpus():
    return _Corpus([
        _Doc("1960_A_1", "A v State", 1960, "Conviction under s. 302 of the Indian Penal Code and s. 34 of the Indian Penal Code."),
        _Doc("1975_B_2", "B v State", 1975, "Appeal on s. 302 of the Indian Penal Code; also article 21 of the Constitution."),
        _Doc("1980_C_3", "C v Union", 1980, "Assessment under s. 66 of the Indian Income-tax Act, 1922."),
    ])


# -- PCNet ----------------------------------------------------------

def test_pcnet_has_all_cases_and_given_edges(corpus):
    edges = [CitationEdge("1975_B_2", "1960_A_1", 95.0, mentions=2, examples=["A v State"])]
    g = build_pcnet(corpus, citation_edges=edges)
    assert g.number_of_nodes() == 3
    assert all(d["ntype"] == "case" for _, d in g.nodes(data=True))
    assert g.has_edge(case_node("1975_B_2"), case_node("1960_A_1"))
    assert g[case_node("1975_B_2")][case_node("1960_A_1")]["etype"] == "cites_case"
    assert g[case_node("1975_B_2")][case_node("1960_A_1")]["weight"] == pytest.approx(0.95)


# -- ICF ----------------------------------------------------------

def test_compute_icf_rarer_target_scores_higher():
    c2t = {"a": {"common", "rare"}, "b": {"common"}, "c": {"common"}}
    icf = compute_icf(c2t, n_cases=3)
    assert icf["rare"] > icf["common"]


# -- Hier-SPCNet ------------------------------------------------

def test_hier_spcnet_adds_statute_layer(corpus):
    g = build_hier_spcnet(corpus, citation_edges=[], hierarchy=load_hierarchy())
    types = {d["ntype"] for _, d in g.nodes(data=True)}
    assert {"case", "act", "section", "chapter"} <= types

    # case -> section edge exists and IPC 302 links up to a chapter then the Act
    assert g.has_edge(case_node("1960_A_1"), "sec:ipc_1860::s302")
    assert g.has_edge("sec:ipc_1860::s302", "chap:ipc_1860::chapter::XVI")
    assert g.has_edge("chap:ipc_1860::chapter::XVI", "act:ipc_1860")
    assert g.nodes["act:ipc_1860"]["in_sample_hierarchy"] is True


def test_hier_spcnet_connects_cases_that_pcnet_leaves_isolated(corpus):
    pc = build_pcnet(corpus, citation_edges=[])
    hs = build_hier_spcnet(corpus, citation_edges=[], hierarchy=load_hierarchy())
    assert nx.number_weakly_connected_components(pc) == 3          # no case-case edges
    # 1960_A_1 and 1975_B_2 both cite IPC 302 -> now in one component
    assert nx.number_weakly_connected_components(hs) < 3


def test_statute_nodes_carry_icf_weight_on_edges(corpus):
    g = build_hier_spcnet(corpus, citation_edges=[], hierarchy=load_hierarchy(), with_icf=True)
    w = g[case_node("1980_C_3")]["sec:income_tax_act_1922::s66"]["weight"]
    assert w > 1.0  # section cited by only one of three cases -> ICF boosted


def test_non_curated_act_still_gets_a_node():
    corpus = _Corpus([_Doc("1990_X_1", "X v Y", 1990,
                           "Considered under s. 9 of the Foobar Regulation Act, 1975.")])
    g = build_hier_spcnet(corpus, citation_edges=[], hierarchy=load_hierarchy())
    node = "act:foobar_regulation_act_1975"
    assert node in g and g.nodes[node]["in_sample_hierarchy"] is False


# -- persistence ----------------------------------------------

def test_node_link_roundtrip(corpus):
    g = build_hier_spcnet(corpus, citation_edges=[], hierarchy=load_hierarchy())
    g2 = from_node_link(to_node_link(g))
    assert g2.number_of_nodes() == g.number_of_nodes()
    assert g2.number_of_edges() == g.number_of_edges()
    assert g2.nodes["act:ipc_1860"]["ntype"] == "act"


# -- real corpus --------------------------------------------

def test_build_on_real_corpus():
    from app.utils.dataset import load_corpus

    corpus = load_corpus()
    hs = build_hier_spcnet(corpus, min_score=90)
    s = graph_summary(hs)
    assert s["nodes_by_type"]["case"] == 295
    assert s["nodes_by_type"]["act"] > 100
    assert s["nodes_by_type"]["section"] > 500
    # the statute layer must connect far more than PCNet's largest component (4)
    assert s["largest_component_size"] > 1000
    assert s["isolated_cases"] < 260
