"""Network construction: PCNet and Hier-SPCNet (pipeline stage: Hier-SPCNet).

    Citation Extraction + Statute Extraction -> [Hier-SPCNet] -> Network Embedding -> ...

Two graphs, following Bhattacharya et al. (2022):

  * **PCNet** (Precedent Citation Network) - cases only. Directed edge
    ``citing_case -> cited_case`` from the fuzzy-resolved citations of Step 6.

  * **Hier-SPCNet** - PCNet augmented with a heterogeneous statute layer from
    Step 7:  ``case -> section``, ``case -> act``, ``section -> chapter/part``,
    ``chapter/part -> act``.  Statute nodes carry an **ICF weight** (Inverse
    Citation Frequency = IDF over the case corpus: rare statutes are informative,
    ubiquitous ones like Article 14 are down-weighted) which is also copied onto
    the ``case -> statute`` edges.

Both are ``networkx.DiGraph``. Node ids are namespaced: ``case:``, ``act:``,
``chap:``, ``sec:``. Every node has a ``ntype`` attribute; every edge an
``etype``.

Our demo corpus yields a *sparse* case->case layer (24 edges, CLAUDE.md §3) - the
statute layer is what carries most of the network signal here. Reported honestly.

No FastAPI / DB dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import networkx as nx

from app.ml.citation_extraction import CitationEdge, build_citation_edges
from app.ml.statute_extraction import (
    DocumentStatutes,
    StatuteHierarchy,
    load_hierarchy,
    resolve_document_statutes,
)

NODE_TYPES = ("case", "act", "chapter", "section")
EDGE_TYPES = ("cites_case", "cites_section", "cites_act", "section_of", "chapter_of")


def case_node(doc_id: str) -> str:
    return f"case:{doc_id}"


def act_node(act_id: str) -> str:
    return f"act:{act_id}"


def section_node(section_key: str) -> str:
    return f"sec:{section_key}"


def chapter_node(container_id: str) -> str:
    return f"chap:{container_id}"


# --------------------------------------------------------------------------- #
# PCNet                                                                       #
# --------------------------------------------------------------------------- #

def build_pcnet(
    corpus,
    *,
    citation_edges: Optional[list[CitationEdge]] = None,
    min_score: float = 90.0,
) -> nx.DiGraph:
    g = nx.DiGraph(name="PCNet")
    for doc_id in corpus.doc_ids:
        g.add_node(case_node(doc_id), ntype="case", doc_id=doc_id,
                   year=corpus[doc_id].year, title=corpus[doc_id].title)

    edges = citation_edges if citation_edges is not None else build_citation_edges(
        corpus, min_score=min_score
    )
    for e in edges:
        g.add_edge(
            case_node(e.src_doc_id), case_node(e.dst_doc_id),
            etype="cites_case", weight=round(e.score / 100.0, 4),
            mentions=e.mentions, examples=list(e.examples),
        )
    return g


# --------------------------------------------------------------------------- #
# Hier-SPCNet                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class _StatuteLayerInputs:
    per_doc: dict[str, DocumentStatutes]


def _collect_statutes(corpus, hierarchy: StatuteHierarchy) -> dict[str, DocumentStatutes]:
    return {
        doc_id: resolve_document_statutes(corpus.text(doc_id), hierarchy=hierarchy)
        for doc_id in corpus.doc_ids
    }


def compute_icf(case_to_targets: dict[str, set[str]], n_cases: int) -> dict[str, float]:
    """ICF(t) = ln((N + 1) / (df(t) + 1)) + 1  - IDF over the case corpus."""
    df: dict[str, int] = {}
    for targets in case_to_targets.values():
        for t in targets:
            df[t] = df.get(t, 0) + 1
    return {t: round(math.log((n_cases + 1) / (c + 1)) + 1.0, 4) for t, c in df.items()}


def build_hier_spcnet(
    corpus,
    *,
    citation_edges: Optional[list[CitationEdge]] = None,
    doc_statutes: Optional[dict[str, DocumentStatutes]] = None,
    hierarchy: Optional[StatuteHierarchy] = None,
    min_score: float = 90.0,
    with_icf: bool = True,
) -> nx.DiGraph:
    hierarchy = hierarchy or load_hierarchy()
    g = build_pcnet(corpus, citation_edges=citation_edges, min_score=min_score)
    g.graph["name"] = "Hier-SPCNet"

    doc_statutes = doc_statutes or _collect_statutes(corpus, hierarchy)
    n_cases = corpus.__len__()

    # first pass: gather case -> {section nodes}, case -> {act nodes} for ICF
    case_sections: dict[str, set[str]] = {}
    case_acts: dict[str, set[str]] = {}
    for doc_id, ds in doc_statutes.items():
        case_sections[doc_id] = {section_node(k) for k in ds.section_keys}
        case_acts[doc_id] = {act_node(a) for a in ds.act_ids}

    icf = {}
    if with_icf:
        merged = {d: case_sections.get(d, set()) | case_acts.get(d, set()) for d in doc_statutes}
        icf = compute_icf(merged, n_cases)

    # second pass: add nodes + edges
    for doc_id, ds in doc_statutes.items():
        c = case_node(doc_id)

        for act_id in ds.act_ids:
            an = act_node(act_id)
            if an not in g:
                g.add_node(an, ntype="act", act_id=act_id,
                           canonical=hierarchy.act_canonical(act_id),
                           in_sample_hierarchy=hierarchy.has_act(act_id),
                           icf=icf.get(an))
            g.add_edge(c, an, etype="cites_act", weight=icf.get(an, 1.0))

        # sections + their hierarchy path
        paths = ds.hierarchy_paths(hierarchy)
        for ref in ds.references:
            key = ref.section_key
            if not key:
                continue
            sn = section_node(key)
            if sn not in g:
                g.add_node(sn, ntype="section", section_key=key, number=ref.number,
                           kind=ref.kind, act_id=ref.act_id, icf=icf.get(sn))
            g.add_edge(c, sn, etype="cites_section", weight=icf.get(sn, 1.0))

            path = paths.get(key)
            if path and len(path) == 2:
                (_, act_id, _), (level, container_id, title) = path
                chn = chapter_node(container_id)
                if chn not in g:
                    g.add_node(chn, ntype="chapter", level=level, title=title,
                               act_id=act_id)
                if not g.has_edge(sn, chn):
                    g.add_edge(sn, chn, etype="section_of", weight=1.0)
                if not g.has_edge(chn, act_node(act_id)):
                    g.add_edge(chn, act_node(act_id), etype="chapter_of", weight=1.0)
            elif ref.act_id:
                # section with no curated chapter -> link straight to its Act
                if not g.has_edge(sn, act_node(ref.act_id)):
                    g.add_edge(sn, act_node(ref.act_id), etype="section_of", weight=1.0)

    return g


# --------------------------------------------------------------------------- #
# Summary + persistence                                                       #
# --------------------------------------------------------------------------- #

def graph_summary(g: nx.DiGraph) -> dict:
    nodes_by_type: dict[str, int] = {}
    for _, data in g.nodes(data=True):
        nodes_by_type[data.get("ntype", "?")] = nodes_by_type.get(data.get("ntype", "?"), 0) + 1
    edges_by_type: dict[str, int] = {}
    for _, _, data in g.edges(data=True):
        edges_by_type[data.get("etype", "?")] = edges_by_type.get(data.get("etype", "?"), 0) + 1

    case_nodes = [n for n, d in g.nodes(data=True) if d.get("ntype") == "case"]
    und = g.to_undirected()
    isolated_cases = [n for n in case_nodes if und.degree(n) == 0]

    # most-connected statute nodes: how many distinct CASES cite them directly
    def _cases_citing(node: str) -> int:
        return sum(
            1 for p in g.predecessors(node)
            if g.nodes[p].get("ntype") == "case"
        )

    stat_in = sorted(
        ((n, _cases_citing(n)) for n, d in g.nodes(data=True)
         if d.get("ntype") in {"act", "section"}),
        key=lambda kv: -kv[1],
    )[:15]

    return {
        "name": g.graph.get("name"),
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "nodes_by_type": nodes_by_type,
        "edges_by_type": edges_by_type,
        "n_case_case_edges": edges_by_type.get("cites_case", 0),
        "isolated_cases": len(isolated_cases),
        "weakly_connected_components": nx.number_weakly_connected_components(g),
        "largest_component_size": max((len(c) for c in nx.weakly_connected_components(g)), default=0),
        "most_connected_statutes": [
            {"node": n, "cited_by_cases": d} for n, d in stat_in
        ],
    }


def to_node_link(g: nx.DiGraph) -> dict:
    return nx.node_link_data(g, edges="links")


def from_node_link(data: dict) -> nx.DiGraph:
    return nx.node_link_graph(data, directed=True, edges="links")
