"""build_graph.py - Step 9: construct PCNet and Hier-SPCNet.

Reads the corpus, resolves case citations (Step 6) and statute references
(Step 7), and assembles two networkx DiGraphs:

    data/graphs/pcnet.json          cases + citing->cited edges
    data/graphs/hier_spcnet.json    PCNet + statute layer (act / chapter / section)
    data/graphs/graph_report.json   node/edge-type counts, connectivity, top statutes

Usage:
    python scripts/build_graph.py
    python scripts/build_graph.py --min-score 88 --no-icf
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.ml.citation_extraction import build_citation_edges  # noqa: E402
from app.ml.network_model import (  # noqa: E402
    build_hier_spcnet,
    build_pcnet,
    graph_summary,
    to_node_link,
)
from app.ml.statute_extraction import load_hierarchy  # noqa: E402
from app.utils.dataset import load_corpus  # noqa: E402
from app.utils.paths import data_graphs_dir, ensure_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-score", type=float, default=90.0)
    parser.add_argument("--no-icf", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    out_dir = ensure_dir(args.out_dir or data_graphs_dir())
    corpus = load_corpus(verify=True)
    hierarchy = load_hierarchy()

    print("resolving case citations ...")
    t0 = time.time()
    cite_edges = build_citation_edges(corpus, min_score=args.min_score)
    print(f"  {len(cite_edges)} case->case edges  ({time.time() - t0:.1f}s)")

    print("building PCNet ...")
    pcnet = build_pcnet(corpus, citation_edges=cite_edges)

    print("building Hier-SPCNet ...")
    t0 = time.time()
    hier = build_hier_spcnet(
        corpus, citation_edges=cite_edges, hierarchy=hierarchy,
        min_score=args.min_score, with_icf=not args.no_icf,
    )
    print(f"  done ({time.time() - t0:.1f}s)")

    pcnet_sum = graph_summary(pcnet)
    hier_sum = graph_summary(hier)

    (out_dir / "pcnet.json").write_text(
        json.dumps(to_node_link(pcnet), ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "hier_spcnet.json").write_text(
        json.dumps(to_node_link(hier), ensure_ascii=False), encoding="utf-8"
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_score": args.min_score,
        "icf": not args.no_icf,
        "pcnet": pcnet_sum,
        "hier_spcnet": hier_sum,
    }
    (out_dir / "graph_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for name, s in (("PCNet", pcnet_sum), ("Hier-SPCNet", hier_sum)):
        print(f"\n-- {name} --")
        print(f"  nodes {s['n_nodes']}  edges {s['n_edges']}")
        print(f"  nodes_by_type: {s['nodes_by_type']}")
        print(f"  edges_by_type: {s['edges_by_type']}")
        print(f"  case->case edges: {s['n_case_case_edges']}")
        print(f"  isolated cases:   {s['isolated_cases']}")
        print(f"  weak components:  {s['weakly_connected_components']} "
              f"(largest {s['largest_component_size']})")
    print("\n  most-connected statutes (Hier-SPCNet):")
    for item in hier_sum["most_connected_statutes"][:8]:
        print(f"    {item['cited_by_cases']:3d} cases  {item['node']}")

    print(f"\n[ OK ] wrote pcnet.json, hier_spcnet.json, graph_report.json -> "
          f"{out_dir.relative_to(_REPO_ROOT).as_posix()}/")
    print("       Proceed to Step 10 (Network Embedding).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
