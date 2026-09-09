"""extract_statutes.py - Step 7: extract & resolve statute references.

For every document it regex-extracts Section / Article references, resolves the
Act via the curated alias table, resolves anaphoric "the Act" to the judgment's
primary Act, and (for Acts in the SAMPLE hierarchy) records the
Act -> Chapter/Part -> Section path.

Writes data/processed/statutes/<doc_id>.json + statute_report.json. These feed
Hier-SPCNet construction in Step 9.

Honest scope: the hierarchy is a small hand-curated sample; 274 of ~297 distinct
Acts here are NOT in it and get a bare Act node (never dropped/invented), per
PROJECT_NOTES.md §3.

Usage:
    python scripts/extract_statutes.py
    python scripts/extract_statutes.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.ml.statute_extraction import (  # noqa: E402
    find_primary_act_definition,
    load_hierarchy,
    resolve_document_statutes,
)
from app.utils.dataset import load_corpus  # noqa: E402
from app.utils.paths import data_processed_dir, ensure_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    hierarchy = load_hierarchy()
    corpus = load_corpus(verify=True)
    out_dir = args.out_dir or data_processed_dir()
    stat_dir = ensure_dir(out_dir / "statutes")
    doc_ids = corpus.doc_ids[: args.limit] if args.limit else corpus.doc_ids

    curated = hierarchy.curated_act_ids
    print(f"extracting statutes for {len(doc_ids)} docs "
          f"(curated hierarchy: {len(curated)} Acts)")

    total_refs = 0
    kind_counts: Counter[str] = Counter()
    act_doc_freq: Counter[str] = Counter()
    section_freq: Counter[str] = Counter()
    anaphoric_resolved = anaphoric_unresolved = 0
    explicit_defs = 0
    n_hier_paths = 0
    per_doc = []

    for i, did in enumerate(doc_ids, 1):
        text = corpus.text(did)
        ds = resolve_document_statutes(text, hierarchy=hierarchy)
        pm = find_primary_act_definition(text, hierarchy=hierarchy)
        explicit_defs += int(pm is not None)

        total_refs += len(ds.references)
        for r in ds.references:
            kind_counts[r.kind] += 1
            if r.section_key:
                section_freq[r.section_key] += 1
        for a in ds.act_ids:
            act_doc_freq[a] += 1
        anaphoric_resolved += ds.n_anaphoric_resolved
        anaphoric_unresolved += ds.n_unresolved
        paths = ds.hierarchy_paths(hierarchy)
        n_hier_paths += len(paths)

        (stat_dir / f"{did}.json").write_text(
            json.dumps(
                {
                    "doc_id": did,
                    "primary_act_id": ds.primary_act_id,
                    "primary_act_canonical": ds.primary_act_canonical,
                    "primary_act_from_explicit_definition": pm is not None,
                    "act_ids": ds.act_ids,
                    "section_keys": ds.section_keys,
                    "n_references": len(ds.references),
                    "n_anaphoric_resolved": ds.n_anaphoric_resolved,
                    "n_unresolved": ds.n_unresolved,
                    "hierarchy_paths": paths,
                    "references": [asdict(r) for r in ds.references],
                },
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        per_doc.append({
            "doc_id": did, "n_references": len(ds.references),
            "n_acts": len(ds.act_ids), "primary_act_id": ds.primary_act_id,
        })
        if i % 50 == 0 or i == len(doc_ids):
            print(f"  {i}/{len(doc_ids)}")

    distinct_acts = list(act_doc_freq)
    in_curated = [a for a in distinct_acts if a in curated]
    ref_counts = [d["n_references"] for d in per_doc]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hierarchy_meta": hierarchy.meta,
        "n_documents": len(per_doc),
        "total_references": total_refs,
        "references_by_kind": dict(kind_counts),
        "docs_with_any_statute": sum(1 for d in per_doc if d["n_acts"]),
        "docs_with_explicit_primary_definition": explicit_defs,
        "anaphoric_the_act_resolved": anaphoric_resolved,
        "anaphoric_the_act_unresolved": anaphoric_unresolved,
        "distinct_acts": len(distinct_acts),
        "distinct_acts_in_curated_hierarchy": len(in_curated),
        "distinct_acts_bare_node_only": len(distinct_acts) - len(in_curated),
        "section_to_hierarchy_paths": n_hier_paths,
        "references_per_doc": {
            "min": min(ref_counts), "median": int(statistics.median(ref_counts)),
            "mean": round(statistics.fmean(ref_counts), 1), "max": max(ref_counts),
        },
        "top_acts_by_doc_frequency": act_doc_freq.most_common(20),
        "top_sections": section_freq.most_common(20),
        "non_curated_acts_sample": [a for a in distinct_acts if a not in curated][:40],
        "documents": per_doc,
    }
    (out_dir / "statute_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n-- summary --")
    print(f"  total references:        {total_refs}  {dict(kind_counts)}")
    print(f"  docs with >=1 statute:   {report['docs_with_any_statute']}/{len(per_doc)}")
    print(f"  explicit 'the Act' defs: {explicit_defs}")
    print(f"  anaphoric 'the Act':     {anaphoric_resolved} resolved, {anaphoric_unresolved} unresolved")
    print(f"  distinct Acts:           {len(distinct_acts)}  "
          f"({len(in_curated)} in curated hierarchy, "
          f"{len(distinct_acts) - len(in_curated)} bare Act nodes)")
    print(f"  section->hierarchy paths: {n_hier_paths}")
    print("  top Acts:")
    for a, ct in act_doc_freq.most_common(8):
        tag = "curated" if a in curated else "bare"
        print(f"    {ct:3d} docs  {a}  [{tag}]")
    print(f"\n[ OK ] wrote statutes/ ({len(per_doc)} files) and statute_report.json")
    print("       Proceed to Step 8 (Text Similarity).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
