"""build_statute_hierarchy.py - Phase 2 P6: build a statute hierarchy from a CSV.

Turns a flat CSV into the JSON shape `app.ml.statute_extraction` expects, so you
can feed an India Code parse / a section-list scrape without hand-editing JSON.
Point the app at the result with `STATUTE_HIERARCHY_PATH=...` - it is *merged
over* the built-in curated sample (existing 41 Acts stay).

CSV columns (header required; blanks allowed):

    act_id,canonical,year,alias,container_id,container_title,section,section_title

  * one row per section (or one row per Act to add just an alias / Act node)
  * `alias` may repeat the act_id row-to-row or be given once
  * `container_id` = chapter/part label (e.g. "XVI", "III"); `container_title` its name

Example:
    ipc_1860,"Indian Penal Code, 1860",1860,indian penal code,XVI,"Of Offences Affecting the Human Body",302,"Punishment for murder"

Usage:
    python scripts/build_statute_hierarchy.py --csv acts.csv --out data/processed/hierarchy.json
    STATUTE_HIERARCHY_PATH=data/processed/hierarchy.json python scripts/build_all.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--merge-into", type=Path, default=None,
                   help="an existing hierarchy JSON to merge these rows on top of")
    args = p.parse_args(argv)

    acts: dict[str, dict] = {}
    aliases: dict[str, str] = {}

    with args.csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            aid = (row.get("act_id") or "").strip()
            if not aid:
                continue
            act = acts.setdefault(aid, {
                "canonical": (row.get("canonical") or aid).strip(),
                "year": int(row["year"]) if row.get("year", "").strip().isdigit() else None,
                "chapters": {},
            })
            if row.get("canonical", "").strip():
                act["canonical"] = row["canonical"].strip()
            al = (row.get("alias") or "").strip().lower()
            if al:
                aliases[al] = aid
            cid = (row.get("container_id") or "").strip()
            sec = (row.get("section") or "").strip()
            if sec:
                ch = act["chapters"].setdefault(cid or "-", {
                    "title": (row.get("container_title") or cid or "").strip(), "sections": {}})
                ch["sections"][sec] = (row.get("section_title") or "").strip()

    data = {"aliases": aliases, "acts": acts}
    if args.merge_into and args.merge_into.is_file():
        from app.ml.statute_extraction import _deep_merge

        base = json.loads(args.merge_into.read_text(encoding="utf-8"))
        data = _deep_merge(base, data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    n_sec = sum(len(c.get("sections", {})) for a in data["acts"].values()
                for c in a.get("chapters", {}).values())
    print(f"[ OK ] {len(data['acts'])} Acts, {len(data['aliases'])} aliases, "
          f"{n_sec} sections -> {args.out}")
    print(f"       use it:  STATUTE_HIERARCHY_PATH={args.out} python scripts/build_all.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
