"""ingest_bulk.py - load a large judgment corpus into the database (Phase 2).

The 296 base-paper documents stay as `corpus="base"` (they carry the expert
pairs). A bulk corpus is stored with its own name and is used for search /
recommendation only - it has no gold scores, so evaluation always stays on the
base subset.

Sources
-------
  --source dir   --path FOLDER            *.txt / *.pdf files (recurses)
  --source jsonl --path FILE.jsonl        one JSON object per line
                 [--text-field text --id-field id --title-field title --year-field year]
  --source hf    --hf-dataset NAME        a HuggingFace dataset (needs `pip install datasets`)
                 [--hf-split train --hf-config CFG]

Examples
--------
  # ILDC (Indian Legal Documents Corpus, ~35k Supreme Court judgments)
  pip install datasets
  python scripts/ingest_bulk.py --source hf --hf-dataset Exploration-Lab/IL-TUR \\
      --hf-config cjpe --hf-split train --corpus ildc --limit 5000 --text-field text

  # a folder of PDFs you already have
  python scripts/ingest_bulk.py --source dir --path ./my_judgments --corpus local

After ingesting, rebuild over the full corpus:
  CORPUS_SOURCE=db python scripts/build_all.py
  CORPUS_SOURCE=db uvicorn app.main:app --app-dir backend
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.database import init_db, session_scope  # noqa: E402
from app.services.repository import bulk_insert_documents  # noqa: E402
from app.utils.dataset import extract_title, parse_year  # noqa: E402


def _rows_from_dir(path: Path):
    from app.ml.text_extraction import ExtractionError, extract_text

    for p in sorted(path.rglob("*")):
        if p.suffix.lower() not in {".txt", ".pdf"}:
            continue
        try:
            text = extract_text(p).raw_text
        except ExtractionError as exc:
            print(f"  skip {p.name}: {exc}", file=sys.stderr)
            continue
        yield {
            "doc_id": p.stem, "text": text,
            "title": extract_title(text), "year": parse_year(p.stem),
            "source_path": str(p),
        }


def _rows_from_jsonl(path: Path, args):
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(args.text_field, "")
            doc_id = str(obj.get(args.id_field, f"{path.stem}_{i}"))
            title = obj.get(args.title_field) if args.title_field else None
            year = obj.get(args.year_field) if args.year_field else None
            yield {"doc_id": doc_id, "text": text,
                   "title": title or (extract_title(text) if text else None),
                   "year": year or parse_year(doc_id)}


def _rows_from_hf(args):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("`--source hf` needs: pip install datasets")
    kw = {}
    if args.hf_config:
        kw["name"] = args.hf_config
    ds = load_dataset(args.hf_dataset, split=args.hf_split, **kw)
    for i, obj in enumerate(ds):
        text = obj.get(args.text_field, "")
        doc_id = str(obj.get(args.id_field, f"{args.hf_dataset.split('/')[-1]}_{i}"))
        title = obj.get(args.title_field) if args.title_field else None
        year = obj.get(args.year_field) if args.year_field else None
        yield {"doc_id": doc_id, "text": text,
               "title": title or (extract_title(text) if text else None),
               "year": year or parse_year(doc_id)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, choices=["dir", "jsonl", "hf"])
    p.add_argument("--path", type=Path)
    p.add_argument("--hf-dataset")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--hf-config", default=None)
    p.add_argument("--corpus", default="bulk", help="corpus name for these docs")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--text-field", default="text")
    p.add_argument("--id-field", default="id")
    p.add_argument("--title-field", default=None)
    p.add_argument("--year-field", default=None)
    p.add_argument("--replace", action="store_true", help="overwrite docs with the same id/sha")
    args = p.parse_args(argv)

    if args.corpus == "base":
        p.error("'base' is reserved for the expert-scored 296 docs; use another --corpus name")
    if args.source in {"dir", "jsonl"} and not args.path:
        p.error("--path is required for --source dir/jsonl")
    if args.source == "hf" and not args.hf_dataset:
        p.error("--hf-dataset is required for --source hf")

    init_db()

    if args.source == "dir":
        gen = _rows_from_dir(args.path)
    elif args.source == "jsonl":
        gen = _rows_from_jsonl(args.path, args)
    else:
        gen = _rows_from_hf(args)

    rows, t0 = [], time.time()
    for row in gen:
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break
        if len(rows) % 2000 == 0:
            print(f"  read {len(rows)} ...")
    print(f"read {len(rows)} documents in {time.time() - t0:.1f}s")

    with session_scope() as db:
        result = bulk_insert_documents(db, rows, corpus_name=args.corpus,
                                       skip_existing=not args.replace)
    print(f"\ninserted {result['inserted']}, skipped {result['skipped']}")
    print(f"database now holds {result['total']} cases: {result['by_corpus']}")
    print("\nnext:")
    print("  CORPUS_SOURCE=db python scripts/build_all.py")
    print("  CORPUS_SOURCE=db uvicorn app.main:app --app-dir backend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
