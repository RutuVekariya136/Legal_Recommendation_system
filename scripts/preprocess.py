"""preprocess.py - Step 4: run the preprocessing pipeline over the whole corpus.

For every unique document it writes three artifacts under data/processed/:

    clean/<doc_id>.txt        normalised, de-hyphenated, cased text  (SBERT / display)
    sentences/<doc_id>.json   abbreviation-aware sentence list       (SBERT / explain)
    tokens/<doc_id>.json      lowercased stopword-filtered tokens    (TF-IDF / Doc2Vec)

and a run summary:

    preprocess_report.json    config used + aggregate & per-doc stats

Nothing here is a model or an evaluation number - it is deterministic text
transformation. Config is recorded in the report so a run is reproducible.

Usage:
    python scripts/preprocess.py                     # full corpus, default config
    python scripts/preprocess.py --limit 5           # first 5 docs (quick check)
    python scripts/preprocess.py --legal-boilerplate # + drop legal-boilerplate tokens
    python scripts/preprocess.py --stem              # + Porter stemming
    python scripts/preprocess.py --no-stopwords      # keep stopwords in tokens
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

from app.ml.preprocessing import PreprocessConfig, preprocess_text  # noqa: E402
from app.utils.dataset import load_corpus  # noqa: E402
from app.utils.paths import data_processed_dir, ensure_dir  # noqa: E402


def build_config(args: argparse.Namespace) -> PreprocessConfig:
    return PreprocessConfig(
        min_token_len=args.min_token_len,
        remove_stopwords=not args.no_stopwords,
        remove_legal_boilerplate=args.legal_boilerplate,
        stem=args.stem,
        strip_header_for_tokens=not args.keep_header,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N docs")
    parser.add_argument("--min-token-len", type=int, default=2)
    parser.add_argument("--no-stopwords", action="store_true")
    parser.add_argument("--legal-boilerplate", action="store_true", help="OUR option (off by default)")
    parser.add_argument("--stem", action="store_true", help="Porter stemming (off by default)")
    parser.add_argument("--keep-header", action="store_true", help="include header text in tokens")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    config = build_config(args)
    out_dir = args.out_dir or data_processed_dir()
    clean_dir = ensure_dir(out_dir / "clean")
    sent_dir = ensure_dir(out_dir / "sentences")
    tok_dir = ensure_dir(out_dir / "tokens")

    corpus = load_corpus(verify=True)
    doc_ids = corpus.doc_ids[: args.limit] if args.limit else corpus.doc_ids

    print(f"preprocessing {len(doc_ids)} documents -> {out_dir.relative_to(_REPO_ROOT).as_posix()}/")
    print(f"config: {asdict(config)}")

    per_doc = []
    vocab: Counter[str] = Counter()
    n_header = 0
    for i, doc_id in enumerate(doc_ids, 1):
        pd = preprocess_text(corpus.text(doc_id), doc_id=doc_id, config=config)

        (clean_dir / f"{doc_id}.txt").write_text(pd.clean_text, encoding="utf-8")
        (sent_dir / f"{doc_id}.json").write_text(
            json.dumps(pd.sentences, ensure_ascii=False), encoding="utf-8"
        )
        (tok_dir / f"{doc_id}.json").write_text(
            json.dumps(pd.tokens, ensure_ascii=False), encoding="utf-8"
        )

        vocab.update(pd.tokens)
        n_header += int(pd.stats["header_detected"])
        per_doc.append({"doc_id": doc_id, **pd.stats})
        if i % 50 == 0 or i == len(doc_ids):
            print(f"  {i}/{len(doc_ids)}")

    tok_counts = [d["n_tokens"] for d in per_doc]
    sent_counts = [d["n_sentences"] for d in per_doc]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "n_documents": len(per_doc),
        "headers_detected": n_header,
        "corpus_vocab_size": len(vocab),
        "total_tokens": sum(tok_counts),
        "tokens_per_doc": {
            "min": min(tok_counts),
            "median": int(statistics.median(tok_counts)),
            "mean": round(statistics.fmean(tok_counts), 1),
            "max": max(tok_counts),
        },
        "sentences_per_doc": {
            "min": min(sent_counts),
            "median": int(statistics.median(sent_counts)),
            "mean": round(statistics.fmean(sent_counts), 1),
            "max": max(sent_counts),
        },
        "top_30_tokens": vocab.most_common(30),
        "documents": per_doc,
    }
    report_path = out_dir / "preprocess_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n-- summary --")
    print(f"  headers detected:   {n_header}/{len(per_doc)}")
    print(f"  corpus vocab size:  {len(vocab):,}")
    print(f"  tokens/doc:         min {report['tokens_per_doc']['min']}, "
          f"median {report['tokens_per_doc']['median']}, "
          f"mean {report['tokens_per_doc']['mean']}, "
          f"max {report['tokens_per_doc']['max']}")
    print(f"  sentences/doc:      min {report['sentences_per_doc']['min']}, "
          f"median {report['sentences_per_doc']['median']}, "
          f"max {report['sentences_per_doc']['max']}")
    print(f"  top tokens:         {', '.join(t for t, _ in vocab.most_common(12))}")
    print(f"\n[ OK ] wrote {report_path.relative_to(_REPO_ROOT).as_posix()} "
          f"and clean/ sentences/ tokens/ ({len(per_doc)} files each).")
    print("       Proceed to Step 5 (PDF/Text Extraction).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
