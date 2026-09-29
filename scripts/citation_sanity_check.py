"""citation_sanity_check.py - weak-supervision evaluation at full corpus scale.

    CORPUS_SOURCE=db python scripts/citation_sanity_check.py

There are no expert-scored pairs at the 33k-document scale (those only exist
for the 296-document base corpus - see EVALUATION.md). This script answers a
weaker, but still legitimate, question using data we DO have at scale: the
resolved case-to-case citation graph (`data/graphs/pcnet.json`).

Claim under test: if case A cites case B, the hybrid model should generally
score (A, B) higher than a random, uncited pair. This is a much weaker signal
than expert similarity judgments (a citation doesn't mean "similar", and many
genuinely similar cases don't cite each other), but it is real, independent
ground truth, requires no new labelling, and is honest about being a
different, easier question than the base-corpus evaluation.

Method: for every citing case A -> cited case B edge, score (A, B) against the
true target B plus `--distractors` random other documents, and record where B
ranks among them (1 = the model correctly rated the true citation highest).
Report mean percentile rank and hit-rate, for the cited pairs vs. a fully
random baseline (real B replaced by a random doc too), so the comparison
isn't just "is citation-following non-trivial" but "does the model do better
than chance at it".

Writes data/processed/citation_sanity_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

import numpy as np  # noqa: E402

from app.ml.evaluation import classification_metrics  # noqa: E402
from app.ml.hybrid_similarity import HybridSimilarityModel  # noqa: E402
from app.ml.network_embedding import NetworkEmbeddingIndex  # noqa: E402
from app.ml.network_similarity import NetworkSimilarityIndex  # noqa: E402
from app.ml.text_similarity import TextSimilarityIndex  # noqa: E402
from app.utils.dataset import load_corpus  # noqa: E402
from app.utils.paths import data_embeddings_dir, data_graphs_dir, data_processed_dir  # noqa: E402

EMB = data_embeddings_dir()


def _load_citation_edges() -> list[tuple[str, str]]:
    pcnet = json.loads((data_graphs_dir() / "pcnet.json").read_text(encoding="utf-8"))
    edges = []
    for link in pcnet["links"]:
        if link.get("etype") != "cites_case":
            continue
        src = link["source"].removeprefix("case:")
        dst = link["target"].removeprefix("case:")
        edges.append((src, dst))
    return edges


def _percentile_rank(true_score: float, distractor_scores: np.ndarray) -> float:
    """1.0 = true target beat every distractor; 0.0 = beaten by all of them."""
    return float((distractor_scores < true_score).sum()) / len(distractor_scores)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--distractors", type=int, default=99,
                        help="random alternative documents to rank the true citation against")
    parser.add_argument("--max-edges", type=int, default=None,
                        help="cap the number of citation edges checked (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    t0 = time.time()
    corpus = load_corpus(verify=False)
    ti = TextSimilarityIndex.load(EMB / "tfidf")
    ei = NetworkEmbeddingIndex.load(EMB / "network_metapath2vec")
    nsi = NetworkSimilarityIndex.load(EMB / "network_similarity", ei, corpus=corpus)
    hm = HybridSimilarityModel.load(EMB / "hybrid", ti, nsi)
    print(f"loaded corpus ({len(corpus)} docs) + model in {time.time() - t0:.1f}s")

    edges = _load_citation_edges()
    doc_ids = [d for d in corpus.doc_ids if ti.has(d)]
    doc_id_arr = np.array(doc_ids)
    rng = np.random.default_rng(args.seed)

    edges = [(a, b) for a, b in edges if ti.has(a) and ti.has(b)]
    if args.max_edges:
        edges = edges[: args.max_edges]
    print(f"scoring {len(edges)} citation edges against {args.distractors} random distractors each ...")

    cited_ranks, random_ranks = [], []
    cited_hit1, random_hit1 = 0, 0
    cited_scores_for_cls, random_scores_for_cls = [], []
    t1 = time.time()
    for i, (a, b) in enumerate(edges):
        distractor_ids = rng.choice(doc_id_arr, size=args.distractors + 1, replace=False)
        distractor_ids = [d for d in distractor_ids if d not in (a, b)][: args.distractors]
        distractor_scores = np.array([hm.score(a, d) for d in distractor_ids])

        true_score = hm.score(a, b)
        rank = _percentile_rank(true_score, distractor_scores)
        cited_ranks.append(rank)
        cited_scores_for_cls.append(true_score)
        if (distractor_scores < true_score).all():
            cited_hit1 += 1

        # same query A, but the "target" is itself a random doc - the null baseline
        random_target = distractor_ids[0] if distractor_ids else b
        random_score = hm.score(a, random_target)
        random_scores_for_cls.append(random_score)
        rrank = _percentile_rank(random_score, distractor_scores[1:] if len(distractor_scores) > 1 else distractor_scores)
        random_ranks.append(rrank)
        if len(distractor_scores) > 1 and (distractor_scores[1:] < random_score).all():
            random_hit1 += 1

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(edges)} ({time.time() - t1:.0f}s)")

    n = len(edges)

    # -- classification framing: cited=1 vs random=0, a balanced set (n + n) --
    # threshold = median of the pooled scores - principled and not picked by
    # looking at the labels (which would leak / overstate the result).
    pooled = np.array(cited_scores_for_cls + random_scores_for_cls)
    y_true = [1] * n + [0] * n
    thr = float(np.median(pooled)) if n else 0.5
    clsm = classification_metrics(pooled.tolist(), y_true, gold_threshold=0.5, pred_threshold=thr)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "for each citing->cited edge, rank the true cited doc's hybrid score "
                  "against N random distractors (percentile rank, 1.0 = beat all)",
        "n_edges": n,
        "n_distractors": args.distractors,
        "cited_pairs": {
            "mean_percentile_rank": round(float(np.mean(cited_ranks)), 4),
            "median_percentile_rank": round(float(np.median(cited_ranks)), 4),
            "hit_rate_beats_all_distractors": round(cited_hit1 / n, 4) if n else None,
        },
        "random_baseline": {
            "mean_percentile_rank": round(float(np.mean(random_ranks)), 4),
            "median_percentile_rank": round(float(np.median(random_ranks)), 4),
            "hit_rate_beats_all_distractors": round(random_hit1 / n, 4) if n else None,
        },
        "classification": {
            **clsm.to_dict(),
            "note": "balanced set: n cited pairs (label=1) + n random pairs (label=0); "
                    "pred_threshold is the median of the pooled scores, not tuned on the labels",
        },
    }

    out = data_processed_dir() / "citation_sanity_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\ncited pairs   : mean percentile rank {report['cited_pairs']['mean_percentile_rank']:.3f}  "
          f"(beats all {args.distractors} distractors {report['cited_pairs']['hit_rate_beats_all_distractors']*100:.1f}% of the time)")
    print(f"random baseline: mean percentile rank {report['random_baseline']['mean_percentile_rank']:.3f}  "
          f"(beats all {args.distractors} distractors {report['random_baseline']['hit_rate_beats_all_distractors']*100:.1f}% of the time)")
    print(f"\nclassification (cited vs random, threshold={thr:.3f}):")
    print(f"  accuracy={clsm.accuracy:.3f}  precision={clsm.precision:.3f}  "
          f"recall={clsm.recall:.3f}  f1={clsm.f1:.3f}  roc_auc={clsm.roc_auc:.3f}")
    print(f"\n[ OK ] wrote {out.relative_to(_REPO_ROOT).as_posix()}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
