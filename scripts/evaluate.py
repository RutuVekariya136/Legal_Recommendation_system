"""evaluate.py - Step 22: benchmark every similarity method against the real
expert-scored pairs (100 validation + 90 test).

    python scripts/evaluate.py                 # full matrix + diagnosis
    python scripts/evaluate.py --quick         # skip Doc2Vec / node2vec configs

Protocol (honest, no leakage):
  * text-only / network-only / weighted-hybrid: direct correlation on each split.
  * learned-hybrid: 5-fold CV on validation; trained on ALL validation, tested on test.

Writes data/processed/evaluation_report.json and prints Markdown tables. Every
number is computed here from data/raw/ - nothing is hard-coded.
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

import numpy as np  # noqa: E402

from app.ml.evaluation import (  # noqa: E402
    ReliabilityCalibrator, classification_metrics, correlation_metrics, diagnose_split,
    evaluate_pairwise, ranking_sanity, youden_threshold,
)

GOLD_THRESHOLD = 0.5  # matches ranking_sanity's own "similar" cutoff, for consistency
from app.ml.hybrid_similarity import HybridFeatures, HybridSimilarityModel, LearnedCombiner, WeightedSumCombiner  # noqa: E402
from app.ml.network_embedding import NetworkEmbeddingIndex  # noqa: E402
from app.ml.network_similarity import NetworkSimilarityIndex  # noqa: E402
from app.ml.text_similarity import TextSimilarityIndex, TfidfEmbedder  # noqa: E402
from app.utils.dataset import load_corpus  # noqa: E402
from app.utils.paths import data_embeddings_dir, data_processed_dir  # noqa: E402

EMB = data_embeddings_dir()


def _load_text(method: str, corpus):
    d = EMB / method
    if (d / "index.json").is_file():
        idx = TextSimilarityIndex.load(d)
        if idx.embedder is not None or method != "tfidf":
            return idx
    if method == "tfidf":
        return TextSimilarityIndex.build(corpus, TfidfEmbedder())
    return None


def _corr_and_rank(score_fn, corpus, pairs, split):
    cm = evaluate_pairwise(score_fn, pairs)
    pred = [float(score_fn(p.doc_id_1, p.doc_id_2)) for p in pairs]
    gold = [float(p.score) for p in pairs]
    clsm = classification_metrics(pred, gold, gold_threshold=GOLD_THRESHOLD)
    # rank_fn: order all corpus docs by score_fn(query, other) desc
    doc_ids = corpus.doc_ids

    def rank_fn(q):
        scored = sorted(((score_fn(q, o), o) for o in doc_ids if o != q), key=lambda t: -t[0])
        return [o for _, o in scored]

    rm = ranking_sanity(rank_fn, pairs, corpus)
    return {"split": split, "correlation": cm.to_dict(), "ranking": rm.to_dict(),
            "classification": clsm.to_dict()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    corpus = load_corpus(verify=True)
    val, test = corpus.pairs_by_split["validation"], corpus.pairs_by_split["test"]

    ti = _load_text("tfidf", corpus)
    ei = NetworkEmbeddingIndex.load(EMB / "network_metapath2vec")
    nsi = NetworkSimilarityIndex.load(EMB / "network_similarity", ei, corpus=corpus)

    si = TextSimilarityIndex.load(EMB / "sbert") if (EMB / "sbert" / "index.json").is_file() else None

    configs: list[tuple[str, str, callable]] = []  # (name, kind, score_fn)

    # -- text-only --
    configs.append(("text: tfidf", "text", lambda a, b: ti.pair_similarity(a, b)))
    if not args.quick and (EMB / "doc2vec" / "index.json").is_file():
        d2v = TextSimilarityIndex.load(EMB / "doc2vec")
        configs.append(("text: doc2vec", "text", lambda a, b: d2v.pair_similarity(a, b)))
    if si is not None:
        configs.append(("text: sbert (our enhancement)", "text", lambda a, b: si.pair_similarity(a, b)))

    # -- network-only --
    nsi.method = "embedding"
    configs.append(("network: metapath2vec (embedding)", "network",
                    lambda a, b: nsi.embedding_score(a, b)))
    configs.append(("network: metapath2vec (blend)", "network",
                    lambda a, b: _blend(nsi, a, b)))
    if not args.quick and (EMB / "network_node2vec" / "index.json").is_file():
        ei2 = NetworkEmbeddingIndex.load(EMB / "network_node2vec")
        nsi2 = NetworkSimilarityIndex(ei2, nsi.features, method="embedding")
        configs.append(("network: node2vec (embedding)", "network",
                        lambda a, b: nsi2.embedding_score(a, b)))

    # -- weighted hybrid --
    for al, be in ((0.5, 0.5), (0.7, 0.3), (0.3, 0.7)):
        hm = HybridSimilarityModel(ti, nsi, WeightedSumCombiner(al, be))
        configs.append((f"hybrid: weighted {al}/{be}", "weighted",
                        (lambda m: (lambda a, b: m.score(a, b)))(hm)))

    # -- trivial baselines (for the diagnosis) --
    configs.append(("baseline: year proximity", "baseline",
                    lambda a, b: 1.0 - abs((corpus[a].year or 0) - (corpus[b].year or 0)) / 100.0))
    _rng = np.random.default_rng(0)
    configs.append(("baseline: random", "baseline", lambda a, b: float(_rng.random())))

    results: list[dict] = []
    print("running correlation + ranking for direct-scored configs ...")
    for name, kind, fn in configs:
        t0 = time.time()
        entry = {"config": name, "kind": kind, "splits": {}}
        for split, pairs in (("validation", val), ("test", test)):
            entry["splits"][split] = _corr_and_rank(fn, corpus, pairs, split)
        results.append(entry)
        print(f"  {name:<38} "
              f"val rho={entry['splits']['validation']['correlation']['spearman']:+.3f}  "
              f"test rho={entry['splits']['test']['correlation']['spearman']:+.3f}  "
              f"({time.time() - t0:.1f}s)")

    # -- learned hybrid (special protocol) --
    print("\nrunning learned-hybrid configs (CV on validation, train-all-val for test) ...")
    nsi.method = "blend"
    hm = HybridSimilarityModel(ti, nsi, LearnedCombiner(model="linear"))
    feat_specs = {
        "hybrid: learned linear (text+net)": ("linear", ("text_score", "network_score")),
        "hybrid: learned linear (all feats)": ("linear",
            ("text_score", "network_score", "embedding_score", "statute_overlap",
             "citation_overlap", "cocitation_count")),
        "hybrid: learned mlp (text+net)": ("mlp", ("text_score", "network_score")),
    }
    val_feats = [hm.features(p.doc_id_1, p.doc_id_2) for p in val]
    val_gold = [p.score for p in val]
    test_feats = [hm.features(p.doc_id_1, p.doc_id_2) for p in test]
    test_gold = [p.score for p in test]

    reliability_report = None
    for name, (mdl, feats) in feat_specs.items():
        lc = LearnedCombiner(model=mdl, features=feats)
        cv_pred = lc.cross_val_predict(val_feats, val_gold, folds=5)
        lc.fit(val_feats, val_gold)
        test_pred = [lc.score(f) for f in test_feats]
        # Threshold tuned via Youden's J on the out-of-sample CV predictions
        # only (never on test, never on in-sample predictions) - a
        # principled alternative to an arbitrary fixed 0.5 cutoff for a raw
        # score that isn't centered on 0.5. Applied as-is to test (no
        # re-tuning there), matching the "trained on all validation" protocol
        # already used for the combiner itself.
        cls_thr = youden_threshold(cv_pred, val_gold, gold_threshold=GOLD_THRESHOLD)

        if name == "hybrid: learned linear (text+net)":
            # This is the config the deployed RecommenderService actually
            # uses (TF-IDF text + network, learned linear) - calibrate on its
            # out-of-sample validation predictions and save alongside it.
            cal = ReliabilityCalibrator(gold_threshold=GOLD_THRESHOLD).fit(cv_pred, val_gold)
            cal.save(EMB / "hybrid" / "calibration.joblib")
            # honest check: does the SAME calibration hold up on test? (it may
            # not, given §3's test-split diagnosis - report it either way)
            test_check = classification_metrics(
                [cal.reliability(s) for s in test_pred], test_gold, gold_threshold=GOLD_THRESHOLD,
                pred_threshold=0.5,
            )
            reliability_report = {
                "fitted_on": "5-fold CV predictions, 100 validation pairs",
                "gold_threshold": GOLD_THRESHOLD,
                "curve": cal.curve(),
                "test_split_check": {
                    "note": "applying the validation-fitted calibration to the test split - "
                            "a sanity check, not a validated guarantee (see §3's test-split diagnosis)",
                    **test_check.to_dict(),
                },
            }
            print(f"  saved reliability calibration -> {EMB / 'hybrid' / 'calibration.joblib'}")

        entry = {
            "config": name, "kind": "learned",
            "splits": {
                "validation": {"split": "validation",
                               "correlation": correlation_metrics(cv_pred, val_gold).to_dict(),
                               "classification": classification_metrics(
                                   cv_pred, val_gold, gold_threshold=GOLD_THRESHOLD,
                                   pred_threshold=cls_thr).to_dict(),
                               "note": "5-fold CV; threshold tuned via Youden's J on these same "
                                       "out-of-sample CV predictions"},
                "test": {"split": "test",
                         "correlation": correlation_metrics(test_pred, test_gold).to_dict(),
                         "classification": classification_metrics(
                             test_pred, test_gold, gold_threshold=GOLD_THRESHOLD,
                             pred_threshold=cls_thr).to_dict(),
                         "note": "trained on all validation; threshold carried over from "
                                 "validation CV, not re-tuned on test"},
            },
        }
        if mdl == "linear":
            entry["coef_"] = {f: round(c, 4) for f, c in zip(feats, np.ravel(lc._model.coef_))}
            entry["intercept_"] = round(float(lc._model.intercept_), 4)
        results.append(entry)
        print(f"  {name:<38} "
              f"val rho={entry['splits']['validation']['correlation']['spearman']:+.3f}  "
              f"test rho={entry['splits']['test']['correlation']['spearman']:+.3f}")

    # -- sbert-based learned hybrid (same protocol, swap text index) --
    if si is not None:
        hm_sbert = HybridSimilarityModel(si, nsi, LearnedCombiner(model="linear"))
        sbert_feat_specs = {
            "hybrid: learned linear (sbert+net)": ("linear", ("text_score", "network_score")),
        }
        val_feats_s = [hm_sbert.features(p.doc_id_1, p.doc_id_2) for p in val]
        test_feats_s = [hm_sbert.features(p.doc_id_1, p.doc_id_2) for p in test]
        for name, (mdl, feats) in sbert_feat_specs.items():
            lc = LearnedCombiner(model=mdl, features=feats)
            cv_pred = lc.cross_val_predict(val_feats_s, val_gold, folds=5)
            lc.fit(val_feats_s, val_gold)
            test_pred = [lc.score(f) for f in test_feats_s]
            cls_thr = youden_threshold(cv_pred, val_gold, gold_threshold=GOLD_THRESHOLD)
            entry = {
                "config": name, "kind": "learned",
                "splits": {
                    "validation": {"split": "validation",
                                   "correlation": correlation_metrics(cv_pred, val_gold).to_dict(),
                                   "classification": classification_metrics(
                                       cv_pred, val_gold, gold_threshold=GOLD_THRESHOLD,
                                       pred_threshold=cls_thr).to_dict(),
                                   "note": "5-fold CV; threshold tuned via Youden's J on these "
                                           "same out-of-sample CV predictions"},
                    "test": {"split": "test",
                             "correlation": correlation_metrics(test_pred, test_gold).to_dict(),
                             "classification": classification_metrics(
                                 test_pred, test_gold, gold_threshold=GOLD_THRESHOLD,
                                 pred_threshold=cls_thr).to_dict(),
                             "note": "trained on all validation; threshold carried over from "
                                     "validation CV, not re-tuned on test"},
                },
            }
            if mdl == "linear":
                entry["coef_"] = {f: round(c, 4) for f, c in zip(feats, np.ravel(lc._model.coef_))}
                entry["intercept_"] = round(float(lc._model.intercept_), 4)
            results.append(entry)
            print(f"  {name:<38} "
                  f"val rho={entry['splits']['validation']['correlation']['spearman']:+.3f}  "
                  f"test rho={entry['splits']['test']['correlation']['spearman']:+.3f}")

    # -- diagnosis --
    isolated = set(ei.missing)
    docs_with_statutes = {d for d, f in nsi.features.items() if f.act_ids or f.section_keys}
    diag = {
        s: diagnose_split(s, p, isolated_case_ids=isolated,
                          docs_with_statutes=docs_with_statutes).to_dict()
        for s, p in (("validation", val), ("test", test))
    }
    # correlation on the test SUBSET where neither case is isolated
    good_test = [p for p in test if p.doc_id_1 not in isolated and p.doc_id_2 not in isolated]
    nsi.method = "blend"
    hm_w = HybridSimilarityModel(ti, nsi, WeightedSumCombiner(0.6, 0.4))
    diag["test_subset_no_isolated_case"] = {
        "n_pairs": len(good_test),
        "hybrid_weighted_0.6_0.4": correlation_metrics(
            [hm_w.score(p.doc_id_1, p.doc_id_2) for p in good_test],
            [p.score for p in good_test],
        ).to_dict(),
        "text_only_tfidf": correlation_metrics(
            [ti.pair_similarity(p.doc_id_1, p.doc_id_2) for p in good_test],
            [p.score for p in good_test],
        ).to_dict(),
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pairs": {"validation": len(val), "test": len(test)},
        "protocol": "text/network/weighted: direct correlation; learned: 5-fold CV on "
                    "validation, trained-on-all-validation for test.",
        "results": results,
        "diagnosis": diag,
        "reliability": reliability_report,
    }
    out = data_processed_dir() / "evaluation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_tables(report)
    print(f"\n[ OK ] wrote {out.relative_to(_REPO_ROOT).as_posix()}")
    return 0


def _blend(nsi, a, b):
    prev, nsi.method = nsi.method, "blend"
    try:
        return nsi.similarity(a, b)
    finally:
        nsi.method = prev


def _print_tables(report: dict) -> None:
    print("\n" + "=" * 78)
    print("CORRELATION WITH EXPERT SCORES  (Spearman rho / Pearson r / MSE)")
    print("=" * 78)
    print(f"{'config':<38} | {'validation':>22} | {'test':>22}")
    print("-" * 88)
    for r in report["results"]:
        v = r["splits"]["validation"]["correlation"]
        t = r["splits"]["test"]["correlation"]
        print(f"{r['config']:<38} | "
              f"rho{v['spearman']:+.2f} r{v['pearson']:+.2f} m{v['mse']:.3f} | "
              f"rho{t['spearman']:+.2f} r{t['pearson']:+.2f} m{t['mse']:.3f}")

    print("\nRANKING SANITY (does the model rank a query's high-scored partner near the top?)")
    print(f"{'config':<38} | {'val MRR / hit@10':>18} | {'test MRR / hit@10':>18}")
    print("-" * 82)
    for r in report["results"]:
        if r["kind"] == "learned":
            continue
        v = r["splits"]["validation"]["ranking"]
        t = r["splits"]["test"]["ranking"]
        print(f"{r['config']:<38} | "
              f"{v['mrr']:.3f} / {v['hit_at'].get('10', float('nan')):.2f}    | "
              f"{t['mrr']:.3f} / {t['hit_at'].get('10', float('nan')):.2f}")

    print("\nDIAGNOSIS")
    for split in ("validation", "test"):
        d = report["diagnosis"][split]
        print(f"  {split}: n={d['n_pairs']}  scores [{d['score_min']:.2f},{d['score_max']:.2f}] "
              f"mean {d['score_mean']:.2f}  distinct {d['distinct_scores']}  "
              f"pairs w/ isolated case: {d['pairs_with_isolated_case']}  "
              f"both have statutes: {d['pairs_both_have_statutes']}")
        for n in d["notes"]:
            print(f"     - {n}")
    sub = report["diagnosis"]["test_subset_no_isolated_case"]
    print(f"  test subset with NO isolated case (n={sub['n_pairs']}): "
          f"hybrid rho={sub['hybrid_weighted_0.6_0.4']['spearman']:+.3f}, "
          f"text-only rho={sub['text_only_tfidf']['spearman']:+.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
