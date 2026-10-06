"""parameter_ablation.py - does supplying a case parameter raise the accuracy of the
true match?  Measured on the real expert-scored pairs, with the weight of the
detail bonus chosen on VALIDATION and the result reported on TEST only.

For each expert pair (A, B) and each generalized parameter p that A states:

  missing   A with every sentence that states p removed    -> accuracy of B
  added     "missing" + p typed back in (augment_text)      -> accuracy of B
  weighted  "added" + a bonus lambda * [B also states p]    -> accuracy of B

"Accuracy of B" is the calibrated reliability the deployed model assigns to B as a
match for A (the number the UI shows per result).  gain = mean(variant - missing).
lambda is one value from LAMBDAS, picked to maximise the gain on validation pairs.
The test column uses that lambda unchanged.

    python scripts/parameter_ablation.py

Writes data/processed/parameter_ablation.json.  Nothing is hard-coded; re-running
regenerates it.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

import numpy as np  # noqa: E402

from app.ml.case_category import LABELS, classify  # noqa: E402
from app.ml.case_parameters import augment_text, match_details, specs_for  # noqa: E402
from app.ml.hybrid_similarity import HybridFeatures  # noqa: E402
from app.ml.network_similarity import extract_doc_features  # noqa: E402
from app.services.recommender import RecommenderService  # noqa: E402
from app.utils.paths import data_processed_dir  # noqa: E402

SIMILAR_MIN, DISSIMILAR_MAX = 0.5, 0.3
LAMBDAS = (0.02, 0.05, 0.10)
MIN_EFFECT = 0.001
_SENT = re.compile(r"(?<=[.;:!?])\s+|\n+")


def _remove_sentences(text: str, pattern: re.Pattern) -> str:
    return " ".join(s for s in _SENT.split(text) if s and not pattern.search(s))


def main() -> int:
    svc = RecommenderService.load_default()
    eng, corpus = svc.engine, svc.corpus
    ti, nsi, comb = eng.model.text_index, eng.model.network_index, eng.model.combiner
    cal = eng.reliability
    if cal is None:
        print("no reliability calibration found - run scripts/evaluate.py first")
        return 1

    def accuracy_of(text, target, cache, detail=None, bonus=0.0):
        key = hash(text)
        if key not in cache:
            qfeat = extract_doc_features(text, title_index=eng.title_index, hierarchy=eng.hierarchy, year=None)
            cache[key] = (qfeat, ti.query_scores(text))
        qfeat, tscores = cache[key]
        stat, cite = nsi.query_overlaps(qfeat, target)
        net = nsi.query_similarity(qfeat, target, embedding_score=0.0)
        hyb = comb.score(HybridFeatures(text_score=tscores.get(target, 0.0), network_score=net,
                                        embedding_score=0.0, statute_overlap=stat,
                                        citation_overlap=cite, cocitation_count=0))
        if detail and bonus:
            hyb = min(1.0, hyb + bonus * sum(match_details(corpus.text(target), detail).values()) / len(detail))
        return float(cal.reliability(hyb))

    pairs = [(sp, p) for sp in ("validation", "test") for p in corpus.pairs_by_split[sp]]
    records: list[dict] = []
    cache: dict = {}
    t0 = time.time()
    for i, (split, p) in enumerate(pairs):
        if p.score >= SIMILAR_MIN:
            bucket = "similar"
        elif p.score <= DISSIMILAR_MAX:
            bucket = "dissimilar"
        else:
            continue
        for a_id, b_id in ((p.doc_id_1, p.doc_id_2), (p.doc_id_2, p.doc_id_1)):
            if a_id not in corpus or b_id not in corpus:
                continue
            a_text = corpus.text(a_id)
            cat = classify(a_text)
            for _, spec in specs_for(cat):
                m = spec.pattern.search(a_text)
                if not m:
                    continue
                missing_text = _remove_sentences(a_text, spec.pattern)
                if len(missing_text) < 100:
                    continue
                value = {spec.id: m.group(0)}
                rec = {"split": split, "bucket": bucket, "category": cat,
                       "id": spec.id, "label": spec.label,
                       "missing": accuracy_of(missing_text, b_id, cache),
                       "added": accuracy_of(augment_text(missing_text, value, cat), b_id, cache)}
                for lam in LAMBDAS:
                    rec[f"w{lam}"] = accuracy_of(missing_text, b_id, cache, detail=value, bonus=lam)
                records.append(rec)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs  ({time.time() - t0:.0f}s)", flush=True)
        cache.clear()

    def gains(recs, variant):
        return np.array([r[variant] - r["missing"] for r in recs]) if recs else np.array([])

    def pick_lambda():
        val_sim = [r for r in records if r["split"] == "validation" and r["bucket"] == "similar"]
        best = max(LAMBDAS, key=lambda l: gains(val_sim, f"w{l}").mean() if val_sim else -1)
        return best

    chosen = pick_lambda()

    def summarize(recs, variant):
        g = gains(recs, variant)
        n = len(g)
        if n == 0:
            return {"n": 0}
        se = float(g.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        return {"n": n, "gain": round(float(g.mean()), 5), "gain_se": round(se, 5),
                "share_improved": round(float((g > 1e-9).mean()), 3)}

    rows: dict[str, dict] = {}
    for r in records:
        key = f"{r['category'] or 'any'}:{r['id']}"
        rows.setdefault(key, {"category": r["category"], "id": r["id"], "label": r["label"], "recs": []})["recs"].append(r)

    out_rows = []
    for key, row in rows.items():
        rs = row["recs"]
        entry = {"key": key, "category": row["category"],
                 "category_label": LABELS.get(row["category"]) if row["category"] else None,
                 "id": row["id"], "label": row["label"]}
        for name, variant in (("added", "added"), ("weighted", f"w{chosen}")):
            for split in ("validation", "test"):
                sim = [r for r in rs if r["split"] == split and r["bucket"] == "similar"]
                dis = [r for r in rs if r["split"] == split and r["bucket"] == "dissimilar"]
                entry[f"{name}_{split}_similar"] = summarize(sim, variant)
                entry[f"{name}_{split}_dissimilar"] = summarize(dis, variant)
        t = entry["added_test_similar"]
        entry["supported_added_test"] = bool(t.get("n", 0) >= 5 and t["gain"] >= MIN_EFFECT
                                             and t["gain"] - 1.96 * t["gain_se"] > 0)
        tw = entry["weighted_test_similar"]
        entry["supported_weighted_test"] = bool(tw.get("n", 0) >= 5 and tw["gain"] >= MIN_EFFECT
                                                and tw["gain"] - 1.96 * tw["gain_se"] > 0)
        out_rows.append(entry)
    out_rows.sort(key=lambda x: -(x["added_test_similar"].get("gain") or 0))

    all_test = [r for r in records if r["split"] == "test" and r["bucket"] == "similar"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": ("Per expert pair (A,B) and each parameter A states: accuracy (calibrated "
                     "reliability) of B as a match for A with the parameter removed ('missing'), typed "
                     "back in ('added'), and typed back in with a bonus lambda for candidates that also "
                     "state it ('weighted'). lambda is chosen by maximum mean gain on VALIDATION similar "
                     "pairs; all TEST figures use that lambda unchanged. Gains are in accuracy units "
                     "(0.01 = 1 point)."),
        "lambda_grid": list(LAMBDAS), "chosen_lambda": chosen,
        "similar_min": SIMILAR_MIN, "dissimilar_max": DISSIMILAR_MAX,
        "overall_test_similar": {"added": summarize(all_test, "added"), "weighted": summarize(all_test, f"w{chosen}")},
        "parameters": out_rows,
    }
    out = data_processed_dir() / "parameter_ablation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    o = report["overall_test_similar"]
    print(f"\nchosen lambda (validation): {chosen}")
    print(f"TEST, expert-similar pairs, all parameters pooled:")
    print(f"  typed in only       gain {o['added']['gain']*100:+.3f} pts  (n={o['added']['n']})")
    print(f"  typed in + weighted gain {o['weighted']['gain']*100:+.3f} pts  (n={o['weighted']['n']})")
    print(f"  supported (test, 95% CI > 0): typed={sum(r['supported_added_test'] for r in out_rows)}  "
          f"weighted={sum(r['supported_weighted_test'] for r in out_rows)}  of {len(out_rows)}")
    print(f"[ OK ] wrote {out.relative_to(_REPO_ROOT).as_posix()}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
