# Project Progress

**All 26 steps complete.** 185 backend tests + 10 frontend tests pass. Build the
pipeline with `python scripts/build_all.py`, benchmark with `python scripts/evaluate.py`,
demo per `DEMO.md`.

```
[✓]  0. Project Analysis
[✓]  1. Environment Setup        .venv (Python 3.12), pinned deps, check_env.py
[✓]  2. Repository Setup         git, README, pyproject, tests scaffold
[✓]  3. Dataset Setup            app/utils/dataset.py, ingest_data.py, manifest (295 uniques)
[✓]  4. Data Preprocessing       app/ml/preprocessing.py, preprocess.py
[✓]  5. PDF/Text Extraction      app/ml/text_extraction.py, extract_text.py (--selftest 99.8%)
[✓]  6. Citation Extraction      app/ml/citation_extraction.py (24 case→case edges @90; §3 limitation)
[✓]  7. Statute Extraction       app/ml/statute_extraction.py + curated sample hierarchy (12,324 refs)
[✓]  8. Text Similarity          TF-IDF / Doc2Vec / SBERT, generate_embeddings.py
[✓]  9. Network Construction     PCNet + Hier-SPCNet (3,108 nodes / 6,725 edges) + ICF
[✓] 10. Network Embedding        node2vec / metapath2vec via Word2Vec on random walks
[✓] 11. Network Similarity       embedding + ICF statute/citation overlap + breakdown
[✓] 12. Hybrid Similarity        weighted-sum + learned (Ridge/MLP, NN-Map-style); CV val Spearman 0.75
[✓] 13. Recommendation Engine    Top-K for case / uploaded text + filters + rationale
[✓] 14. Explainability           SimilarityExplanation: contribution attribution + labelled overlaps
[✓] 15. Case Comparison          HighlightedText + SideBySide on ComparePage
[✓] 16. Citation Graph           ego_graph service + /api/graph/case + GraphView SVG
[✓] 17. Database                 SQLite + SQLAlchemy 2.0 models + repository; 295 cases loaded
[✓] 18. FastAPI Backend          RecommenderService + 9 routers under /api
[✓] 19. React Frontend           Vite + Tailwind + Router + Recharts; 9 pages
[✓] 20. Search                   /api/search title/semantic/hybrid + facets
[✓] 21. Upload Workflow          POST /api/upload → extract → recommend/compare
[✓] 22. Evaluation               evaluation.py + evaluate.py + /api/evaluate + EVALUATION.md
[✓] 23. Testing                  pytest (185) + vitest (10)
[✓] 24. Optimization             route code-split (618KB → 173KB), no-refit service load
[✓] 25. Documentation            README, ARCHITECTURE.md, SETUP.md, EVALUATION.md, notebook
[✓] 26. Final Demo               DEMO.md, Makefile, run.ps1, pipeline_walkthrough.py
```

## Headline evaluation (real, from `scripts/evaluate.py`)

| | Validation (n=100) | Test (n=90) |
|---|---|---|
| text: TF-IDF | Spearman **+0.76** | ≈ 0 |
| network: metapath2vec (blend) | Spearman +0.48 | ≈ 0 |
| **hybrid: learned linear** | Spearman **+0.75**, MSE **0.055** | ≈ 0 |
| baseline: year proximity / random | +0.29 / −0.13 | ≈ 0 |

Test-split correlation is ≈ 0 for *every* method incl. trivial baselines — a property
of that 90-pair split (range restriction; 31 pairs touch a graph-isolated case; the
clean subset is still ≈ 0). Ranking sanity stays non-zero on test. We do **not** claim
the paper's +11.8 % hybrid gain — their corpus has 53k cases and a dense citation graph;
ours has 295 cases / 24 case→case edges. Full analysis: `EVALUATION.md`.
