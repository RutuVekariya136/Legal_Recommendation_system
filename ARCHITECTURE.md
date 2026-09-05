# Architecture

## 1. Pipeline

```
                      ┌─────────────────── ingestion ───────────────────┐
 upload (PDF/txt) ──▶ Text Extraction ─┐
                                       ├─▶ Preprocessing ─┬─▶ Text Embedding ──▶ Text Similarity ─┐
 corpus case ─────────────────────────┘                  │                                        │
                                                          ├─▶ Citation Extraction ─┐              │
                                                          └─▶ Statute Extraction ──┤              │
                                                                                   ▼              ▼
                                                                          Hier-SPCNet ──▶ Network Embedding ──▶ Network Similarity
                                                                                                                        │
                                                       Text Similarity + Network Similarity ──▶ Hybrid Similarity ◀──────┘
                                                                                                        │
                                                                       Case Ranking ──▶ Top-K Recommendation
                                                                                                        │
                                                       Explanation  +  Case Comparison  +  Citation Graph
```

## 2. What is the base paper's method vs. ours

Bhattacharya, Ghosh, Pal & Ghosh (2022), *"Legal Case Document Similarity: You Need
Both Network and Text"*, IP&M 59(6) 103069. The paper released the **dataset only**;
this whole pipeline is implemented from scratch.

| Component | Base paper | Ours / our choice |
|---|---|---|
| Text embedding | **Doc2Vec (PV-DM)** | TF-IDF baseline · Sentence-BERT option · exact hyper-params (paper published none) |
| Precedent network | **PCNet** (case→case) | fuzzy party-name resolution (paper's corpus had resolvable ids) |
| Statute network | **Hier-SPCNet** (case + Act + Act→Part→Chapter→Section) | small **hand-curated sample hierarchy** (23 Acts); the paper's 12,814-Act tree is unpublished |
| Node embedding | **metapath2vec / node2vec** style | gensim Word2Vec over random walks; our metapath schemas |
| Statute weighting | **ICF** (inverse citation frequency) | same formula, `ln((N+1)/(df+1))+1` |
| Hybrid combination | **NN-Map+Conc** (learned) | weighted-sum + a learned Ridge/MLP combiner *in that spirit* |
| FAISS index, web app, explainability UI, evaluation dashboard | — | entirely ours |

The paper's headline gains (**NN-Map+Conc: +11.8% over best text-only, +20.6% over
best network-only**) are **their** result on their **53,210-case** corpus. Our demo
corpus is **295 cases with 24 resolved case→case edges** — see `EVALUATION.md` for
what we actually measure.

## 3. Code layout

```
backend/app/
  config.py            pydantic-settings (DB url, data dirs, model defaults)
  database.py          SQLAlchemy engine / session / init
  main.py              FastAPI app factory, CORS, lifespan
  models/              Case, UploadedDocument, RecommendationRun/Result
  schemas/models.py    Pydantic request/response models
  routers/             health · stats · cases · search · recommend · compare · upload · graph · evaluate
  services/
    recommender.py     RecommenderService — loads the ML pipeline once, serves routers
    repository.py       DB access helpers
  ml/                  ← pure, no FastAPI/DB import; each module runs from a plain script
    preprocessing.py         clean / sentence-split / tokenise
    text_extraction.py       PDF & txt → text (PyMuPDF)
    citation_extraction.py   "A v B" regex + rapidfuzz resolution to corpus titles
    statute_extraction.py    Section/Article regex + Act resolution + anaphora + hierarchy
    resources/statute_hierarchy.json   the curated sample hierarchy
    text_similarity.py       TfidfEmbedder · Doc2VecEmbedder · SbertEmbedder + TextSimilarityIndex
    network_model.py         build_pcnet / build_hier_spcnet / compute_icf (networkx)
    network_embedding.py     node2vec / metapath2vec walks → Word2Vec → NetworkEmbeddingIndex
    network_similarity.py    embedding + ICF statute/citation overlap + breakdown
    hybrid_similarity.py     WeightedSumCombiner · LearnedCombiner + HybridSimilarityModel
    recommendation.py        RecommendationEngine (Top-K, filters, rationale)
    explainability.py        Explainer → SimilarityExplanation (contribution attribution, labelled overlaps)
    evaluation.py            correlation / ranking / split-diagnosis metrics
  utils/               dataset.py (Corpus loader) · paths.py

frontend/src/
  services/api.js      one axios fn per endpoint
  hooks/useAsync.js
  components/          Layout · ui · RecommendationList · ExplanationPanel (Recharts)
                      · HighlightedText · SideBySide · GraphView (SVG)
  pages/              Home · Search · Case · Recommend · Compare · Graph · Upload · Evaluation · About

scripts/              ingest_data · preprocess · extract_citations · extract_statutes
                      · build_graph · generate_embeddings · generate_network_embeddings
                      · build_network_similarity · build_hybrid · recommend · explain
                      · evaluate · init_db · build_all

data/
  raw/                 the real dataset (committed): validation/ test/ + similarity_scores.csv
  processed/           generated: clean/ sentences/ tokens/ citations/ statutes/ + *_report.json  (git-ignored)
  graphs/              pcnet.json · hier_spcnet.json  (git-ignored)
  embeddings/          tfidf/ doc2vec/ network_*/ network_similarity/ hybrid/  (git-ignored)
```

## 4. Runtime data flow (a recommendation request)

```
GET /api/recommend/case/2005_S_628?k=10
  → routers/recommend.py
  → RecommenderService.recommend_case()
      → RecommendationEngine.recommend_for_case()
          text_score   = TextSimilarityIndex.scores_for(query)          # TF-IDF cosine
          network      = NetworkSimilarityIndex.breakdown(query, cand)  # metapath2vec cosine + ICF overlap
          hybrid       = HybridSimilarityModel.combiner.score(features) # learned Ridge
          rationale    = rule-based string from the breakdown
  → repository.record_run()  (persist to SQLite)
  → RecommendResponse (JSON)
```

The `RecommenderService` is a process-wide singleton (`functools.lru_cache`), built
from `data/embeddings/` on first use (`scripts/build_all.py` pre-builds everything).

## 5. Scaling to a larger corpus

Nothing is hard-coded to 295 documents. To swap in a bigger corpus: drop more
`.txt` files into `data/raw/<split>/documents/`, adjust the expected counts in
`app/utils/dataset.py` (or make them advisory), and re-run `scripts/build_all.py`.
The graph, embeddings and indexes all rebuild from `data/raw/`. SQLite → PostgreSQL
is a `DATABASE_URL` change.
```
