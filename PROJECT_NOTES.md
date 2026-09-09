# AI-Based Legal Case Similarity and Recommendation System — Design Notes

Final-year Computer Engineering project. Built incrementally, one step at a time,
with each step verified and run before the next. The running record is in
[CHANGE_LOG.md](CHANGE_LOG.md) and [PROGRESS.md](PROGRESS.md).

## 1. What this project is

A system that takes a legal case (uploaded PDF/text, or one already indexed) and
recommends the most similar prior cases by combining two signals:

1. **Text similarity** — semantic similarity of case content (TF-IDF baseline, Doc2Vec per
   the base paper, Sentence-BERT / InLegalBERT as our own enhancement).
2. **Network similarity** — similarity based on citation/statute relationships, using a
   heterogeneous graph (Hier-SPCNet) of cases, Acts, and statute hierarchy (Act → Part →
   Chapter → Section).

These combine into a **hybrid similarity score** used for Top-K ranking, with an
**explainability breakdown** (text score / network score / hybrid score / common citations /
common statutes), a **case comparison** view, and an **interactive citation graph**.

Full pipeline:
```
Legal Case → Text Extraction → Preprocessing → Text Embedding → Text Similarity
Legal Case → Citation Extraction → Hier-SPCNet → Network Embedding → Network Similarity
Text Similarity + Network Similarity → Hybrid Similarity → Case Ranking →
Top-K Recommendation → Explanation + Case Comparison + Citation Graph
```

## 2. Base paper (research foundation)

Bhattacharya, P., Ghosh, K., Pal, A., & Ghosh, S. (2022). "Legal Case Document Similarity:
You Need Both Network and Text." *Information Processing & Management*, 59(6), 103069.
DOI: 10.1016/j.ipm.2022.103069

- The paper proposes **Hier-SPCNet**: augments the Precedent Citation Network (PCNet, cases
  only) with a heterogeneous network of statutes and their hierarchy.
- Their best combination method is **NN-Map+Conc** (a learned combination of text and
  network similarity), which improved correlation with expert scores by 11.8% over the best
  text-only method and 20.6% over the best network-only method — this is *their* reported
  result on *their* 53,210-case corpus, never claimed as ours.
- **The paper released no model code**, only the dataset:
  https://github.com/Law-AI/document-similarity
  This whole pipeline (extraction, PCNet, Hier-SPCNet, embeddings, ICF, hybrid combination)
  is built from scratch, following the paper's described methodology — a genuine
  implementation, not a port of an existing repo.

## 3. The dataset (`data/raw/`, verified)

| Split | Case documents | Expert-scored pairs (`similarity_scores.csv`) |
|---|---|---|
| validation | 143 `.txt` files | 100 pairs (`doc1,doc2,score`), matches the paper exactly |
| test | 153 `.txt` files | 90 pairs, matches the paper exactly |

- Real, unredacted Indian Supreme Court judgment text (public-domain government works).
- **Statute references are reliably regex-extractable**: patterns like
  `"Section 66(1) of the Indian Income-tax Act (XI of 1922)"` appear cleanly.
- **Case citations are NOT resolvable to document IDs directly** — they appear as party
  names ("Rangachari v. Secretary"), not case numbers. Case→case network edges use **fuzzy
  string matching** (rapidfuzz + a blocking index) against the title line of the
  ~296-document corpus, not a lookup table. This is a documented limitation, not hidden.
- **No statute hierarchy data is published anywhere.** We ship a small, explicit, labelled
  sample hierarchy for frequently-litigated Acts (IPC, Income-tax Act, CrPC, NDPS, PMLA,
  IBC, POCSO, …) rather than fabricating the full 12,814-Act tree the paper used. Any cited
  Act not in the sample still gets an Act-level node, never dropped or invented. The
  hierarchy is extensible via `STATUTE_HIERARCHY_PATH` (see [SCALING.md](SCALING.md)).
- The original paper's corpus was 53,210 cases / 12,814 Acts — we do not have this. The demo
  corpus is the ~296 documents above. The architecture is written so a larger corpus can be
  swapped in (`scripts/ingest_bulk.py`, `CORPUS_SOURCE=db`) without a redesign.

## 4. Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full module map. Summary:

```
frontend/   React + Vite + Tailwind + React Router + Axios + Recharts
backend/app/
  main.py, config.py, database.py
  models/      SQLAlchemy models
  schemas/     Pydantic schemas
  routers/     cases, search, recommend, compare, graph, stats, evaluate, health, upload
  services/    orchestration between routers/ and ml/
  ml/          text_similarity · citation_extraction · statute_extraction · network_model
               network_embedding · network_similarity · hybrid_similarity · recommendation
               explainability · evaluation · vector_index   (fully decoupled from routers/)
  utils/
data/  raw/ processed/ embeddings/ graphs/ indexes/
scripts/  ingest_data · preprocess · extract_citations · extract_statutes · build_graph
          generate_embeddings · generate_network_embeddings · build_network_similarity
          build_hybrid · build_index · recommend · explain · evaluate · init_db
          ingest_bulk · build_statute_hierarchy · build_all
```

- **Database:** SQLite for local development; SQLAlchemy models written so switching to
  PostgreSQL is a `DATABASE_URL` change (Alembic migrations included).
- **`app/ml/` is fully decoupled from `app/routers/`** — every ML module is independently
  testable and runnable from a plain script / pytest, no FastAPI app required.

## 5. Design principles held throughout

- **Never fabricate evaluation numbers.** Every Precision@K / Recall@K / MRR / Pearson / MSE
  figure comes from actually running `scripts/evaluate.py` against the real 190 pairs in
  `data/raw/`. Where the results are weak (the test split), they are reported plainly with a
  diagnosis — see [EVALUATION.md](EVALUATION.md).
- **Scope is fixed:** "text + network → hybrid legal case similarity." Not a chatbot, not a
  generic document-similarity tool, not a legal-advice system. **Not legal advice.**
- **Base-paper method vs. our additions is labelled everywhere** — base paper: Doc2Vec,
  PCNet, Hier-SPCNet, Metapath2Vec, ICF, NN-Map. Ours: TF-IDF baseline, SBERT/InLegalBERT
  option, FAISS retrieval, the web app, the explainability UI, the evaluation dashboard,
  bulk ingestion.
- **Every documented limitation stays visible** — sparse case→case citation resolution, the
  sample statute hierarchy, the test-split correlation.

## 6. Build order (all complete — see PROGRESS.md)

```
0 Analysis · 1 Environment · 2 Repository · 3 Dataset · 4 Preprocessing · 5 PDF/Text
Extraction · 6 Citation Extraction · 7 Statute Extraction · 8 Text Similarity · 9 Network
Construction · 10 Network Embedding · 11 Network Similarity · 12 Hybrid Similarity ·
13 Recommendation Engine · 14 Explainability · 15 Case Comparison · 16 Citation Graph ·
17 Database · 18 FastAPI Backend · 19 React Frontend · 20 Search · 21 Upload Workflow ·
22 Evaluation · 23 Testing · 24 Optimization · 25 Documentation · 26 Final Demo

Phase 2 (scaling): bulk ingestion · ANN retrieval · PostgreSQL · citation blocking ·
statute-hierarchy expansion · faster random walks · legal-domain embedding plumbing
```
