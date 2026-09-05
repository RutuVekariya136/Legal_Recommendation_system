# AI-Based Legal Case Similarity and Recommendation System — Project Context for Claude Code

Read this file fully before doing anything. This project is being built incrementally, one
step at a time, with the developer (a final-year Computer Engineering student) verifying and
running each step before the next one starts. Follow the **Working Rules** section exactly.

## 1. What this project is

An AI system that takes a legal case (uploaded PDF/text, or one already indexed) and
recommends the most similar prior cases by combining two signals:

1. **Text similarity** — semantic similarity of case content (TF-IDF baseline, Doc2Vec per
   the base paper, Sentence-BERT as our own enhancement).
2. **Network similarity** — similarity based on citation/statute relationships, using a
   heterogeneous graph (Hier-SPCNet) of cases, Acts, and statute hierarchy (Act → Part →
   Chapter → Topic → Section).

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

## 2. Base paper (research foundation — do not misrepresent)

Bhattacharya, P., Ghosh, K., Pal, A., & Ghosh, S. (2022). "Legal Case Document Similarity:
You Need Both Network and Text." *Information Processing & Management*, 59(6), 103069.
DOI: 10.1016/j.ipm.2022.103069

- The paper proposes **Hier-SPCNet**: augments the Precedent Citation Network (PCNet, cases
  only) with a heterogeneous network of statutes and their hierarchy.
- Their best combination method is called **NN-Map+Conc** (a learned combination of text and
  network similarity), which improved correlation with expert scores by 11.8% over the best
  text-only method and 20.6% over the best network-only method — cite this as *their* result,
  never as ours.
- **The paper released NO model code.** Only the dataset: https://github.com/Law-AI/document-similarity
  This whole pipeline (extraction, PCNet, Hier-SPCNet, embeddings, ICF, hybrid combination) is
  being built from scratch here, following the paper's described methodology as closely as
  possible — this is a genuine implementation, not a clone of an existing repo.

## 3. The dataset (real, already in `data/raw/`, verified — do not re-fabricate these numbers)

| Split | Case documents | Expert-scored pairs (`similarity_scores.csv`) |
|---|---|---|
| validation | 143 `.txt` files | 100 pairs (`doc1,doc2,score`), matches paper exactly |
| test | 153 `.txt` files | 90 pairs, matches paper exactly |

- Real, unredacted Indian Supreme Court judgment text.
- **Statute references are reliably regex-extractable**: patterns like
  `"Section 66(1) of the Indian Income-tax Act (XI of 1922)"` appear cleanly.
- **Case citations are NOT resolvable to document IDs directly** — they appear as party names
  ("Rangachari v. Secretary"), not case numbers. Case→case network edges require **fuzzy
  string matching** (rapidfuzz) against the title line of our own ~296-document corpus, not a
  lookup table. This is a real, documented limitation — don't hide it or pretend citation
  resolution is perfect.
- **No statute hierarchy data is published anywhere.** We ship a small, explicit, clearly
  labeled sample hierarchy for a handful of frequently-cited Acts in this corpus (IPC, Income
  Tax Act, CrPC, etc.) rather than fabricating the full 12,814-Act tree the paper used. Any
  cited Act not in the sample hierarchy still gets an Act-level node, never dropped or invented.
- The original paper's corpus was 53,210 cases / 12,814 Acts — **we do not have this**. Our
  demo corpus is the ~296 documents above. The architecture must be written so a larger corpus
  can be swapped in later (bigger `data/raw/`, re-run the embedding/index scripts) without
  redesigning anything.

## 4. Architecture

```
frontend/   React + Vite + Tailwind + React Router + Axios + Recharts + Cytoscape.js/React Flow
backend/
  app/
    main.py, config.py, database.py
    models/      SQLAlchemy models
    schemas/     Pydantic schemas
    routers/     cases, search, recommend, compare, graph, stats, evaluate, health
    services/    orchestration between routers/ and ml/
    ml/
      text_similarity.py      TF-IDF, Doc2Vec (base paper), SBERT (our enhancement) — pluggable
      citation_extraction.py  regex + fuzzy resolution (see §3)
      network_model.py        PCNet, Hier-SPCNet, Metapath2Vec/Node2Vec, ICF weighting
      hybrid_similarity.py    weighted-sum (configurable α/β) + NN-map variant
      recommendation.py       Top-K ranking, filters, explanation generation
    utils/
data/  raw/ processed/ embeddings/ graphs/ indexes/
scripts/  ingest_data.py, preprocess.py, extract_citations.py, build_graph.py,
          generate_embeddings.py, build_index.py, evaluate.py
notebooks/
tests/
```

Database: **SQLite for local development** (file-based, zero setup), SQLAlchemy models written
so switching to PostgreSQL later is a connection-string change, not a rewrite.

ML logic in `app/ml/` must stay fully decoupled from `app/routers/` — every ML module should be
independently testable and runnable from a plain script/pytest, no FastAPI app required.

## 5. Current state (already done, do not redo)

- Full analysis of the base paper, the dataset repo, and the project PPT — completed.
- Searched GitHub for prior art — nothing implements this hybrid pipeline; a few adjacent repos
  exist for citation-network analysis and graph visualization (idc9/law-net,
  NLeSC/case-law-app, maastrichtlawtech/case-law-explorer) but nothing to directly reuse.
- Directory structure created (see §4).
- Dataset copied into `data/raw/validation/` and `data/raw/test/` (verified counts above).
- **No pipeline code has been written yet.** Everything in `backend/app/ml/`, `scripts/`,
  `backend/app/routers/` etc. is empty/stub. This is genuinely Step 1 onward.

## 6. Working Rules — follow exactly, every time

This developer explicitly wants **one step at a time**, not the whole project dumped at once.

- Work through phases in order: Environment/Repo Setup → Dataset Setup → Preprocessing → PDF
  Extraction → Citation Extraction → Statute Extraction → Text Similarity → Network
  Construction → Network Embedding → Network Similarity → Hybrid Similarity → Recommendation
  Engine → Explainability → Case Comparison → Citation Graph → Database → FastAPI Backend →
  React Frontend → Search → Upload Workflow → Evaluation → Testing → Optimization →
  Documentation → Final Demo.
- **After finishing one step, stop.** Report: what was implemented, files changed, the exact
  command to run it, the expected output, and how to verify it worked. Then wait for
  confirmation (`DONE`/`WORKING`/`NEXT`/`CONTINUE`) before starting the next step.
- Every file you create or modify: give the full path and complete code — no
  `# implement this yourself` placeholders for real functionality.
- **Never fabricate evaluation numbers.** If something hasn't actually been run, say
  "Not executed yet" and give the exact command to run it. Every Precision@K/Recall@K/MRR/
  Pearson/MSE number that ends up in this project must come from actually running code
  against the real 190 pairs in `data/raw/`.
- Keep a running `CHANGE_LOG.md` and update the checklist in `PROGRESS.md` after each step.
- Do not change the project's direction (it stays "text + network → hybrid legal case
  similarity," never a generic chatbot, generic doc-similarity tool, or legal-advice system).
- Clearly distinguish, in comments/docs, what's the base paper's method (Doc2Vec, PCNet,
  Hier-SPCNet, Metapath2Vec, ICF) vs. our own additions (SBERT option, FAISS, the web app,
  explainability UI, evaluation dashboard).

## 7. Master checklist (copy into PROGRESS.md and update as you go)

```
[✓] 0. Project Analysis
[ ] 1. Environment Setup
[~] 2. Repository Setup          (dirs created, dataset in place — this session)
[ ] 3. Dataset Setup             (loader + validation script)
[ ] 4. Data Preprocessing
[ ] 5. PDF/Text Extraction
[ ] 6. Citation Extraction
[ ] 7. Statute Extraction
[ ] 8. Text Similarity
[ ] 9. Network Construction
[ ] 10. Network Embedding
[ ] 11. Network Similarity
[ ] 12. Hybrid Similarity
[ ] 13. Recommendation Engine
[ ] 14. Explainability
[ ] 15. Case Comparison
[ ] 16. Citation Graph
[ ] 17. Database
[ ] 18. FastAPI Backend
[ ] 19. React Frontend
[ ] 20. Search
[ ] 21. Upload Workflow
[ ] 22. Evaluation
[ ] 23. Testing
[ ] 24. Optimization
[ ] 25. Documentation
[ ] 26. Final Demo
```
