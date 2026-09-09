# AI-Based Legal Case Similarity and Recommendation System

[![tests](https://github.com/2005Hari/Legal_Recommendation_system/actions/workflows/ci.yml/badge.svg)](https://github.com/2005Hari/Legal_Recommendation_system/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/)
[![Backend: FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Frontend: React + Vite](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61dafb.svg)](https://vite.dev/)
[![Not legal advice](https://img.shields.io/badge/⚠-not%20legal%20advice-lightgrey.svg)](#scope)

Given a legal case (uploaded PDF/text, or one already in the corpus), recommend the
most similar prior cases by combining **two signals**:

1. **Text similarity** — semantic similarity of case content
   (TF-IDF baseline · Doc2Vec per the base paper · Sentence-BERT as our enhancement).
2. **Network similarity** — similarity from citation/statute relationships, using a
   heterogeneous **Hier-SPCNet** graph of cases, Acts, and statute hierarchy
   (Act → Part → Chapter → Topic → Section).

These combine into a **hybrid similarity score** for Top-K ranking, with an
explainability breakdown (text / network / hybrid score, common citations, common
statutes), a case-comparison view, and an interactive citation graph.

```
Legal Case → Text Extraction → Preprocessing → Text Embedding → Text Similarity
Legal Case → Citation Extraction → Hier-SPCNet → Network Embedding → Network Similarity
Text Similarity + Network Similarity → Hybrid Similarity → Case Ranking →
Top-K Recommendation → Explanation + Case Comparison + Citation Graph
```

## Base paper

Bhattacharya, P., Ghosh, K., Pal, A., & Ghosh, S. (2022).
*Legal Case Document Similarity: You Need Both Network and Text.*
Information Processing & Management, 59(6), 103069.
[doi:10.1016/j.ipm.2022.103069](https://doi.org/10.1016/j.ipm.2022.103069)

The paper released **only the dataset**
([Law-AI/document-similarity](https://github.com/Law-AI/document-similarity)),
**no model code**. This entire pipeline (extraction, PCNet, Hier-SPCNet, embeddings,
ICF, hybrid combination) is implemented from scratch here, following the paper's
described methodology — it is a genuine implementation, not a port of an existing repo.
The paper's headline gains (NN-Map+Conc: +11.8% over best text-only, +20.6% over best
network-only) are **their** reported results, never claimed as ours.

## Dataset

Real, unredacted Indian Supreme Court judgment text, already in `data/raw/`:

| Split      | Case documents | Expert-scored pairs |
|------------|----------------|---------------------|
| validation | 143 `.txt`     | 100 (`doc1,doc2,score`) |
| test       | 153 `.txt`     | 90                  |

Demo corpus is ~296 documents (the paper used 53,210 cases / 12,814 Acts — not
publicly available). The architecture is written so a larger corpus can be dropped
into `data/raw/` and the embedding/index scripts re-run without redesign.
See `PROJECT_NOTES.md` §3 for documented data limitations (citation resolution needs fuzzy
title matching; statute hierarchy ships as a small labeled sample).

## Repository layout

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full module map and data flow.

```
backend/app/
  config.py database.py main.py
  models/  schemas/  routers/(9)  services/(recommender, repository)
  ml/     preprocessing · text_extraction · citation_extraction · statute_extraction
          text_similarity · network_model · network_embedding · network_similarity
          hybrid_similarity · recommendation · explainability · evaluation
          resources/statute_hierarchy.json        (ml/ has no FastAPI/DB import)
  utils/  dataset.py  paths.py
  requirements.txt / requirements-ml.txt (SBERT/FAISS) / requirements.lock.txt
frontend/src/   services/api.js  hooks/  components/  pages/(9)      React + Vite + Tailwind
data/   raw/ (committed)   processed/ graphs/ embeddings/ indexes/ (generated, git-ignored)
scripts/   ingest_data preprocess extract_citations extract_statutes build_graph
           generate_embeddings generate_network_embeddings build_network_similarity
           build_hybrid recommend explain evaluate init_db build_all
tests/     14 pytest modules (185 tests)   frontend/src/__tests__/  (10 vitest)
```

## Setup

See [SETUP.md](SETUP.md). Short version (repo root, Python **3.10–3.12**):

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
python scripts/check_env.py        # all [ OK ] → ready
```

## Build the pipeline, then run everything

```bash
python scripts/build_all.py         # preprocess → citations → statutes → graph →
                                    # embeddings → network sim → hybrid → init DB  (~2–3 min)
python scripts/evaluate.py          # correlation + ranking vs the 190 expert pairs
uvicorn app.main:app --reload --app-dir backend      # API at :8000  (/docs)
cd frontend && npm install && npm run dev            # UI at :5173  (proxies /api → :8000)
```

Then the demo walkthrough in [DEMO.md](DEMO.md).

### Scripts

| Script | Does | Output |
|---|---|---|
| `check_env.py` | verify interpreter + deps + dataset | — |
| `ingest_data.py` | load & validate the corpus | `data/processed/corpus_manifest.json` |
| `preprocess.py` | clean / sentence-split / tokenise | `data/processed/{clean,sentences,tokens}/` |
| `extract_citations.py` | "A v B" extraction + fuzzy resolution | `data/processed/citations/` |
| `extract_statutes.py` | Section/Act extraction + hierarchy | `data/processed/statutes/` |
| `build_graph.py` | PCNet + Hier-SPCNet | `data/graphs/*.json` |
| `generate_embeddings.py` | TF-IDF / Doc2Vec / SBERT text embeddings | `data/embeddings/<method>/` |
| `generate_network_embeddings.py` | node2vec / metapath2vec | `data/embeddings/network_<method>/` |
| `build_network_similarity.py` | embedding + ICF overlap feature store | `data/embeddings/network_similarity/` |
| `build_hybrid.py` | weighted / learned combiner | `data/embeddings/hybrid/` |
| `build_index.py` | ANN index for candidate retrieval | `data/indexes/{text,network}/` |
| `build_statute_hierarchy.py` | CSV → statute-hierarchy JSON (extends the sample) | a hierarchy JSON |
| `recommend.py` | CLI Top-K for a case / file / text | stdout |
| `explain.py` | CLI pairwise explanation | stdout |
| `evaluate.py` | full benchmark + test-split diagnosis | `data/processed/evaluation_report.json` |
| `init_db.py` | create tables + load 295 cases (with text) | `legal_case_similarity.db` |
| `ingest_bulk.py` | add a large corpus (ILDC / PDFs / JSONL) to the DB | `Case` rows |
| `build_all.py` | all of the above, in order | everything |

## Scaling to a real corpus

The demo runs on 296 documents. **[SCALING.md](SCALING.md)** has the full plan; the
scaling backbone is in place:

- **bulk ingest** — `scripts/ingest_bulk.py --source dir|jsonl|hf` (the `hf` path targets
  ILDC / IL-TUR); documents live in the DB (`Case.text`), `CORPUS_SOURCE=db` merges
  base + bulk everywhere.
- **ANN retrieval** — `app.ml.vector_index` (FAISS-HNSW or exact numpy) + `build_index.py`;
  the recommender scores a candidate pool, not the whole corpus.
- **PostgreSQL** — `docker compose up -d db`, `pip install -r backend/requirements-pg.txt`,
  `alembic upgrade head`; `DATABASE_URL` is the only switch.
- **citation blocking** — an inverted title index keeps fuzzy resolution ~linear in mentions.

```bash
python scripts/ingest_bulk.py --source hf --hf-dataset Exploration-Lab/IL-TUR \
    --hf-config cjpe --hf-split train --corpus ildc --limit 35000
CORPUS_SOURCE=db python scripts/build_all.py
CORPUS_SOURCE=db uvicorn app.main:app --app-dir backend
```

### API (`/api`, full docs at `/docs`)

`GET /health` · `GET /health/ready` · `GET /stats`
`GET /cases` · `GET /cases/{id}` · `GET /cases/{id}/text`
`GET /search?q=&mode=semantic|title|hybrid` · `GET /search/facets`
`GET /recommend/case/{id}?k=&year_min=&year_max=` · `POST /recommend/upload/{id}`
`GET /compare?a=&b=` · `GET /compare/upload/{id}?target=`
`POST /upload` (multipart) · `GET /upload/{id}` · `GET /upload/{id}/text`
`GET /graph/case/{id}` · `GET /evaluate` · `GET /evaluate/summary`

## Status

All 26 steps complete, plus a scaling phase — see [PROGRESS.md](PROGRESS.md) and
[CHANGE_LOG.md](CHANGE_LOG.md). Backend + frontend test suites pass in CI. Evaluation
results and their honest interpretation are in [EVALUATION.md](EVALUATION.md) — every
metric is computed by `scripts/evaluate.py` from the real 190 expert-scored pairs.

## Scope

This project stays "text + network → hybrid legal case similarity." It is **not**
a chatbot, a generic document-similarity tool, or a legal-advice system. Academic
final-year project; **not legal advice**.

## License

[MIT](LICENSE) for the source code. The judgment texts in `data/raw/` are from the
Bhattacharya et al. (2022) dataset ([Law-AI/document-similarity](https://github.com/Law-AI/document-similarity))
— public-domain Indian Supreme Court text; cite the paper if you use it.
