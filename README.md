# AI-Based Legal Case Similarity and Recommendation System

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
See `CLAUDE.md` §3 for documented data limitations (citation resolution needs fuzzy
title matching; statute hierarchy ships as a small labeled sample).

## Repository layout

```
backend/
  app/
    main.py, config.py, database.py      (added in later steps)
    models/  schemas/  routers/  services/
    ml/          text_similarity · citation_extraction · network_model
                 hybrid_similarity · recommendation   (decoupled from routers/)
    utils/
  requirements.txt         core deps (Steps 3–18)
  requirements-ml.txt      optional SBERT / FAISS extras (our enhancements)
  requirements.lock.txt    pinned snapshot
frontend/                  React + Vite + Tailwind (scaffolded at Step 19)
data/   raw/ processed/ embeddings/ graphs/ indexes/
scripts/  ingest_data · preprocess · extract_citations · build_graph ·
          generate_embeddings · build_index · evaluate   (added per step)
notebooks/   tests/
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

## Running the tests

```bash
pytest
```

## Build status

Incremental build, one step at a time — see [PROGRESS.md](PROGRESS.md) and
[CHANGE_LOG.md](CHANGE_LOG.md). Current: Step 2 (Repository Setup).

No evaluation metrics are reported until the pipeline is actually run against the
190 expert-scored pairs in `data/raw/`.

## Scope

This project stays "text + network → hybrid legal case similarity." It is **not**
a chatbot, a generic document-similarity tool, or a legal-advice system. Academic
final-year project; not legal advice.
