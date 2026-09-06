# Scaling to a real Indian judgment corpus

The demo runs on the **296 base-paper documents**. This document is the plan and
the how-to for scaling to tens of thousands (or lakhs) of judgments.

## Design: two corpora in one database

| `corpus` value | source | has expert pairs? | used for |
|---|---|---|---|
| `base` | `data/raw/` (the 296 paper docs) | **yes** (190 pairs) | evaluation, and search/recommend |
| anything else (`ildc`, `local`, …) | bulk ingest | no | search / recommend only |

Set **`CORPUS_SOURCE=db`** and every script + the API load *all* `Case` rows
(base + bulk). Expert-score evaluation (`scripts/evaluate.py`) always runs on the
190 base pairs, which are now a labelled subset of a much larger unlabelled corpus.

```bash
python scripts/init_db.py --drop            # base 296 → DB (with text)
python scripts/ingest_bulk.py --source ...  # add a bulk corpus (below)
CORPUS_SOURCE=db python scripts/build_all.py         # rebuild over everything
CORPUS_SOURCE=db uvicorn app.main:app --app-dir backend
```

Verified end-to-end at 415 docs (296 base + a synthetic bulk set): the full
pipeline builds and the API serves recommendations over the merged corpus;
evaluation numbers stay within run-to-run noise of the 296-doc baseline.

## Where to get the data

| Source | What | Access |
|---|---|---|
| **ILDC** (Malik et al. 2021, IIT-KGP) | 34,816 Supreme Court judgments | free · HuggingFace / GitHub — **best clean start** |
| **Law-AI / OpenNyAI** corpora | IL-TUR benchmark, rhetorical-role, summarization, NER | free · HuggingFace (`Exploration-Lab`, `L-NLProc`, `opennyai`) |
| **Supreme Court of India** portal (main.sci.gov.in, judgments.ecourts.gov.in) | ~50k+ SC judgment PDFs | free public-domain text; scraping is manual / CAPTCHA'd |
| **eCourts** High Court judgments | millions of HC judgments | free text, heavy CAPTCHA, no bulk API |
| **Indian Kanoon API** (indiankanoon.org/api) | millions of docs **with resolved citations** | paid per-document; ToS limits bulk |
| Manupatra / SCC Online | comprehensive, editorial | commercial licence, no export |

Judgments are public-domain government works (§52(1)(q), Copyright Act 1957); the
aggregators' *access terms* are the constraint. Clean research route:
**ILDC + OpenNyAI + the SCI portal**. For real case→case citation edges (what the
demo lacks — see `EVALUATION.md`), the Indian Kanoon API gives resolved citations
instead of fuzzy title matching.

## PostgreSQL (P3)

```bash
docker compose up -d db
pip install -r backend/requirements-pg.txt
export DATABASE_URL=postgresql+psycopg2://lcs:lcs@localhost:5432/lcs
alembic upgrade head                      # create the schema
CORPUS_SOURCE=db python scripts/init_db.py --drop   # load base 296
python scripts/ingest_bulk.py --source hf --hf-dataset ... --corpus ildc
CORPUS_SOURCE=db python scripts/build_all.py
CORPUS_SOURCE=db uvicorn app.main:app --app-dir backend
```

After changing a model, regenerate the migration: `alembic revision --autogenerate -m "..."`.
SQLite dev still works with zero setup (`create_all`).

## Ingesting

```bash
# ILDC via HuggingFace
pip install datasets
python scripts/ingest_bulk.py --source hf --hf-dataset Exploration-Lab/IL-TUR \
    --hf-config cjpe --hf-split train --text-field text --id-field id \
    --corpus ildc --limit 35000

# a folder of PDFs / .txt files
python scripts/ingest_bulk.py --source dir --path ./my_judgments --corpus local

# a JSONL file (one object per line)
python scripts/ingest_bulk.py --source jsonl --path judgments.jsonl \
    --text-field judgment --id-field case_id --year-field year --corpus mine
```

Dedups on `doc_id` and on text SHA-1. Batched inserts.

## Extending the statute hierarchy (P6)

```bash
# acts.csv columns: act_id,canonical,year,alias,container_id,container_title,section,section_title
python scripts/build_statute_hierarchy.py --csv acts.csv --out data/processed/hierarchy.json \
    --merge-into backend/app/ml/resources/statute_hierarchy.json
STATUTE_HIERARCHY_PATH=data/processed/hierarchy.json python scripts/build_all.py
```

The built-in 41-Act sample stays; your file is merged on top. Feed it an India Code
parse or a section-list dump.

## Remaining engineering (phased)

| Phase | Change | Why |
|---|---|---|
| **P1 ✅** | bulk ingest + DB-backed corpus (`CORPUS_SOURCE=db`) | get real data in |
| **P2 ✅** | ANN index (`app.ml.vector_index`, `scripts/build_index.py`, `data/indexes/`) — FAISS-HNSW if `faiss-cpu` is installed, exact numpy fallback otherwise. The recommendation engine ANN-retrieves a candidate pool (text ∪ network, default 300) and hybrid-re-ranks only that pool. | brute-force = O(N) hybrid evaluations per query (≈10 s at 35k); ANN keeps it ≈100 ms. Verified top-1 agreement 40/40 vs brute force on the demo corpus. |
| **P3 ✅** | `docker-compose.yml` (Postgres 16), `backend/requirements-pg.txt` (`psycopg2-binary`, `alembic`), Alembic wired to the app settings + models (`alembic upgrade head`), initial migration checked in. `make_engine` handles any URL. | SQLite won't hold the text or the write load; the switch is now a `DATABASE_URL` change + one migrate |
| **P4 ✅** | citation-match **blocking** — a `token -> doc_ids` inverted index over distinctive title tokens; a query is only fuzzy-compared against titles that share a token, and a resolution must share a token outside a place-name / common-surname stop set. | the old fuzzy match scanned every title per mention (O(mentions × corpus)); blocking makes it roughly linear in mentions |
| **P5** | graph backend: NetworkX → `scipy.sparse` / `igraph`, walks via `pecanpy` | ~35k cases + statute layer ≈ 200k nodes; NetworkX walks are the bottleneck |
| **P6 ✅** | curated sample expanded to **41 Acts** (added NDPS, PMLA, IBC, POCSO, DV Act, UAPA, SC/ST Act, Arbitration & Conciliation, Companies 2013, Consumer Protection, RTI, RTE, Registration, Specific Relief, Hindu Marriage, …) with section titles + `_SECTION_CEILINGS`. `load_hierarchy()` merges an external `STATUTE_HIERARCHY_PATH` JSON over the sample; `scripts/build_statute_hierarchy.py` builds that JSON from a flat CSV (India Code parse / section-list scrape). | 12,814-Act tree unpublished; this covers the statutes that actually recur in SC/HC litigation and is now extensible without code |
| **P7 ✅ (plumbing)** | `generate_embeddings.py --method sbert --sbert-model law-ai/InLegalBERT` flows through to `SbertEmbedder`. Actually running it needs `requirements-ml.txt` + a GPU + the model download. | better semantic signal than TF-IDF / generic SBERT |

## Hardware

- **Build** (embeddings, graph, index): ~64 GB RAM, ~500 GB disk; 1 GPU if using
  InLegalBERT (CPU ≈ a day for 35k long docs; GPU ≈ 1–2 h).
- **Serve**: modest — a FAISS index of 35k × 768d ≈ 100 MB, fits in RAM; sub-second
  queries once P2 lands.
