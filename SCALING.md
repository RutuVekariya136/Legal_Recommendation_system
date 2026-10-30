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
# ILDC — 38,904 Supreme Court judgments (1947–2020) via HuggingFace
pip install -r backend/requirements-scale.txt
python scripts/init_db.py --drop
python scripts/ingest_bulk.py --source hf --hf-dataset jayadityagandham9/ILDC_35k_COMPLETE \
    --hf-split train --text-field text --id-field id --corpus ildc
# -> ~33k rows after SHA-1 dedup (ILDC has ~15% near-duplicate batch judgments)

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
| **P5 ✅ (partial)** | `metapath_walks` precomputes per-node *{neighbour-type → (nodes, weights)}* once, so each walk step is an O(1) lookup instead of an O(degree) type-scan (metapath walk build ≈ 17 s → 9 s on the demo graph). Above ~100k nodes, swap in `pecanpy` (fast C node2vec) or move the graph to `scipy.sparse` CSR — the `nx.Graph` construction itself becomes the limit there. | NetworkX walks were the bottleneck |
| **P6 ✅** | curated sample expanded to **41 Acts** (added NDPS, PMLA, IBC, POCSO, DV Act, UAPA, SC/ST Act, Arbitration & Conciliation, Companies 2013, Consumer Protection, RTI, RTE, Registration, Specific Relief, Hindu Marriage, …) with section titles + `_SECTION_CEILINGS`. `load_hierarchy()` merges an external `STATUTE_HIERARCHY_PATH` JSON over the sample; `scripts/build_statute_hierarchy.py` builds that JSON from a flat CSV (India Code parse / section-list scrape). | 12,814-Act tree unpublished; this covers the statutes that actually recur in SC/HC litigation and is now extensible without code |
| **P7 ✅ (plumbing)** | `generate_embeddings.py --method sbert --sbert-model law-ai/InLegalBERT` flows through to `SbertEmbedder`. Actually running it needs `requirements-ml.txt` + a GPU + the model download. | better semantic signal than TF-IDF / generic SBERT |

## Hardware

- **Serve** (any scale, once built): ~2–6 GB RAM, 1–2 vCPU, no GPU.
- **Build the ~33k-doc ILDC corpus:**
  - Apple Silicon (M-series) / a desktop CPU: **~30–60 min** for the default
    TF-IDF + metapath2vec build.
  - A U-series laptop (e.g. i7-1255U): ~2–3 h — the random walks over the ~40k-node
    graph are the bottleneck. Works, just slow.
  - RAM peak ≈ 6–8 GB (sparse TF-IDF, the NetworkX graph, the walk corpus).
- **InLegalBERT embeddings (optional):** needs a GPU. On Apple Silicon `torch`
  auto-uses **MPS** and `sentence-transformers` picks it up — no config. ~10–30 min
  for 33k docs. CPU-only ≈ several hours.
- **Lakhs (100k–500k+):** 32–64 GB RAM for the build, PostgreSQL, and the P5
  `pecanpy` / sparse-CSR graph swap.

## Full run-through: ILDC on a fresh machine

```bash
git clone https://github.com/RutuVekariya136/Legal_Recommendation_system.git
cd Legal_Recommendation_system

python3.12 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r backend/requirements.txt
pip install -r backend/requirements-scale.txt              # datasets + faiss-cpu
python scripts/check_env.py

python scripts/init_db.py --drop
python scripts/ingest_bulk.py --source hf \
  --hf-dataset jayadityagandham9/ILDC_35k_COMPLETE \
  --hf-split train --text-field text --id-field id --corpus ildc
#   downloads ~765 MB from HuggingFace, keeps ~33,138 after SHA-1 dedup

export CORPUS_SOURCE=db          # Windows: set CORPUS_SOURCE=db  (or prefix each command)
python scripts/build_all.py      # preprocess -> ... -> embeddings -> indexes

# optional: legal-domain embeddings (uses the Apple-Silicon GPU automatically)
pip install -r backend/requirements-ml.txt
python scripts/generate_embeddings.py --method sbert --sbert-model law-ai/InLegalBERT --no-evaluate

# run
WARM_ON_STARTUP=1 uvicorn app.main:app --app-dir backend          # API  :8000
cd frontend && npm install && npm run dev                         # UI   :5173
```

`numpy<2` / `gensim 4.3.3` / `faiss-cpu` all ship arm64 macOS wheels for Python 3.12.
If `python3.12` is missing on macOS: `brew install python@3.12`.
