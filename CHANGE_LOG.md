# Change Log

## Pre-Step-1 (project setup, done before development started)
- Analyzed base paper (Bhattacharya et al., 2022) and confirmed no public model code exists,
  only the dataset repo (Law-AI/document-similarity).
- Analyzed project PPT for architecture/requirements.
- Searched GitHub for prior art (none directly reusable — see CLAUDE.md §5).
- Created project directory structure (backend/, frontend/, data/, scripts/, tests/, notebooks/).
- Copied real dataset into data/raw/validation/ and data/raw/test/ (143+153 docs, 100+90
  expert-scored pairs — verified counts, matches paper).

## Step 1 — Environment Setup
- Selected CPython 3.12.5 as the target interpreter (system default 3.14 is too new for
  gensim / numpy<2); created `.venv/` from it.
- Rewrote `backend/requirements.txt` as a coherent, gensim-4.3.3-compatible pinned set
  (numpy<2, scipy<1.14) covering Steps 3–18; added pydantic-settings, python-dotenv,
  python-multipart, httpx.
- Added `backend/requirements-ml.txt` (sentence-transformers, faiss-cpu) — our own
  enhancements, deferred to the Text Similarity / Build Index steps.
- Installed core deps into `.venv/`; captured exact versions in
  `backend/requirements.lock.txt` (pip freeze).
- Added `backend/.env.example` config template (SQLite URL, data dirs, text-sim method,
  hybrid α/β, Top-K, CORS).
- Added `scripts/check_env.py` — verifies interpreter range, core imports, numpy<2, and
  the live data/raw/ counts (143/153 docs, 100/90 pairs). Run: passes, exit 0.
- Added `SETUP.md`; extended `.gitignore` (data/graphs, *.faiss, *.index, .DS_Store);
  added missing `.gitkeep` files.
- Node 25.8.2 / npm 11.11.1 present — frontend scaffold deferred to Step 19.

## Step 2 — Repository Setup
- `git init` + first commit (`4ae1bd8`) on `master`, using the dev's git identity.
- Added `README.md` (project overview, base-paper attribution, dataset table, layout,
  scope statement), `frontend/README.md` (scaffold deferred to Step 19).
- Added `pyproject.toml`: pytest config (`testpaths=tests`, `pythonpath=["backend"]` so
  `import app.ml...` works everywhere) + optional ruff lint config.
- Filled package `__init__.py` files (`app.__version__ = "0.1.0"`, docstrings noting the
  ml/ ↔ routers/ decoupling rule).
- Added `tests/` scaffold: `__init__.py`, `conftest.py` (repo_root / raw_data_dir
  fixtures), `test_smoke.py` — 8 tests, all passing (package imports, app.ml imports
  without FastAPI, numpy<2, live dataset counts 143/153 docs & 100/90 pairs).
- Added `.editorconfig` and `.gitattributes` (`data/raw/** -text` keeps the judgment
  text byte-exact).
- `.gitkeep` placeholders for empty tree dirs (data/*, notebooks/, frontend/src/*).

## Step 3 — Dataset Setup
- `backend/app/utils/paths.py` — env-driven path resolution (DATA_RAW_DIR etc., repo-root
  relative), so a bigger corpus is swapped via env, not code edits.
- `backend/app/utils/dataset.py` — corpus loader, no FastAPI/ML deps:
  `CaseDocument` (lazy cached text/title, sha1), `ExpertPair`, `Corpus` (accessors +
  `check_integrity()`), `load_corpus()`, `load_expert_pairs()`, `parse_year()`,
  `extract_title()`.
- Verified against real data (recorded in the module docstring):
  * one byte-identical filename (`2011_I_16`) in both splits ⇒ **295 unique documents**,
    that doc belongs to both splits.
  * 100% referential integrity — every doc_id in each split's `similarity_scores.csv`
    resolves to a document in that same split.
  * 235/295 docs have a party-style title line; 60 have `title=None` (surfaced, not hidden).
- `scripts/ingest_data.py` — loads + validates + prints report (per-split counts, score
  stats, docs-by-decade, title coverage, length stats) and writes
  `data/processed/corpus_manifest.json` (per-doc: splits, year, title, char_len, sha1).
  `--check` validates only. Ran clean, exit 0.
- Score stats (live from data, NOT an evaluation result): validation 100 pairs
  min 0.0 / max 0.97 / mean 0.466; test 90 pairs min 0.15 / max 0.90 / mean 0.507.
  Corpus: 1953–2018, 9.58M chars total, median doc 23.9k chars.
- `tests/test_dataset.py` — 12 tests (parse_year, extract_title, 295 uniques, overlap
  splits, per-split counts, referential integrity, title coverage). Total suite: 20 pass.
- `data/processed/corpus_manifest.json` is git-ignored (regenerable via the script).

## Step 4 — Data Preprocessing
- `backend/app/ml/preprocessing.py` — deterministic text pipeline, no FastAPI/model deps.
  Produces three representations per judgment:
  * `clean_text` — NFKC + quote/dash folding, de-hyphenation, page-marker strip,
    whitespace collapse; cased & punctuated. Idempotent. For SBERT / display.
  * `sentences` — abbreviation-aware rule-based segmentation (protects "v.", "No.",
    "u/art.", "C.C.", single-letter initials, ~80 legal abbrevs); no model download.
  * `tokens` — lowercased alphabetic tokens, gensim English stopwords removed. BoW input
    for TF-IDF / Doc2Vec (base-paper methods).
- Clearly-labelled OUR options, all OFF by default so the Doc2Vec/TF-IDF baseline stays
  faithful: `remove_legal_boilerplate` (drops "court", "appellant", …), `stem` (gensim
  Porter), `min_token_len`. `PreprocessConfig` is frozen and logged with every run.
- Header detection (court/date/cause-title/"delivered by" → body split) is conservative
  and only affects `tokens`; `clean_text` keeps the full document.
- `scripts/preprocess.py` — runs the pipeline over all 295 docs, writes
  `data/processed/{clean,sentences,tokens}/<doc_id>.*` + `preprocess_report.json`
  (config + aggregate/per-doc stats). Flags: `--limit`, `--stem`, `--legal-boilerplate`,
  `--no-stopwords`, `--keep-header`.
- Ran full corpus (default config): headers detected 295/295; corpus vocab 26,477;
  tokens/doc median 1,689 / mean 2,360 / max 15,597; sentences/doc median 122.
  Top tokens: court, act, case, state, appellant, high, order, said, section.
  (Descriptive counts, not evaluation results.)
- `tests/test_preprocessing.py` — 16 tests (idempotence, de-hyphenation, abbreviation
  non-splitting, stopword/stem/boilerplate opt-in, real-doc end-to-end). Suite: 36 pass.
- All `data/processed/` outputs git-ignored (regenerable).

## Step 5 — PDF/Text Extraction
- `backend/app/ml/text_extraction.py` — the upload-path extractor (PyMuPDF), no FastAPI dep:
  * `extract_text_from_txt` — encoding fallback chain (utf-8-sig → utf-8 → cp1252 → latin-1,
    then replace); never raises on decodable bytes, raises on empty.
  * `extract_text_from_pdf` — text-layer extraction page by page; detects password-protected
    PDFs (rejects, empty-password opened transparently); flags scanned/image PDFs as
    `needs_ocr=True` with a warning (< 30 chars/page heuristic); optional repeated
    header/footer stripping (`drop_repeated_lines`, on by default).
  * `extract_text` — dispatch by extension (path) or `%PDF-` magic (bytes).
  * `extract_and_preprocess` — extraction → the SAME `app.ml.preprocessing` pipeline, so an
    uploaded case and a corpus case are represented identically.
  * `ExtractedDocument` dataclass: source_name/type, raw_text, page_count, char_count,
    metadata, warnings, needs_ocr.
- Limitation surfaced, not hidden: no OCR engine is a dependency; image-only PDFs are
  detected and reported, not silently returned empty.
- `scripts/extract_text.py` — CLI: `extract_text.py <file>` [`--preprocess` `--out` `--stdout`],
  plus `--selftest` which round-trips corpus doc `1953_24` through a generated 42-page PDF.
  Selftest result: 99.8% word overlap with the corpus source, `needs_ocr=False` — passes.
- `tests/test_text_extraction.py` — 11 tests (txt utf-8 / latin-1 / empty, pdf round-trip,
  repeated-header stripping, scanned-PDF OCR flag, dispatch by extension & magic bytes,
  unsupported type, extract+preprocess). Suite: 47 pass.

## Step 6 — Citation Extraction (case → case)
- `backend/app/ml/citation_extraction.py` — regex extraction + rapidfuzz resolution, no
  FastAPI/model deps:
  * `extract_case_citations()` — token-walk outward from each " v "/" vs "/" versus ",
    strips lead-in words ("in", "see", "per", "Lord X in", judge names), stops at sentence
    boundaries and citation debris, captures a trailing reporter year ("[1953] S.C.R. 589").
  * `normalize_party()` / `normalize_citation()` — lowercase, drop "the/of/M/s/and others",
    fold "&"→"and", unify Ltd/Co.
  * `CorpusTitleIndex.from_corpus()` — fuzzy index over the 234/295 docs with a
    party-style cause-title.
  * `resolve()` — `fuzz.token_set_ratio` ≥ min_score, plus guards: ≥2 distinctive query
    tokens (one ≥4 chars), `token_sort_ratio` ≥ 60, ≥2 shared distinctive tokens,
    **temporal guard** (a judgment cannot cite a later case), optional reporter-year window,
    self-exclusion.
  * `build_citation_edges()` — directed citing→cited edges for the whole corpus (feeds
    PCNet in Step 9); dedups, keeps best score + mention count.
- `scripts/extract_citations.py` — writes `data/processed/citations/<doc_id>.json` +
  `citation_report.json`. Flags: `--min-score` (default 90), `--limit`.
- Ran on the real corpus (min_score 90): **1,857 raw mentions → 24 resolved case→case
  edges**, resolution rate **1.4%**, 23/295 docs cite ≥1 corpus case, 20 distinct cited.
  Threshold sensitivity: 88→34 edges, 90→24, 92→19.
- This sparsity is the documented §3 limitation, stated plainly in the module docstring and
  the report: most cited cases are English/foreign or lower-court decisions absent from a
  295-doc demo corpus. The network signal will lean on the statute graph (Steps 7, 9).
  Precision is prioritised over recall (a wrong edge = a false similarity signal).
- `tests/test_citation_extraction.py` — 16 tests (extraction, lead-in stripping, sentence
  boundary, reporter-year capture, normalisation, synthetic-index resolution incl. temporal
  guard / self-exclusion / debris rejection, real-corpus edge build). Suite: 63 pass.

## Step 7 — Statute Extraction
- `backend/app/ml/resources/statute_hierarchy.json` — HAND-CURATED SAMPLE hierarchy
  (Act → Part/Chapter → Section) for 23 frequently-cited Acts (IPC, CrPC, CPC,
  Constitution, Evidence/Contract Act, Income-tax 1922 & 1961, etc.) + a ~55-entry alias
  table. Clearly labelled in `_meta` as NOT the paper's unpublished 12,814-Act tree.
- `backend/app/ml/statute_extraction.py` — regex extraction + Act resolution, no
  FastAPI/model deps:
  * `_STATUTE_RE` — "Section/Article/Rule N(sub)(sub) [and M] of the <Name> Act, YYYY";
    handles interior lowercase words ("Government of India Act"), roman Act numbers
    ("(XI of 1922)"), multi-number refs expanded to one `StatuteReference` each.
  * `StatuteHierarchy.resolve_act()` — alias match with year disambiguation; **any Act not
    in the sample still returns an `ActMatch` with `in_sample_hierarchy=False`** (bare node,
    never dropped/invented).
  * `find_primary_act_definition()` — detects "the X Act, YYYY (hereinafter … 'the Act')".
  * `resolve_document_statutes()` — resolves anaphoric "the Act"/"the said Act"/"the Rules"
    to the doc's primary Act (explicit definition, else most-mentioned substantive Act,
    preferring a real Act over the Constitution/General Clauses Act).
  * `hierarchy_paths()` — Act → Chapter/Part path for curated sections (feeds Hier-SPCNet).
- `scripts/extract_statutes.py` — writes `data/processed/statutes/<doc_id>.json` +
  `statute_report.json`. Ran on the real corpus:
  **12,324 references (10,202 section + 2,122 article)**, 258/295 docs cite ≥1 Act,
  33 docs have an explicit "the Act" definition, **8,205 anaphoric "the Act" resolved** /
  473 unresolved, **297 distinct Acts (23 curated + 274 bare Act nodes)**,
  561 section→hierarchy paths. (Descriptive counts, not evaluation results.)
- `tests/test_statute_extraction.py` — 14 tests (alias/year/abbrev resolution, bare-node
  fallback, §3 example "Section 66(1) … (XI of 1922)", article→Constitution default,
  anaphora resolution, multi-number expansion, hierarchy path, real corpus). Suite: 77 pass.

## Step 8 — Text Similarity
- `backend/app/ml/text_similarity.py` — three embedders behind one `BaseTextEmbedder`
  interface, each with a `provenance` label, no FastAPI/DB deps:
  * `TfidfEmbedder` — sklearn TF-IDF (1–2 grams, sublinear tf, L2). BASELINE (ours).
  * `Doc2VecEmbedder` — gensim PV-DM, vector_size 200, window 15, 40 epochs, workers=1
    (reproducible). BASE PAPER method; hyper-params are ours (paper published none).
  * `SbertEmbedder` — sentence-transformers `all-MiniLM-L6-v2`, long docs chunked
    (~320 words) and mean-pooled. OUR ENHANCEMENT; lazy import, clear error if
    requirements-ml.txt not installed.
  * `TextSimilarityIndex` — fitted embedder + (n_docs, dim) L2-normalised matrix +
    doc-id lookup; `pair_similarity`, `most_similar`, `query_vector`, `query_text`,
    `save`/`load` (matrix.npy + index.json).
  * `prepare_documents()` reuses Step 4's `data/processed/{tokens,clean}/` when present.
- `scripts/generate_embeddings.py` — `--method tfidf|doc2vec|sbert`, `--evaluate`.
  Writes `data/embeddings/<method>/` (git-ignored, regenerable).
- Ran on the real corpus (`--evaluate`): TEXT-ONLY cosine-vs-expert-score sanity check
  (NOT the hybrid evaluation — Step 22), numbers exactly as computed:
  | method  | build | validation (n=100)        | test (n=90)                |
  |---------|-------|---------------------------|----------------------------|
  | tfidf   | 3.9s  | Pearson +0.664 / Spearman +0.765 | Pearson -0.014 / Spearman -0.023 |
  | doc2vec | 132s  | Pearson +0.654 / Spearman +0.668 | Pearson -0.009 / Spearman -0.023 |
  **FLAG for Step 22:** both embedders correlate well on validation but are near-zero on
  test. Same CSV format both splits; referential integrity verified. Test pairs sit in a
  restricted 0.15–0.90 range and several topically-close pairs (e.g. two Land Acquisition
  cases) are scored *low* while topically-distant pairs score high — consistent with the
  paper's thesis that text alone is insufficient, but the magnitude needs investigating
  (embedding choice, or a genuine property of the 90-pair test set). No number here is
  presented as a result — it is a wiring sanity check.
- `tests/test_text_similarity.py` — 13 tests (l2-normalise, provenance, TF-IDF toy
  clustering + self-sim=1 + self-exclusion, Doc2Vec build/infer, save/load roundtrip,
  real-corpus correlation *direction* on both splits). Suite: 90 pass.

## Step 9 — Network Construction
- `backend/app/ml/network_model.py` — PCNet + Hier-SPCNet as `networkx.DiGraph`, no
  FastAPI/DB deps. Namespaced node ids (`case:`/`act:`/`chap:`/`sec:`), every node has
  `ntype`, every edge an `etype`.
  * `build_pcnet()` — 295 case nodes + citing→cited edges from Step 6 (weight = score/100).
  * `build_hier_spcnet()` — PCNet + statute layer from Step 7: `case→section`, `case→act`,
    `section→chapter/part`, `chapter→act`; non-curated Acts get a bare `act:` node
    (`in_sample_hierarchy=False`), never dropped.
  * `compute_icf()` — Inverse Citation Frequency = IDF over the case corpus
    (`ln((N+1)/(df+1))+1`); stored on statute nodes and copied onto `case→statute` edges as
    `weight`, so ubiquitous provisions (Constitution, IPC) are down-weighted.
  * `graph_summary()`, `to_node_link()` / `from_node_link()` (JSON persistence).
- `scripts/build_graph.py` — writes `data/graphs/{pcnet,hier_spcnet}.json` +
  `graph_report.json`. Flags: `--min-score`, `--no-icf`.
- Built on the real corpus (min_score 90):
  * **PCNet**: 295 nodes, 24 edges, 271 weakly-connected components (largest 4),
    252 isolated cases — the sparse case→case layer of §3.
  * **Hier-SPCNet**: 3,108 nodes (295 case + 297 act + 2,484 section + 32 chapter),
    6,725 edges; **41 components, largest 2,974**; isolated cases 252 → **31**.
  * Most-cited: Constitution (148 cases), IPC (63), Art. 14 (43), IPC 302 (40) — exactly
    the provisions ICF suppresses.
  This shows the paper's core claim structurally: the statute layer is what makes the
  network usable on this corpus; PCNet alone barely connects.
- `tests/test_network_model.py` — 8 tests (PCNet edges/weights, ICF ordering, statute-layer
  node/edge types + hierarchy path, Hier-SPCNet connects PCNet-isolated cases, ICF edge
  weights, non-curated Act node, node-link roundtrip, real corpus). Suite: 98 pass.
- `data/graphs/*` git-ignored (regenerable).
