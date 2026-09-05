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

## Step 10 — Network Embedding
- `backend/app/ml/network_embedding.py` — random-walk node embeddings over Hier-SPCNet
  via gensim `Word2Vec` on walk "sentences" (no extra dependency), no FastAPI/DB deps:
  * `node2vec_walks()` — biased walks (p, q), ICF edge weights, precomputed adjacency
    sets for speed; trivial 1-node walks dropped. BASELINE.
  * `metapath_walks()` — walks constrained to type schemas (`case-act-case`,
    `case-section-case`, `case-section-chapter-section-case`,
    `case-section-chapter-act-chapter-section-case`, `case-case`). BASE-PAPER-style.
  * `NetworkEmbedder` — `fit(graph)` → skip-gram Word2Vec (workers=1, seeded);
    `case_matrix(doc_ids)` returns `(n_cases, dim)` + the list of isolated cases with
    no embedding (zero vector).
  * `NetworkEmbeddingIndex` — L2-normalised case matrix, `pair_similarity` (0 if either
    case is isolated), `most_similar`, `save`/`load`.
- `scripts/generate_network_embeddings.py` — `--method`, `--graph`, `--evaluate`; loads
  `data/graphs/hier_spcnet.json` if present. Writes `data/embeddings/network_<method>/`.
- Real corpus, NETWORK-ONLY sanity correlation (NOT the hybrid eval — Step 22), exactly as
  computed:
  | method       | build | validation (n=100)         | test (n=90)                 |
  |--------------|-------|----------------------------|-----------------------------|
  | metapath2vec | 17s   | Pearson +0.346 / Spearman +0.444 | Pearson -0.043 / Spearman -0.052 |
  | node2vec     | ~90s  | Pearson +0.361 / Spearman +0.481 | Pearson -0.093 / Spearman -0.087 |
  Network signal is weaker than text on validation (Spearman ~0.45 vs ~0.75) but positive.
  Test stays ~0 — **and 31 of the 90 test pairs involve a case that is isolated in the
  graph** (zero network similarity by construction), vs only 8/100 on validation. Adds to
  the Step 22 investigation of the test split. No number here is a result.
- `tests/test_network_embedding.py` — 8 tests (walk reproducibility, metapath schema
  adherence, index shapes + isolated-case tracking, cluster separation, self/missing
  exclusion, save/load). Suite: 106 pass.

## Step 11 — Network Similarity
- Reprioritised remaining work (PROGRESS.md): MVP path = 11→12→13→14→17→18→20→21→19→22,
  then optional 15/16/23/24/25/26.
- `backend/app/ml/network_similarity.py` — combines the Step 10 embedding with a
  structural feature store into a `network_score(a,b)` + explanation breakdown. No
  FastAPI/DB deps.
  * `build_feature_store()` — per doc: resolved cited doc_ids, normalised raw citations,
    Act ids, section keys (reuses Steps 6 & 7).
  * `compute_statute_icf()` — IDF over the case corpus for Act/section overlap weighting.
  * Two scoring methods: `"embedding"` (base-paper signal: Hier-SPCNet cosine only) and
    `"blend"` (OURS: 0.60·embedding + 0.35·ICF-weighted statute Jaccard + 0.05·citation
    Jaccard, weights configurable via `BlendWeights`).
  * `NetworkSimilarityBreakdown` — network/embedding/statute-overlap/citation-overlap
    scores + shared Acts (with ICF), shared sections (with ICF), shared citation doc_ids,
    shared raw citations, co-citation count. This is what Step 14 explainability consumes.
  * `NetworkSimilarityIndex` — `similarity`, `breakdown`, `most_similar`, `save`/`load`
    (`feature_store.json`).
- `scripts/build_network_similarity.py` — `--method`, `--emb-method`, `--evaluate`.
- Real corpus, network sanity correlation (NOT hybrid eval — Step 22):
  | method    | validation (n=100)          | test (n=90)                  |
  |-----------|-----------------------------|------------------------------|
  | embedding | Pearson +0.346 / Spearman +0.444 | Pearson -0.043 / Spearman -0.052 |
  | blend     | Pearson +0.373 / Spearman +0.476 | Pearson -0.045 / Spearman -0.047 |
  Blending the statute overlap in nudges validation up (Spearman +0.03); test still ~0.
- `tests/test_network_similarity.py` — 10 tests (jaccard / weighted-jaccard / ICF,
  embedding vs blend method, breakdown lists, score bounds, most_similar ordering,
  save/load, real corpus). Suite: 116 pass.

## Step 12 — Hybrid Similarity
- `backend/app/ml/hybrid_similarity.py` — combines text score (Step 8) + network score
  (Step 11), no FastAPI/DB deps.
  * `WeightedSumCombiner` — `clip(α·text + β·network)`, α/β auto-normalised. OUR simple hybrid.
  * `LearnedCombiner` — Ridge (default) or small MLP trained on the expert pairs to map
    features → expert similarity; feature set configurable (default
    `[text_score, network_score]`, can add embedding/statute/citation/co-citation).
    Follows the *idea* of the base paper's **NN-Map+Conc**; the paper's reported gains are
    theirs on their 53k-case corpus, not ours. Includes `cross_val_pearson()` for an honest
    in-sample estimate.
  * `HybridSimilarityModel` — `features(a,b)`, `score`, `explain` (full breakdown incl. the
    Step 11 network breakdown), `most_similar`, `fit_learned(split="validation")`,
    `evaluate_correlation(split)`, `save`.
- `scripts/build_hybrid.py` — `--method weighted|learned`, `--alpha/--beta`, `--model`,
  `--evaluate`. Saves `data/embeddings/hybrid/` (config + `learned_model.joblib`).
- Real corpus (tfidf text + metapath2vec/blend network), numbers exactly as computed:
  | combiner            | validation                       | test                       |
  |---------------------|----------------------------------|----------------------------|
  | weighted 0.5/0.5    | Pearson +0.628 / Spearman +0.703 | -0.043 / -0.035  MSE 0.174 |
  | learned (Ridge)     | **5-fold CV** Pearson +0.647 / Spearman +0.749 / MSE 0.055 | -0.035 / -0.051  MSE 0.066 |
  Learned weights: text 0.86, network 0.27, intercept +0.21 — text carries the signal on
  validation; the learned intercept/scaling cuts MSE roughly in half vs the weighted sum.
  **Test correlation is ~0 for weighted AND learned** — now confirmed across text, network,
  and hybrid → this is a property of the 90-pair test split, not a pipeline bug. Step 22
  will characterise it (range restriction; 31/90 pairs touch a graph-isolated case;
  legal ≠ topical similarity).
- `tests/test_hybrid_similarity.py` — 12 tests (α/β normalise, clipping, learned fit
  recovers a linear target, CV metrics, unfitted raises, model features/score/explain/
  most_similar, fit_learned guard, save). Suite: 128 pass.

## Step 13 — Recommendation Engine
- `backend/app/ml/recommendation.py` — `RecommendationEngine` over the Step 8/11/12
  artifacts, no FastAPI/DB deps:
  * `recommend_for_case(doc_id, k, filters)` — ranks the corpus by hybrid score.
  * `recommend_for_text(text, k, year, filters)` — for an UPLOADED case: text score via
    `query_scores`, network score = **structural overlap only** (no graph node →
    `embedding_score=0`, `query_in_corpus=False` in the result — stated, not hidden).
  * `Filters` — year range, exclude ids, min hybrid score.
  * `Recommendation` — rank, scores (hybrid/text/network/embedding/statute/citation),
    shared Acts/sections/citations, co-citation count, and a rule-based `rationale`
    string ("strong textual overlap; shares 2 statute section(s) (s.92, s.88); …").
  * `RecommendationResult.to_dict()` is JSON-safe (ready for the FastAPI layer).
  * `explain_pair(a, b)` → the full Step 12 hybrid explanation.
- Supporting additions: `TextSimilarityIndex.query_scores()` / `.scores_for()` (all-doc
  cosine); embedder now persisted to `embedder.joblib` on `save()` (picklable ones) so a
  loaded index can serve raw-text queries; `NetworkSimilarityIndex.query_breakdown()` /
  `.query_overlaps()` for ad-hoc `DocFeatures`; `extract_doc_features()` factored out of
  `build_feature_store()`.
- `scripts/recommend.py` — CLI: `--case` / `--file` / `--text`, `--k`, `--year-min/max`,
  `--hybrid`, `--explain TARGET`.
- Qualitative check: query = "Jacob Mathew v State of Punjab" (landmark medical negligence)
  → top hybrid neighbours are **Kusum Sharma (0.91), Dr. P. B. Desai (0.73),
  Martin F. D'Souza (0.70)** — the medical-negligence precedent line, each with a
  breakdown and rationale. The engine does what the project set out to do.
- `tests/test_recommendation.py` — 7 tests (filters, case ranking + self-exclusion + year
  filter, breakdown/rationale, unknown case, uploaded-text path, JSON-safety, real corpus
  surfaces the medical-negligence line). Also updated 2 text-similarity persistence tests.
  Suite: 137 pass.

## Step 14 — Explainability
- `backend/app/ml/explainability.py` — `Explainer` producing one frontend-ready
  `SimilarityExplanation` per pair, no FastAPI/DB deps:
  * `contributions` — per-component attribution. Weighted: `alpha·text` vs `beta·network`
    (%s sum to 100). Learned-linear: `coef·feature` per feature + intercept term. MLP /
    unfitted: raw inputs, no attribution (stated).
  * `shared_statutes` — Act canonical name + section title from the curated hierarchy +
    ICF + `in_curated_hierarchy` flag; a shared Act is dropped if a shared section already
    represents it.
  * `shared_cited_cases` — resolved cited doc_ids → corpus title + year (unresolved corpus
    ids reported with `title=None`).
  * `shared_raw_citations` — the unresolved party-name overlaps (this is where the signal
    often is: two medical-negligence judgments share *Bolam*, *Hunter v Hanley*,
    *Hucks v Cole*, *Maynard*).
  * `summary` — natural-language, names the dominant driver + the shared provisions.
  * `explain_pair(query_id, target_id)` and `explain_text(text, target_id, year)` (upload).
- `StatuteHierarchy.section_title()` added; curated IPC Chapter IV expanded (ss. 79–93,
  the "General Exceptions" relevant to negligence/consent).
- `scripts/explain.py` — `--query`/`--file`, `--target`, `--hybrid`, `--json`.
- Demo (Jacob Mathew vs Kusum Sharma, learned combiner): text 62% / network 15% /
  intercept 23%; shared IPC s.88 & s.92 with titles; 9 shared citation strings incl. the
  classic English medical-negligence authorities. This is the explainability breakdown the
  project promised.
- `tests/test_explainability.py` — 7 tests (weighted contributions sum to score, learned
  intercept term, pair labels from hierarchy, JSON-safety, weak vs strong summary, upload
  path, real corpus finds the shared English authorities). Suite: 144 pass.

## Step 17 — Database
- `backend/app/config.py` — `Settings` (pydantic-settings) from `backend/.env`: DB URL, data
  dirs, model/hybrid defaults, CORS, upload cap. `database_url_resolved` makes a relative
  SQLite path absolute; `is_sqlite`, `cors_origins_list` helpers.
- `backend/app/database.py` — `make_engine()`, module `engine`/`SessionLocal`, `init_db()`,
  `bind()` (repoint for tests), `session_scope()` context manager, `get_db()` FastAPI
  dependency. SQLite `check_same_thread=False`.
- `backend/app/models/` — SQLAlchemy 2.0 (`Mapped`/`mapped_column`):
  * `Case` — one row per unique corpus doc (doc_id unique, year indexed, splits/JSON, title,
    char_len, sha1).
  * `UploadedDocument` — user upload (uuid id, filename, source_type, raw_text, page_count,
    needs_ocr, warnings/JSON, year_hint, created_at).
  * `RecommendationRun` + `RecommendationResult` — query history; run has ordered results
    (cascade delete), each result stores scores + rationale + the full explanation JSON.
- `backend/app/services/repository.py` — dependency-light DAO: `upsert_cases_from_corpus`
  (idempotent), `get_case`/`list_cases` (title/year filters + pagination)/`case_stats`,
  `create_upload`/`get_upload`, `record_run`/`get_run`/`list_recent_runs`.
- `scripts/init_db.py` — `--drop`; creates the 4 tables and upserts all 295 cases.
  Ran: 295 cases (235 with title, 1953–2018). DB file git-ignored.
- `tests/test_database.py` — 7 tests (schema, idempotent upsert, list filters, stats,
  upload + JSON warnings roundtrip, run stores ordered results + explanation JSON,
  recent runs + cascade delete). Suite: 151 pass.

## Step 18 — FastAPI Backend
- `backend/app/services/recommender.py` — `RecommenderService`: `load_default()` loads the
  persisted artifacts (`data/embeddings/{tfidf, network_metapath2vec, network_similarity,
  hybrid}`) or rebuilds any that are missing; holds corpus + `RecommendationEngine` +
  `Explainer` + `ServiceMeta`. `get_service()` is `lru_cache`d. Cold load ≈ 4s with
  artifacts present, ≈ 25s rebuilding.
- `backend/app/ml/hybrid_similarity.py` — added `HybridSimilarityModel.load()` (restores
  weighted or learned combiner incl. `learned_model.joblib`).
- `backend/app/schemas/models.py` — Pydantic v2 response models (health/ready, corpus
  stats, case summary/detail/list, recommendation item/response, explanation response).
- `backend/app/routers/` — `health` (`/health`, `/health/ready`), `stats` (`/stats`),
  `cases` (`/cases` list+filters+pagination, `/cases/{id}`, `/cases/{id}/text`),
  `recommend` (`/recommend/case/{id}` → Top-K, records a `RecommendationRun` unless
  `save=false`), `compare` (`/compare?a=&b=` → full explanation). All under `/api`.
- `backend/app/main.py` — `create_app()`: CORS from settings, routers mounted, lifespan
  runs `init_db()` (+ optional `WARM_ON_STARTUP=1`), `/` → `/docs`.
- `scripts/build_all.py` — runs the whole pipeline (preprocess → citations → statutes →
  graph → tfidf → metapath2vec → network-sim → hybrid → init_db) in order.
- Manual `TestClient` check: service loads in 3.8s; `/api/recommend/case/2005_S_628`
  returns Kusum Sharma / Dr P.B. Desai / Martin D'Souza with a persisted `run_id`;
  `/api/compare` returns the labelled breakdown.
- `tests/test_api.py` — 12 tests via an **injected lightweight service** + tmp SQLite
  (`dependency_overrides`): health/ready/stats, case list+filter+detail+text+404,
  recommend ranking + run persistence + year filter + 404, compare explanation +
  contribution %s + 404, root redirect, OpenAPI schema. `test_smoke` FastAPI-decoupling
  check moved to a subprocess (session-pollution-proof). Suite: 163 pass.
- README: added "Running the backend API" (`scripts/build_all.py` then `uvicorn`).

## Step 20 — Search
- `RecommenderService.search()` — three modes: `title` (DB-style substring on the title),
  `semantic` (TF-IDF `query_scores` content ranking over all 295 cases), `hybrid`
  (`0.8·semantic + 0.2·title-hit`). Year / split filters, pagination, per-hit `snippet`
  (the sentence with the most query-term hits) and `title_match` flag.
- `backend/app/services/repository.py` — `facets()`: year-decade histogram + split counts.
- `backend/app/routers/search.py` — `GET /api/search` (q, mode, year_min/max, split,
  limit, offset, with_facets) and `GET /api/search/facets`.
- `backend/app/schemas/models.py` — `SearchHit`, `SearchResponse`.
- Qualitative: `q="medical negligence"` → Martin D'Souza / Kusum Sharma / Jacob Mathew /
  Dr. P. B. Desai with on-point snippets; `q="bail"` (hybrid) → the bail cases;
  facets = {1950s:54 … 2010s:34}, {validation:143, test:153}.
- "Recommend from a search result" needs no new endpoint — the hit's `doc_id` goes to
  `GET /api/recommend/case/{doc_id}`.
- `tests/test_api.py` — +6 search tests (semantic ranking, title-mode filter, year filter
  + facets, facets endpoint, pagination, invalid-mode 422). Suite: 169 pass.

## Step 21 — Upload Workflow
- `backend/app/routers/upload.py` — `POST /api/upload` (multipart PDF/txt + optional
  `year`): extension + `%PDF-` magic + size (`upload_max_bytes`, 20 MB) + empty checks,
  then `extract_text()` → `UploadedDocument` row → 201 with id / char_count / page_count /
  `needs_ocr` / warnings / preview. `GET /api/upload/{id}` and `/{id}/text`.
- `backend/app/routers/recommend.py` — `POST /api/recommend/upload/{id}`: runs
  `recommend_for_text` on the stored upload (uses `year_hint`), records a
  `RecommendationRun` (`query_type="upload"`, `query_upload_id`).
- `backend/app/routers/compare.py` — `GET /api/compare/upload/{id}?target=`:
  `explain_text` for the upload vs a corpus case.
- `backend/app/schemas/models.py` — `UploadResponse`.
- Manual check: upload a generated 24-page PDF of Kusum Sharma → `POST
  /recommend/upload/{id}` returns Kusum Sharma (1.0), Jacob Mathew, Dr. P. B. Desai,
  Martin D'Souza; `GET /compare/upload/{id}?target=2005_S_628` → hybrid 0.78, shared
  authorities Bolam / Hunter v Hanley / Hucks v Cole.
- `tests/test_api.py` — +7 upload tests (txt round-trip → recommend + compare, pdf,
  bad extension / fake pdf / empty → 422, unknown upload / target → 404). Suite: 176 pass.

## Step 19 — React Frontend
- Scaffolded `frontend/` — React 18 + Vite 6 + Tailwind v4 (`@tailwindcss/vite`) + React
  Router 6 + Axios + Recharts. `npm install` (146 pkgs), `npm run build` clean
  (618 KB JS — Recharts; chunk-size warning only, Step 24 can code-split).
- `vite.config.js` proxies `/api` → `http://localhost:8000` (override `VITE_API_TARGET`).
- `src/services/api.js` — one axios fn per endpoint; `src/hooks/useAsync.js` — fetch-on-deps.
- Components: `Layout` (nav + global search), `ui.jsx` (`ScoreBar`, `Badge`, `Loading`,
  `ErrorBox`, `Section`), `RecommendationList` (per-result score bars + shared-statute
  badges + rationale), `ExplanationPanel` (score bars + **Recharts contribution chart** +
  shared-statutes table with Act name/section title/ICF + shared-citations badges + summary).
- Pages: Home (live `/stats`), Search (semantic/title/hybrid modes + decade facet bars),
  Case (detail + lazy full text), Recommend (case or upload, k / year filters), Compare
  (pair or upload-vs-target), Upload (PDF/txt → extraction result → "find similar").
- `.claude/launch.json` for `preview_start`. `frontend/dist` + `.claude/settings.local.json`
  git-ignored; `package-lock.json` committed.
- Verified live against the running backend in a browser: search "medical negligence" →
  the medical-negligence line with snippets + facets; `/recommend/case/2005_S_628` → ranked
  results with score bars; `/compare?a=2005_S_628&b=2010_B_40` → contribution chart
  (text 62% / network 15% / intercept 23%), IPC s.88/s.92 with titles + ICF, 9 shared
  English authorities (Bolam, Hunter v Hanley, …). No console errors.
- No automated frontend tests yet (optional Step 23). Backend suite still 176 pass.

## Step 22 — Evaluation
- `backend/app/ml/evaluation.py` — `correlation_metrics` (Pearson/Spearman/MSE/MAE,
  NaN-safe for constant predictions), `evaluate_pairwise`, `ranking_sanity` (MRR / hit@k
  over query docs with an expert partner ≥ 0.5), `diagnose_split` (range restriction +
  isolated-case flags). No FastAPI/DB deps.
- `backend/app/ml/hybrid_similarity.py` — `LearnedCombiner.cross_val_predict()` factored out.
- `scripts/evaluate.py` — benchmarks 13 configs (text tfidf/doc2vec, network
  metapath2vec/node2vec embedding & blend, weighted hybrids, learned linear/MLP, +year &
  random baselines) × 2 splits with an honest protocol (direct correlation; learned = 5-fold
  CV on validation, train-all-validation for test). Writes `data/processed/
  evaluation_report.json`, prints tables + a test-split diagnosis.
- `backend/app/routers/evaluate.py` — `GET /api/evaluate` (full report) and
  `/api/evaluate/summary` (compact table); 404 with instructions if not generated.
- **Results (real, from the 190 pairs):**
  * Validation — text TF-IDF ρ **+0.76**; network metapath2vec ρ +0.44 (blend +0.48);
    **learned-linear hybrid ρ +0.75, MSE 0.055** (≈½ the text-only MSE). Year-proximity
    baseline +0.29, random −0.13. → network mainly improves *calibration* here, not
    re-ranking; gain is smaller than the paper's (+11.8%) because our corpus has 24
    case→case edges vs their dense 53k graph — **not claimed as ours**.
  * Test — every method incl. both baselines ≈ 0 correlation. Diagnosed: score range
    restricted to [0.15,0.90]/16 values, 31/90 pairs touch a graph-isolated case, and even
    the 59-pair no-isolation subset is ρ ≈ −0.03. Ranking sanity is still non-zero on test
    (Doc2Vec MRR 0.30, hit@10 0.45) — the system retrieves similar cases even where it
    can't reproduce the expert score gradations. Full analysis in `EVALUATION.md`.
- `EVALUATION.md` — the writeup (tables + honest interpretation).
- `tests/test_evaluation.py` — 8 tests (metric math incl. constant-prediction NaN, ranking,
  diagnosis flags, real validation ρ > 0.5); `test_api.py` +1 (`/api/evaluate` via
  monkeypatched report dir). Suite: 184 pass.

## Step 15 — Case Comparison (side-by-side text)
- `frontend/src/components/HighlightedText.jsx` — wraps matches of shared statute refs
  (green) and shared cited-party keywords (amber) in `<mark>` via one combined RegExp.
- `frontend/src/components/SideBySide.jsx` — lazily fetches both judgments' full text
  (`/cases/{id}/text`, or the upload preview for the uploaded side) and renders them in two
  scrollable columns with the highlights from the explanation's `shared_statutes` /
  `shared_raw_citations`.
- `ComparePage` now shows the scoring breakdown *and* a "Side-by-side text" section.
- Verified in-browser: Jacob Mathew vs Kusum Sharma renders both texts with the shared
  authorities and s.302/s.88 highlighted.

## Step 16 — Citation Graph
- `RecommenderService.ego_graph(doc_id)` — a compact Hier-SPCNet neighbourhood from the
  feature store: the case, its top statutes (curated-labelled first, then by ICF), its
  top-6 similar cases, and edges where a similar case shares one of those statutes. No nx
  graph load needed.
- `backend/app/routers/graph.py` — `GET /api/graph/case/{doc_id}?similar=&statutes=`.
- `frontend/src/components/GraphView.jsx` — dependency-free radial SVG (center case,
  similar cases on the left arc, statutes on the right; edge thickness = hybrid score;
  click a case node to open it). `GraphPage` at `/case/:docId/graph`, linked from `CasePage`.
- `tests/test_api.py` +1 (ego graph shape + 404). Backend suite: 185 pass.

## Step 23 — Testing (frontend)
- Vitest + jsdom + @testing-library/react wired (`vite.config.js` `test` block,
  `src/setup-tests.js`, `npm run test`).
- `src/__tests__/components.test.jsx` (5) — ScoreBar, Badge, HighlightedText.
- `src/__tests__/api.test.js` (5) — axios client mocked: URL/param shaping for
  recommendCase / compare / egoGraph, FormData build for upload, `apiError` fallbacks.
- `npm run test` → 10 passing. Backend still 185.

## Step 24 — Optimization
- Route-level code splitting (`React.lazy` + `Suspense` in `App.jsx`): initial JS bundle
  **618 KB → 173 KB** (56 KB gzip); Recharts (382 KB) now loads only on `/compare`, the
  graph SVG only on `/case/:id/graph`. Chunk-size warning gone.
- `RecommenderService.ego_graph` reads the in-memory feature store (no graph rebuild);
  the learned combiner is loaded from `hybrid/learned_model.joblib` (no refit) when
  `build_all.py` has run — cold service load ≈ 1–4 s.
