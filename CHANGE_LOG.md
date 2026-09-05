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
