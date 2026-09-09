# Environment Setup

This is **Step 1** of the incremental build (see `PROJECT_NOTES.md` §6-7). It gets the
Python toolchain ready; no pipeline code runs yet.

## Prerequisites (already verified on the dev machine)

| Tool   | Required            | Found on this machine        |
|--------|--------------------|------------------------------|
| Python | **3.10 - 3.12**    | 3.12.5 (`C:\Python312`)      |
| Node   | 18+                | 25.8.2                       |
| npm    | 9+                 | 11.11.1                      |
| git    | any recent         | 2.53.0                       |

> **Do not use Python 3.13 / 3.14.** `gensim` (Doc2Vec, the base-paper method)
> and the pinned `numpy<2` scientific stack have no stable wheels for those
> versions yet. The system default `python` here is 3.14 — use the explicit
> 3.12 interpreter for the virtual environment.

## One-time setup

From the repo root (`legal-case-similarity/`):

```bash
# 1. Create the virtual environment with Python 3.12
py -3.12 -m venv .venv
#    (if `py` is unavailable:  "C:\Python312\python.exe" -m venv .venv)

# 2. Activate it
#    PowerShell:
.venv\Scripts\Activate.ps1
#    Git Bash / WSL:
source .venv/Scripts/activate
#    cmd.exe:
.venv\Scripts\activate.bat

# 3. Upgrade pip and install the core dependencies
python -m pip install --upgrade pip
pip install -r backend/requirements.txt

# 4. Create your local env file from the template
copy backend\.env.example backend\.env      # PowerShell/cmd
#    cp backend/.env.example backend/.env    # Git Bash
```

### Optional ML extras (install later, NOT now)

`backend/requirements-ml.txt` holds `sentence-transformers` and `faiss-cpu`,
used only by **our own enhancements** (SBERT embeddings, FAISS search). They are
large and are installed at the Text Similarity / Build Index steps:

```bash
pip install -r backend/requirements-ml.txt
```

## Verify the environment

```bash
python scripts/check_env.py
```

Expected: every line prints `[ OK ]`, ending with

```
[ OK ] Environment is ready. Proceed to Step 2 (Repository Setup).
```

The script checks the interpreter version, imports every core package, confirms
`numpy<2` (gensim constraint), and confirms the real dataset in `data/raw/`
(143 validation + 153 test documents; 100 + 90 expert-scored pairs — the counts
from the base paper, read live from disk, never hard-coded as results).

## Files in this step

| File                          | Purpose                                                        |
|-------------------------------|---------------------------------------------------------------|
| `backend/requirements.txt`    | Core deps for Steps 3-18 (pinned as a gensim-compatible set).  |
| `backend/requirements-ml.txt` | Optional heavy deps for our SBERT / FAISS enhancements.        |
| `backend/requirements.lock.txt` | `pip freeze` snapshot of the exact versions installed here.  |
| `backend/.env.example`        | Config template; copy to `backend/.env` (git-ignored).        |
| `scripts/check_env.py`        | Environment + dataset verification (exit 0 = ready).           |

## Reproduce the exact versions used during development

```bash
pip install -r backend/requirements.lock.txt
```
