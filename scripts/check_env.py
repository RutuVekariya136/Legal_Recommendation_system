"""
check_env.py - Step 1 environment verification.

Run this after creating the virtual environment and installing
backend/requirements.txt. It verifies, without touching any pipeline code:

  1. The Python interpreter version is in the supported range (3.10 - 3.12).
  2. Every core third-party package imports and reports a version.
  3. The scientific stack is internally consistent (numpy < 2, gensim imports).
  4. The real dataset is present in data/raw/ with the expected structure
     and document / expert-pair counts (see CLAUDE.md sec 3).

Exit code 0 = environment is ready for Step 2/3. Non-zero = something is off,
with a printed explanation. No numbers here are fabricated - counts are read
live from the filesystem and compared against the values documented in the
base paper.

Usage:
    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (import_name, pretty_name) for every core dependency in requirements.txt
CORE_PACKAGES: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("pandas", "pandas"),
    ("gensim", "gensim"),
    ("networkx", "networkx"),
    ("rapidfuzz", "rapidfuzz"),
    ("pymupdf", "pymupdf"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("pydantic_settings", "pydantic-settings"),
    ("dotenv", "python-dotenv"),
    ("multipart", "python-multipart"),
    ("sqlalchemy", "sqlalchemy"),
    ("pytest", "pytest"),
    ("httpx", "httpx"),
]

# Expected dataset shape - from CLAUDE.md sec 3 / the base paper. Do NOT edit
# these to make the check pass; fix the data instead.
EXPECTED = {
    "validation": {"documents": 143, "pairs": 100},
    "test": {"documents": 153, "pairs": 90},
}

PASS = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def _version(mod, pretty: str) -> str:
    try:
        return importlib.metadata.version(pretty)
    except importlib.metadata.PackageNotFoundError:
        pass
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
        if v is not None:
            return str(v)
    return "unknown"


def check_python() -> bool:
    major, minor = sys.version_info[:2]
    ok = (major == 3) and (10 <= minor <= 12)
    line = f"{PASS if ok else FAIL} Python {major}.{minor}.{sys.version_info[2]}"
    print(line)
    if not ok:
        print(
            "       Supported range is CPython 3.10 - 3.12. "
            "gensim / numpy<2 do not have stable wheels for 3.13+ yet.\n"
            "       Create the venv with a 3.12 interpreter, e.g.:\n"
            '       py -3.12 -m venv .venv     (or)     "C:\\Python312\\python.exe" -m venv .venv'
        )
    print(f"       executable: {sys.executable}")
    return ok


def check_packages() -> bool:
    all_ok = True
    numpy_major = None
    for import_name, pretty in CORE_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            ver = _version(mod, pretty)
            print(f"{PASS} {pretty:<20} {ver}")
            if import_name == "numpy":
                numpy_major = int(ver.split(".")[0])
        except Exception as exc:  # noqa: BLE001 - want the message verbatim
            all_ok = False
            print(f"{FAIL} {pretty:<20} import failed: {exc}")

    if numpy_major is not None and numpy_major >= 2:
        all_ok = False
        print(
            f"{FAIL} numpy {numpy_major}.x is installed, but gensim 4.3.3 requires "
            "numpy<2. Reinstall with: pip install -r backend/requirements.txt"
        )
    return all_ok


def _count_txt(path: Path) -> int:
    return len(list(path.glob("*.txt"))) if path.is_dir() else 0


def _count_pairs(csv_path: Path) -> int:
    """Count non-empty data rows in similarity_scores.csv (doc1,doc2,score).

    The file ships without a header; every non-blank line is a pair.
    """
    if not csv_path.is_file():
        return 0
    rows = 0
    with csv_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            first = line.split(",")[0].strip().lower()
            if first in {"doc1", "doc_1", "query", "case1"}:  # tolerate a header
                continue
            rows += 1
    return rows


def check_dataset() -> bool:
    raw = REPO_ROOT / "data" / "raw"
    if not raw.is_dir():
        print(f"{FAIL} data/raw/ not found at {raw}")
        return False

    all_ok = True
    for split, exp in EXPECTED.items():
        split_dir = raw / split
        docs_dir = split_dir / "documents"
        csv_path = split_dir / "similarity_scores.csv"

        n_docs = _count_txt(docs_dir)
        n_pairs = _count_pairs(csv_path)

        docs_ok = n_docs == exp["documents"]
        pairs_ok = n_pairs == exp["pairs"]

        tag = PASS if (docs_ok and pairs_ok) else FAIL
        all_ok = all_ok and docs_ok and pairs_ok
        print(
            f"{tag} {split:<11} documents={n_docs} (expect {exp['documents']}), "
            f"pairs={n_pairs} (expect {exp['pairs']})"
        )
        if not docs_dir.is_dir():
            print(f"       missing folder: {docs_dir}")
        if not csv_path.is_file():
            print(f"       missing file:   {csv_path}")

    total_docs = _count_txt(raw / "validation" / "documents") + _count_txt(
        raw / "test" / "documents"
    )
    print(f"       corpus size: {total_docs} documents total (demo corpus, ~296 expected)")
    return all_ok


def check_dirs() -> bool:
    needed = [
        "data/raw",
        "data/processed",
        "data/embeddings",
        "data/graphs",
        "data/indexes",
        "backend/app/ml",
        "backend/app/routers",
        "scripts",
        "tests",
    ]
    all_ok = True
    for rel in needed:
        p = REPO_ROOT / rel
        ok = p.is_dir()
        all_ok = all_ok and ok
        print(f"{PASS if ok else FAIL} dir {rel}")
    return all_ok


def main() -> int:
    print("=" * 70)
    print("Step 1 - Environment verification")
    print("=" * 70)

    print("\n-- Python interpreter --")
    py_ok = check_python()

    print("\n-- Core packages --")
    pkg_ok = check_packages()

    print("\n-- Project directories --")
    dir_ok = check_dirs()

    print("\n-- Dataset (data/raw/) --")
    data_ok = check_dataset()

    print("\n" + "=" * 70)
    everything_ok = py_ok and pkg_ok and dir_ok and data_ok
    if everything_ok:
        print(f"{PASS} Environment is ready. Proceed to Step 2 (Repository Setup).")
        return 0
    print(f"{FAIL} Environment is NOT ready - see the messages above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
