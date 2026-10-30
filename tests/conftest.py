"""Shared pytest fixtures.

Kept deliberately small for now; grows as pipeline steps land.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Ensure tests always run against the committed evaluation corpus (data/raw/)
os.environ["CORPUS_SOURCE"] = "files"

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def raw_data_dir(repo_root: Path) -> Path:
    d = repo_root / "data" / "raw"
    if not d.is_dir():
        pytest.skip("data/raw/ not present")
    return d
