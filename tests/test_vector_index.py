"""Tests for app.ml.vector_index (Phase 2, P2)."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.vector_index import VectorIndex


def _toy():
    # two clusters: 0-2 near [1,0,..], 3-5 near [0,1,..]
    rng = np.random.default_rng(0)
    a = np.tile([1.0, 0.0, 0.0], (3, 1)) + 0.05 * rng.normal(size=(3, 3))
    b = np.tile([0.0, 1.0, 0.0], (3, 1)) + 0.05 * rng.normal(size=(3, 3))
    return ["a0", "a1", "a2", "b0", "b1", "b2"], np.vstack([a, b]).astype(np.float32)


@pytest.mark.parametrize("backend", ["numpy", "faiss"])
def test_search_and_most_similar(backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
    ids, mat = _toy()
    vi = VectorIndex(ids, mat, backend=backend)
    assert vi.backend == backend

    nn = vi.most_similar("a0", k=2)
    assert {n.doc_id for n in nn} <= {"a1", "a2"}
    assert "a0" not in {n.doc_id for n in nn}

    hits = vi.search(np.array([0.0, 1.0, 0.0], dtype=np.float32), k=3)
    assert {h.doc_id for h in hits} == {"b0", "b1", "b2"}
    assert hits[0].score >= hits[-1].score  # descending


def test_backends_agree_on_ranking():
    faiss = pytest.importorskip("faiss")  # noqa: F841
    ids, mat = _toy()
    vn = VectorIndex(ids, mat, backend="numpy")
    vf = VectorIndex(ids, mat, backend="faiss")
    q = mat[0]
    assert [n.doc_id for n in vn.search(q, 4)] == [n.doc_id for n in vf.search(q, 4)]


def test_zero_query_returns_empty():
    ids, mat = _toy()
    vi = VectorIndex(ids, mat, backend="numpy")
    assert vi.search(np.zeros(3, dtype=np.float32), k=3) == []


def test_save_load_roundtrip(tmp_path):
    ids, mat = _toy()
    vi = VectorIndex(ids, mat, backend="numpy")
    vi.save(tmp_path)
    loaded = VectorIndex.load(tmp_path, backend="numpy")
    assert loaded.doc_ids == ids
    assert [n.doc_id for n in loaded.most_similar("b0", 2)] == \
           [n.doc_id for n in vi.most_similar("b0", 2)]


def test_requesting_faiss_without_install_raises(monkeypatch):
    import app.ml.vector_index as m

    monkeypatch.setattr(m, "_HAVE_FAISS", False)
    ids, mat = _toy()
    with pytest.raises(RuntimeError):
        VectorIndex(ids, mat, backend="faiss")
    # auto still works (falls back to numpy)
    assert VectorIndex(ids, mat, backend="auto").backend == "numpy"
