"""Opt-in vector quantization on the Edge store (correctness, not the size/recall trade-off).

What quantization buys or costs at scale is measured by ``benchmarks/edge_quantization.py``;
these tests only check that each mode is actually applied and that search stays correct.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qdrant_edge")

from loci.edge import EdgeMemoryStore, Memory  # noqa: E402

DIM = 128


def _fill(store, n=300, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, DIM))
    for i, v in enumerate(vecs):
        store.put(Memory(f"k{i}", v.tolist(), 0.5, 0.5, 0.0, 1_000 + i, text=f"item number {i}"))
    return vecs, rng


@pytest.mark.parametrize("mode", [None, "scalar", "binary"])
def test_search_stays_correct_in_every_mode(tmp_path, mode):
    store = EdgeMemoryStore(tmp_path / "s", DIM, "d", quantization=mode)
    vecs, rng = _fill(store)
    store.optimize()
    hits = 0
    for i in range(0, 300, 10):
        q = vecs[i] + 0.05 * rng.normal(size=DIM)
        hits += store.search(vector=q.tolist(), limit=1)[0].payload["key"] == f"k{i}"
    assert hits == 30
    # hybrid (dense + BM25) still works with quantized dense vectors
    top = store.search(vector=vecs[7].tolist(), text="item number 7", limit=3)
    assert top[0].payload["key"] == "k7"
    store.close()


@pytest.mark.parametrize("mode", ["scalar", "binary"])
def test_quantized_copy_really_exists_after_optimize(tmp_path, mode):
    store = EdgeMemoryStore(tmp_path / "s", DIM, "d", quantization=mode, indexing_threshold_kb=10)
    _fill(store, n=200)
    store.optimize()
    store.close()
    assert list((tmp_path / "s").rglob("quantized.data")), "quantization config was not applied"


def test_unquantized_store_has_no_quantized_files(tmp_path):
    store = EdgeMemoryStore(tmp_path / "s", DIM, "d")
    _fill(store, n=200)
    store.optimize()
    store.close()
    assert not list((tmp_path / "s").rglob("quantized.data"))


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="quantization"):
        EdgeMemoryStore(tmp_path / "s", DIM, "d", quantization="turbo")


def test_mirror_shard_uses_the_same_quantization(tmp_path):
    store = EdgeMemoryStore(
        tmp_path / "s",
        DIM,
        "d",
        mirror_path=tmp_path / "m",
        quantization="binary",
        indexing_threshold_kb=10,
    )
    rng = np.random.default_rng(1)
    pts = [
        {
            "id": f"00000000-0000-4000-8000-{i:012d}",
            "vector": rng.normal(size=DIM).tolist(),
            "payload": {"text": f"m{i}", "x": 0.1, "y": 0.1, "version": 1},
        }
        for i in range(200)
    ]
    store.mirror_upsert(pts)
    store._mirror.optimize()
    store.close()
    assert list((tmp_path / "m").rglob("quantized.data"))


def test_small_shards_stay_unquantized_by_design(tmp_path):
    """Below the indexing threshold Edge keeps a plain segment: quantization is a no-op there."""
    store = EdgeMemoryStore(tmp_path / "s", DIM, "d", quantization="binary")
    _fill(store, n=200)
    store.optimize()
    store.close()
    assert not list((tmp_path / "s").rglob("quantized.data"))
