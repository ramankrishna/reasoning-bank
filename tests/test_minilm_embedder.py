"""MiniLMEmbedder smoke + registry test.

Uses the real sentence-transformers model (declared dep). Slow on first run
because it has to download the model, fast thereafter.
"""

from __future__ import annotations

import pytest

from reasoning_bank.embedders import Embedder, available, get_embedder, register
from reasoning_bank.embedders.minilm import DEFAULT_DIM, MiniLMEmbedder


def test_registry_has_minilm():
    assert "minilm" in available()


def test_get_embedder_returns_minilm_instance():
    e = get_embedder("minilm")
    assert isinstance(e, Embedder)
    assert e.dim == DEFAULT_DIM


def test_get_embedder_unknown_raises():
    with pytest.raises(KeyError):
        get_embedder("not-a-real-embedder")


def test_register_custom_factory():
    class _Fake:
        @property
        def dim(self) -> int:
            return 2

        def embed(self, texts):
            return [[0.0, 0.0] for _ in texts]

    register("fake-test-only", lambda **kw: _Fake())
    inst = get_embedder("fake-test-only")
    assert inst.dim == 2
    assert inst.embed(["a", "b"]) == [[0.0, 0.0], [0.0, 0.0]]


@pytest.mark.slow
def test_minilm_embed_produces_correct_dim():
    e = MiniLMEmbedder()
    vecs = e.embed(["hello world"])
    assert len(vecs) == 1
    assert len(vecs[0]) == DEFAULT_DIM
    assert all(isinstance(x, float) for x in vecs[0])
