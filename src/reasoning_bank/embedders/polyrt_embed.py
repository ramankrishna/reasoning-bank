"""Polyrt-hosted embedder — stub (coming v0.2).

Will wire to polyrt's embeddings API once the upstream surface stabilizes.
Not registered yet so callers see a ``KeyError`` from
``get_embedder("polyrt")`` rather than a runtime failure on first
``embed()``.
"""

from __future__ import annotations


class PolyRTEmbedder:
    """Stub polyrt-hosted embedder. Raises NotImplementedError on use."""

    @property
    def dim(self) -> int:
        return 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "polyrt-hosted embedder is a stub — coming in a later release"
        )
