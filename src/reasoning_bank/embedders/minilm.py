"""MiniLM embedder — sentence-transformers all-MiniLM-L6-v2 (384-d)."""

from __future__ import annotations

from typing import Any

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIM = 384


class MiniLMEmbedder:
    """Lazy-loaded sentence-transformers wrapper.

    The underlying model is downloaded and instantiated only on the first
    call to ``embed()``, so importing this module is cheap and safe.
    """

    def __init__(
        self, model_name: str = DEFAULT_MODEL, device: str | None = None
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None

    @property
    def dim(self) -> int:
        return DEFAULT_DIM

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=False
        )
        return [list(map(float, vec)) for vec in vectors]
