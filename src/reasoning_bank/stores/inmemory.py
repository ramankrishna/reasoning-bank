"""In-memory MemoryStore — dict-backed, primarily for unit tests."""

from __future__ import annotations

import math
from typing import Any

from reasoning_bank.item import MemoryItem


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class InMemoryStore:
    """Process-local MemoryStore. Not persistent."""

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    def add(self, item: MemoryItem) -> None:
        if item.id in self._items:
            raise ValueError(f"MemoryItem {item.id} already exists")
        if item.embedding is None:
            raise ValueError(f"MemoryItem {item.id} has no embedding")
        self._items[item.id] = item.model_copy(deep=True)

    def get(self, item_id: str) -> MemoryItem | None:
        existing = self._items.get(item_id)
        return existing.model_copy(deep=True) if existing else None

    def update(self, item: MemoryItem) -> None:
        if item.id not in self._items:
            raise KeyError(item.id)
        self._items[item.id] = item.model_copy(deep=True)

    def delete(self, item_id: str) -> None:
        self._items.pop(item_id, None)

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        scored: list[tuple[MemoryItem, float]] = []
        for item in self._items.values():
            if item.embedding is None:
                continue
            if filters and not _matches(item, filters):
                continue
            scored.append(
                (item.model_copy(deep=True), _cosine(query_embedding, item.embedding))
            )
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def list_all(self, scope: str | None = None) -> list[MemoryItem]:
        items = self._items.values()
        if scope is not None:
            items = [it for it in items if it.scope == scope]  # type: ignore[assignment]
        return [it.model_copy(deep=True) for it in items]


def _matches(item: MemoryItem, filters: dict[str, Any]) -> bool:
    for key, value in filters.items():
        if getattr(item, key, None) != value:
            return False
    return True
