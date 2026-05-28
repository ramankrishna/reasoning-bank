"""Retrieve relevant memories from a ReasoningBank for a given task."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from reasoning_bank.item import MemoryItem

if TYPE_CHECKING:
    from reasoning_bank.bank import ReasoningBank


async def retrieve_relevant(
    bank: "ReasoningBank",
    task: str,
    k: int = 5,
    scope: str | None = None,
    min_confidence: float = 0.3,
) -> list[MemoryItem]:
    """Return up to ``k`` memories ranked by ``similarity * confidence``.

    Side effects: each returned item has its ``use_count`` and
    ``last_used_at`` updated and written back to the store.
    """
    if k <= 0:
        return []

    embedder = bank.embedder
    query_vec = embedder.embed([task])[0]

    filters: dict[str, Any] | None = None
    if scope is not None:
        filters = {"scope": scope}

    candidates = bank.store.search(query_vec, k=k * 2, filters=filters)

    eligible: list[tuple[MemoryItem, float, float]] = []
    for item, sim in candidates:
        if item.confidence < min_confidence:
            continue
        eligible.append((item, sim, sim * item.confidence))

    eligible.sort(key=lambda triple: triple[2], reverse=True)
    selected = [item for item, _, _ in eligible[:k]]

    now = datetime.utcnow()
    for item in selected:
        item.use_count += 1
        item.last_used_at = now
        try:
            bank.store.update(item)
        except KeyError:
            continue

    return selected
