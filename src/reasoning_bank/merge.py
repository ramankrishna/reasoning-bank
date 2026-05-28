"""Merge logic for integrating new candidate memories into the ReasoningBank.

Three branches:
- ``sim > merge_threshold`` (default 0.92): call merge prompt, replace existing.
- ``link_threshold < sim <= merge_threshold`` (default 0.75 < sim <= 0.92): keep
  both, link via ``related_ids``.
- ``sim <= link_threshold``: insert as new.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable, Literal

from reasoning_bank._json import load_prompt, parse_json_strict
from reasoning_bank.item import MemoryItem

if TYPE_CHECKING:
    from reasoning_bank.bank import ReasoningBank


LLMCallable = Callable[..., Awaitable[str]]
MergeAction = Literal["insert", "link", "merge", "replace", "keep_both"]


async def integrate_candidate(
    bank: "ReasoningBank",
    candidate: MemoryItem,
) -> tuple[MergeAction, MemoryItem]:
    """Decide what to do with a new candidate, applying the change to the store.

    Returns ``(action, item)`` where ``item`` is the final stored MemoryItem
    (newly inserted, merged, or linked).
    """
    if candidate.embedding is None:
        vec = bank.embedder.embed([candidate.signature_text()])[0]
        candidate.embedding = vec

    link_thr, merge_thr = bank.merge_thresholds
    neighbors = bank.store.search(
        candidate.embedding,
        k=3,
        filters={"scope": candidate.scope},
    )

    if not neighbors:
        bank.store.add(candidate)
        return "insert", candidate

    best_item, best_sim = neighbors[0]

    if best_sim > merge_thr:
        if bank.llm is None:
            bank.store.update(_replace_fields(best_item, candidate))
            return "replace", best_item
        action, merged = await _llm_merge(bank.llm, best_item, candidate)
        if action == "keep_both":
            return _link(bank, best_item, candidate)
        bank.store.update(merged)
        return action, merged

    if best_sim > link_thr:
        return _link(bank, best_item, candidate)

    bank.store.add(candidate)
    return "insert", candidate


def _link(
    bank: "ReasoningBank", existing: MemoryItem, candidate: MemoryItem
) -> tuple[MergeAction, MemoryItem]:
    if existing.id not in candidate.related_ids:
        candidate.related_ids.append(existing.id)
    bank.store.add(candidate)
    if candidate.id not in existing.related_ids:
        existing.related_ids.append(candidate.id)
        bank.store.update(existing)
    return "link", candidate


def _replace_fields(existing: MemoryItem, candidate: MemoryItem) -> MemoryItem:
    existing.title = candidate.title
    existing.description = candidate.description
    existing.content = candidate.content
    existing.source = candidate.source
    existing.embedding = candidate.embedding
    return existing


async def _llm_merge(
    llm: LLMCallable, existing: MemoryItem, candidate: MemoryItem
) -> tuple[MergeAction, MemoryItem]:
    template = load_prompt("merge")
    rendered = template.format(
        existing_title=existing.title,
        existing_description=existing.description,
        existing_content=existing.content,
        new_title=candidate.title,
        new_description=candidate.description,
        new_content=candidate.content,
    )
    raw = await llm(rendered, system=None)
    payload = parse_json_strict(raw or "")

    if not isinstance(payload, dict):
        raise ValueError(
            f"merge response must be a JSON object, got {type(payload).__name__}"
        )

    action = str(payload.get("action", "")).strip().lower()
    if action not in ("merge", "replace", "keep_both"):
        raise ValueError(
            f"merge action must be merge/replace/keep_both, got {action!r}"
        )

    if action == "keep_both":
        return "keep_both", existing

    if action == "replace":
        return "replace", _replace_fields(existing, candidate)

    merged_title = str(payload.get("merged_title", "")).strip() or existing.title
    merged_description = (
        str(payload.get("merged_description", "")).strip() or existing.description
    )
    merged_content = (
        str(payload.get("merged_content", "")).strip() or existing.content
    )

    existing.title = merged_title
    existing.description = merged_description
    existing.content = merged_content
    existing.embedding = candidate.embedding
    return "merge", existing
