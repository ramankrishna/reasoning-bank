"""Distill generalizable memory items from agent trajectories."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Literal

from reasoning_bank._json import load_prompt, parse_json_strict
from reasoning_bank.item import MemoryItem
from reasoning_bank.types import render_trajectory


LLMCallable = Callable[..., Awaitable[str]]


async def distill_memories(
    trajectory: list[Any],
    task: str,
    outcome: Literal["success", "failure"],
    llm: LLMCallable,
    *,
    scope: str = "global",
) -> list[MemoryItem]:
    """Use an LLM to extract 1-3 generalizable ``MemoryItem`` records from a trajectory.

    The LLM is an async callable of the form
    ``async def llm(prompt: str, *, system: str | None = None) -> str``.
    Returned items have ``embedding=None`` — the bank fills them in before storage.
    """
    if outcome not in ("success", "failure"):
        raise ValueError(f"outcome must be 'success' or 'failure', got {outcome!r}")

    template_name = (
        "induction_success" if outcome == "success" else "induction_failure"
    )
    template = load_prompt(template_name)
    rendered = template.format(task=task, trajectory=render_trajectory(trajectory))

    raw = await llm(rendered, system=None)
    payload = parse_json_strict(raw or "")

    if not isinstance(payload, list):
        raise ValueError(
            f"induction response must be a JSON array, got {type(payload).__name__}"
        )

    items: list[MemoryItem] = []
    for entry in payload[:3]:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title", "")).strip()
        description = str(entry.get("description", "")).strip()
        content = str(entry.get("content", "")).strip()
        if not (title and description and content):
            continue
        items.append(
            MemoryItem(
                title=title,
                description=description,
                content=content,
                source=outcome,
                task_signature=task,
                scope=scope,
                embedding=None,
            )
        )
    return items
