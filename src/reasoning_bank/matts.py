"""Memory-Aware Test-Time Scaling — framework-agnostic.

MaTTS runs the same task k times in parallel through a caller-supplied
``rollout_fn`` and then asks a contrast LLM to compare the resulting
trajectories and extract higher-quality memories than any single rollout
could produce on its own. Contrast memories are stored with
``source="matts_contrast"`` and pass through the bank's normal merge logic.

The design decouples MaTTS from any particular agent framework: the caller
supplies a coroutine ``rollout_fn() -> Trajectory`` that runs one rollout
of their agent. MaTTS schedules ``k`` of them in parallel.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from reasoning_bank._json import load_prompt, parse_json_strict
from reasoning_bank.item import MemoryItem
from reasoning_bank.types import Trajectory, _coerce_trajectory, render_trajectory

if TYPE_CHECKING:
    from reasoning_bank.bank import ReasoningBank

logger = logging.getLogger(__name__)

RolloutFn = Callable[[], Awaitable[Any]]
LLMCallable = Callable[..., Awaitable[str]]


async def matts_run(
    rollout_fn: RolloutFn,
    task: str,
    bank: "ReasoningBank",
    k: int = 4,
    *,
    contrast_llm: LLMCallable | None = None,
) -> tuple[list[Trajectory], list[MemoryItem]]:
    """Run ``rollout_fn`` k times in parallel and contrast-distill memories.

    ``rollout_fn`` is an async zero-arg callable that returns one trajectory
    (in any accepted form — Turn list, dict list, or duck-typed objects).
    MaTTS coerces each result, renders them for the contrast prompt, and
    asks ``contrast_llm`` (defaults to ``bank.llm``) for contrastive
    insights.

    Returns ``(trajectories, distilled_memories)``. If the contrast LLM
    returns truncated or invalid JSON, ``distilled_memories`` is ``[]`` and
    a warning is logged — this never raises.
    """
    if k < 1:
        raise ValueError(f"matts_run: k must be >= 1, got {k}")

    llm = contrast_llm if contrast_llm is not None else bank.llm
    if llm is None:
        raise ValueError(
            "matts_run: no contrast LLM — pass contrast_llm or set llm on the bank."
        )

    raw_results = await asyncio.gather(*[rollout_fn() for _ in range(k)])
    trajectories: list[Trajectory] = [_coerce_trajectory(r) for r in raw_results]

    distilled = await _contrast_distill(trajectories, task, llm, bank)
    return trajectories, distilled


async def _contrast_distill(
    trajectories: list[Trajectory],
    task: str,
    llm: LLMCallable,
    bank: "ReasoningBank",
) -> list[MemoryItem]:
    from reasoning_bank.merge import integrate_candidate

    rendered = [
        f"=== Trajectory {i} ===\n{render_trajectory(t)}"
        for i, t in enumerate(trajectories, start=1)
    ]
    template = load_prompt("matts_contrast")
    prompt = template.format(task=task, trajectories="\n\n".join(rendered))

    raw = await llm(prompt, system=None)
    raw = raw or ""
    try:
        payload = parse_json_strict(raw)
    except ValueError as exc:
        hint = (
            " — response appears truncated, consider raising max_tokens on the contrast LLM"
            if _looks_truncated(raw)
            else ""
        )
        logger.warning(
            "matts_contrast: skipping distillation, could not parse LLM response as JSON%s (%s)",
            hint,
            exc,
        )
        return []

    if not isinstance(payload, list):
        logger.warning(
            "matts_contrast: skipping distillation, response must be a JSON array, got %s",
            type(payload).__name__,
        )
        return []

    integrated: list[MemoryItem] = []
    for entry in payload[:3]:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title", "")).strip()
        description = str(entry.get("description", "")).strip()
        content = str(entry.get("content", "")).strip()
        if not (title and description and content):
            continue
        candidate = MemoryItem(
            title=title,
            description=description,
            content=content,
            source="matts_contrast",
            task_signature=task,
            scope=bank.scope,
            embedding=None,
        )
        candidate.embedding = bank.embedder.embed([candidate.signature_text()])[0]
        _action, stored = await integrate_candidate(bank, candidate)
        integrated.append(stored)
    return integrated


def _looks_truncated(text: str) -> bool:
    stripped = text.strip().rstrip("`").rstrip()
    return bool(stripped) and stripped[-1] not in ("}", "]")
