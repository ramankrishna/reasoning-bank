"""LLM-as-judge — decide whether a trajectory succeeded or failed."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Literal

from reasoning_bank._json import load_prompt, parse_json_strict
from reasoning_bank.types import render_trajectory


LLMCallable = Callable[..., Awaitable[str]]


async def judge_trajectory(
    trajectory: list[Any],
    task: str,
    llm: LLMCallable,
) -> tuple[Literal["success", "failure"], str]:
    """Return ``(outcome, rationale)`` for a trajectory.

    The judge is strict: ambiguous or partial outcomes are classified as failures.
    ``llm`` is an async callable
    ``async def llm(prompt: str, *, system: str | None = None) -> str``.
    """
    template = load_prompt("judge")
    rendered = template.format(task=task, trajectory=render_trajectory(trajectory))

    raw = await llm(rendered, system=None)
    payload = parse_json_strict(raw or "")

    if not isinstance(payload, dict):
        raise ValueError(
            f"judge response must be a JSON object, got {type(payload).__name__}"
        )

    outcome = str(payload.get("outcome", "")).strip().lower()
    rationale = str(payload.get("rationale", "")).strip()
    if outcome not in ("success", "failure"):
        raise ValueError(
            f"judge outcome must be 'success' or 'failure', got {outcome!r}"
        )
    return outcome, rationale  # type: ignore[return-value]
