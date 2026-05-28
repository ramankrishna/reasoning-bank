"""distill_memories with a mock async LLM callable."""

from __future__ import annotations

import json

import pytest

from reasoning_bank.induction import distill_memories
from reasoning_bank.types import Turn


def _make_llm(reply: str):
    captured: dict = {}

    async def llm(prompt: str, *, system: str | None = None) -> str:
        captured["prompt"] = prompt
        captured["system"] = system
        return reply

    return llm, captured


async def test_distill_returns_memory_items_on_success():
    payload = json.dumps(
        [
            {
                "title": "Verify env before API calls",
                "description": "Apply when calling external HTTP tools.",
                "content": "Detail goes here.",
            }
        ]
    )
    llm, _ = _make_llm(payload)
    trajectory = [Turn(role="user", content="task done")]

    items = await distill_memories(
        trajectory=trajectory, task="t", outcome="success", llm=llm
    )
    assert len(items) == 1
    assert items[0].source == "success"
    assert items[0].title == "Verify env before API calls"
    assert items[0].embedding is None
    assert items[0].task_signature == "t"


async def test_distill_failure_branch_uses_failure_prompt():
    payload = json.dumps(
        [
            {
                "title": "Do not retry on timeout",
                "description": "Apply when a tool times out.",
                "content": "Detail.",
            }
        ]
    )
    llm, captured = _make_llm(payload)
    trajectory = [Turn(role="user", content="broken")]

    items = await distill_memories(
        trajectory=trajectory, task="t", outcome="failure", llm=llm
    )
    assert items[0].source == "failure"
    assert "FAILED" in captured["prompt"]


async def test_distill_caps_at_three_entries():
    payload = json.dumps(
        [
            {"title": f"t{i}", "description": "d", "content": "c"} for i in range(7)
        ]
    )
    llm, _ = _make_llm(payload)
    items = await distill_memories(
        trajectory=[Turn(role="user", content="x")],
        task="t",
        outcome="success",
        llm=llm,
    )
    assert len(items) == 3


async def test_distill_drops_incomplete_entries():
    payload = json.dumps(
        [
            {"title": "", "description": "d", "content": "c"},
            {"title": "ok", "description": "d", "content": "c"},
            {"title": "bad", "description": "", "content": "c"},
        ]
    )
    llm, _ = _make_llm(payload)
    items = await distill_memories(
        trajectory=[Turn(role="user", content="x")],
        task="t",
        outcome="success",
        llm=llm,
    )
    assert [it.title for it in items] == ["ok"]


async def test_distill_invalid_outcome_raises():
    llm, _ = _make_llm("[]")
    with pytest.raises(ValueError):
        await distill_memories(
            trajectory=[Turn(role="user", content="x")],
            task="t",
            outcome="other",  # type: ignore[arg-type]
            llm=llm,
        )


async def test_distill_strips_json_fence():
    payload = '```json\n[{"title": "t", "description": "d", "content": "c"}]\n```'
    llm, _ = _make_llm(payload)
    items = await distill_memories(
        trajectory=[Turn(role="user", content="x")],
        task="t",
        outcome="success",
        llm=llm,
    )
    assert len(items) == 1


async def test_distill_non_array_payload_raises():
    llm, _ = _make_llm('{"not": "a list"}')
    with pytest.raises(ValueError):
        await distill_memories(
            trajectory=[Turn(role="user", content="x")],
            task="t",
            outcome="success",
            llm=llm,
        )
