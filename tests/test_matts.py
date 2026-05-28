"""matts_run: k parallel rollouts + contrast-distill with truncation safety."""

from __future__ import annotations

import json

import pytest

from reasoning_bank import InMemoryStore, ReasoningBank, Turn, matts_run


class _FakeEmbedder:
    dim = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


def _make_llm(replies: list[str]):
    calls = {"i": 0, "prompts": []}

    async def llm(prompt: str, *, system: str | None = None) -> str:
        calls["prompts"].append(prompt)
        i = min(calls["i"], len(replies) - 1)
        calls["i"] += 1
        return replies[i]

    return llm, calls


def _bank(llm) -> ReasoningBank:
    return ReasoningBank(llm=llm, store=InMemoryStore(), embedder=_FakeEmbedder())


async def test_matts_runs_k_rollouts_and_distills_memories():
    contrast_reply = json.dumps(
        [
            {
                "title": "validate intermediate results",
                "description": "when chaining tool calls",
                "content": "compare winners vs losers",
            }
        ]
    )
    llm, _ = _make_llm([contrast_reply])
    bank = _bank(llm)

    rollout_count = {"n": 0}

    async def rollout_fn():
        rollout_count["n"] += 1
        return [
            Turn(role="user", content="task"),
            Turn(role="assistant", content=f"reply {rollout_count['n']}"),
        ]

    trajectories, memories = await matts_run(rollout_fn, "task", bank, k=4)
    assert rollout_count["n"] == 4
    assert len(trajectories) == 4
    assert len(memories) == 1
    assert memories[0].source == "matts_contrast"


async def test_matts_uses_separate_contrast_llm_when_passed():
    bank_llm, bank_calls = _make_llm(["ignored"])
    bank = _bank(bank_llm)

    contrast_reply = json.dumps(
        [{"title": "t", "description": "d", "content": "c"}]
    )
    contrast_llm, contrast_calls = _make_llm([contrast_reply])

    async def rollout_fn():
        return [Turn(role="assistant", content="x")]

    _, memories = await matts_run(
        rollout_fn, "task", bank, k=2, contrast_llm=contrast_llm
    )
    assert len(memories) == 1
    assert contrast_calls["i"] == 1
    assert bank_calls["i"] == 0


async def test_matts_accepts_dict_trajectories_from_rollout_fn():
    contrast_reply = json.dumps(
        [{"title": "t", "description": "d", "content": "c"}]
    )
    llm, _ = _make_llm([contrast_reply])
    bank = _bank(llm)

    async def rollout_fn():
        return [{"role": "assistant", "content": "x"}]

    trajectories, memories = await matts_run(rollout_fn, "task", bank, k=2)
    assert len(trajectories) == 2
    assert all(t[0].role == "assistant" for t in trajectories)
    assert len(memories) == 1


async def test_matts_truncated_json_returns_empty_memories(caplog):
    truncated = '[{"title": "t", "description": "d", "content": "c'
    llm, _ = _make_llm([truncated])
    bank = _bank(llm)

    async def rollout_fn():
        return [Turn(role="assistant", content="x")]

    with caplog.at_level("WARNING", logger="reasoning_bank.matts"):
        trajectories, memories = await matts_run(rollout_fn, "task", bank, k=2)

    assert len(trajectories) == 2
    assert memories == []
    assert any("truncated" in rec.message.lower() for rec in caplog.records)


async def test_matts_non_array_payload_returns_empty_memories(caplog):
    llm, _ = _make_llm(['{"oops": "not a list"}'])
    bank = _bank(llm)

    async def rollout_fn():
        return [Turn(role="assistant", content="x")]

    with caplog.at_level("WARNING", logger="reasoning_bank.matts"):
        _, memories = await matts_run(rollout_fn, "task", bank, k=2)
    assert memories == []
    assert any(
        "must be a json array" in rec.message.lower() for rec in caplog.records
    )


async def test_matts_invalid_json_returns_empty_memories():
    llm, _ = _make_llm(["this is not json at all"])
    bank = _bank(llm)

    async def rollout_fn():
        return [Turn(role="assistant", content="x")]

    _, memories = await matts_run(rollout_fn, "task", bank, k=2)
    assert memories == []


async def test_matts_k_less_than_one_raises():
    llm, _ = _make_llm(["[]"])
    bank = _bank(llm)

    async def rollout_fn():
        return []

    with pytest.raises(ValueError):
        await matts_run(rollout_fn, "task", bank, k=0)


async def test_matts_drops_incomplete_entries_in_payload():
    payload = json.dumps(
        [
            {"title": "ok", "description": "d", "content": "c"},
            {"title": "", "description": "d", "content": "c"},
            {"title": "also ok", "description": "d", "content": "c"},
        ]
    )
    llm, _ = _make_llm([payload])
    bank = _bank(llm)

    async def rollout_fn():
        return [Turn(role="assistant", content="x")]

    _, memories = await matts_run(rollout_fn, "task", bank, k=2)
    titles = {m.title for m in memories}
    assert titles == {"ok", "also ok"}
