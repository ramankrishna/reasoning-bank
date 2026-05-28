"""ReasoningBank end-to-end with a mock LLM + InMemoryStore."""

from __future__ import annotations

import json

import pytest

from reasoning_bank import (
    Embedder,
    InMemoryStore,
    MemoryItem,
    MemoryStore,
    ReasoningBank,
    Trajectory,
    Turn,
)


class _FakeEmbedder:
    """Embed by first character → one-hot over 6 dims (a..f)."""

    dim = 6
    _CHARS = "abcdef"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * 6
            ch = (text.strip()[:1] or "a").lower()
            if ch in self._CHARS:
                vec[self._CHARS.index(ch)] = 1.0
            out.append(vec)
        return out


def _make_llm(replies: list[str]):
    """Return an llm that yields ``replies`` in order; repeats the last forever."""

    calls = {"i": 0, "prompts": []}

    async def llm(prompt: str, *, system: str | None = None) -> str:
        calls["prompts"].append(prompt)
        i = min(calls["i"], len(replies) - 1)
        calls["i"] += 1
        return replies[i]

    return llm, calls


def _bank(llm=None) -> ReasoningBank:
    if llm is None:
        llm, _ = _make_llm(["[]"])
    return ReasoningBank(llm=llm, store=InMemoryStore(), embedder=_FakeEmbedder())


def test_public_exports():
    # Smoke: every name listed in the package's __init__ exists.
    from reasoning_bank import __all__ as exported

    for name in exported:
        from importlib import import_module

        mod = import_module("reasoning_bank")
        assert hasattr(mod, name), f"missing public export: {name}"


def test_protocol_satisfaction():
    bank = _bank()
    assert isinstance(bank.store, MemoryStore)
    assert isinstance(bank.embedder, Embedder)


def test_llm_must_be_callable():
    with pytest.raises(TypeError):
        ReasoningBank(llm="not-a-callable", store=InMemoryStore(), embedder=_FakeEmbedder())  # type: ignore[arg-type]


async def test_add_manual_then_retrieve():
    bank = _bank()
    item = await bank.add_manual(
        title="alpha lessons",
        description="things to do",
        content="content",
    )
    assert item.source == "manual"

    out = await bank.retrieve("a question about alpha", k=1)
    assert len(out) == 1
    assert out[0].id == item.id


async def test_ingest_trajectory_with_explicit_outcome():
    induction_reply = json.dumps(
        [
            {
                "title": "do thing right",
                "description": "when X happens",
                "content": "do Y",
            }
        ]
    )
    llm, calls = _make_llm([induction_reply])
    bank = ReasoningBank(
        llm=llm, store=InMemoryStore(), embedder=_FakeEmbedder()
    )
    trajectory: Trajectory = [
        Turn(role="user", content="solve"),
        Turn(role="assistant", content="solved"),
    ]
    integrated = await bank.ingest_trajectory(trajectory, "task", outcome="success")
    assert len(integrated) == 1
    assert integrated[0].title == "do thing right"
    assert integrated[0].source == "success"
    # Only one LLM call expected: induction (no judge needed because outcome given)
    assert calls["i"] == 1


async def test_ingest_trajectory_with_judge_when_outcome_none():
    judge_reply = json.dumps({"outcome": "failure", "rationale": "agent gave up"})
    induction_reply = json.dumps(
        [
            {
                "title": "lesson",
                "description": "when",
                "content": "do this",
            }
        ]
    )
    llm, calls = _make_llm([judge_reply, induction_reply])
    bank = ReasoningBank(llm=llm, store=InMemoryStore(), embedder=_FakeEmbedder())
    integrated = await bank.ingest_trajectory(
        [{"role": "assistant", "content": "x"}], "task"
    )
    assert integrated[0].source == "failure"
    assert calls["i"] == 2


async def test_ingest_trajectory_accepts_dict_input():
    induction_reply = json.dumps(
        [{"title": "t", "description": "d", "content": "c"}]
    )
    llm, _ = _make_llm([induction_reply])
    bank = ReasoningBank(llm=llm, store=InMemoryStore(), embedder=_FakeEmbedder())
    out = await bank.ingest_trajectory(
        [{"role": "user", "content": "hi"}], "task", outcome="success"
    )
    assert len(out) == 1


async def test_ingest_trajectory_empty_returns_empty():
    bank = _bank()
    out = await bank.ingest_trajectory([], "task", outcome="success")
    assert out == []


async def test_list_and_delete():
    bank = _bank()
    a = await bank.add_manual("a", "d", "c")
    b = await bank.add_manual("b", "d", "c")
    listed = await bank.list()
    assert {it.id for it in listed} == {a.id, b.id}
    await bank.delete(a.id)
    listed = await bank.list()
    assert {it.id for it in listed} == {b.id}


async def test_export_import_round_trip(tmp_path):
    src = _bank()
    a = await src.add_manual("a", "d", "c")
    b = await src.add_manual("b", "d", "c")

    out_path = tmp_path / "export.jsonl"
    await src.export(out_path)
    assert out_path.exists()
    lines = out_path.read_text().splitlines()
    assert len(lines) == 2

    dst = _bank()
    count = await dst.import_(out_path)
    assert count == 2
    listed = await dst.list()
    assert {it.id for it in listed} == {a.id, b.id}


async def test_export_import_round_trip_overwrites_duplicates(tmp_path):
    """Importing into a bank that already has the same ids should not raise."""
    src = _bank()
    a = await src.add_manual("a", "d", "c")

    out_path = tmp_path / "export.jsonl"
    await src.export(out_path)
    count = await src.import_(out_path)
    assert count == 1
    listed = await src.list()
    assert len(listed) == 1
    assert listed[0].id == a.id


def test_format_as_system_block_empty_is_empty():
    bank = _bank()
    assert bank.format_as_system_block([]) == ""


def test_format_as_system_block_includes_title_and_content():
    bank = _bank()
    items = [
        MemoryItem(
            title="lesson",
            description="when X",
            content="do Y",
            source="manual",
            task_signature="",
        )
    ]
    out = bank.format_as_system_block(items)
    assert "lesson" in out
    assert "when X" in out
    assert "do Y" in out
    assert out.startswith("# Reasoning bank")
