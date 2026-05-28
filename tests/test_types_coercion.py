"""Verify that ``_coerce_trajectory`` accepts Turn/dict/duck-typed inputs."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from reasoning_bank.types import Trajectory, Turn, _coerce_trajectory, render_trajectory


@dataclass
class FakeAgentMessage:
    """Stand-in for fleet.core.messages.AgentMessage."""

    role: str
    content: str
    tool_calls: list | None = None
    tool_results: list | None = None


def _expected() -> Trajectory:
    return [
        Turn(role="user", content="hi"),
        Turn(role="assistant", content="hello"),
    ]


def test_coerces_list_of_turns_identity():
    turns = _expected()
    out = _coerce_trajectory(turns)
    assert all(isinstance(t, Turn) for t in out)
    assert [(t.role, t.content) for t in out] == [
        ("user", "hi"),
        ("assistant", "hello"),
    ]


def test_coerces_list_of_dicts():
    out = _coerce_trajectory(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert [(t.role, t.content) for t in out] == [
        ("user", "hi"),
        ("assistant", "hello"),
    ]


def test_coerces_duck_typed_objects():
    msgs = [
        FakeAgentMessage(role="user", content="hi"),
        FakeAgentMessage(role="assistant", content="hello"),
    ]
    out = _coerce_trajectory(msgs)
    assert [(t.role, t.content) for t in out] == [
        ("user", "hi"),
        ("assistant", "hello"),
    ]


def test_dict_extra_keys_land_in_meta():
    out = _coerce_trajectory([{"role": "user", "content": "hi", "extra": 7}])
    assert out[0].meta == {"extra": 7}


def test_duck_typed_tool_calls_preserved_in_meta():
    msgs = [
        FakeAgentMessage(
            role="assistant",
            content="calling tool",
            tool_calls=[{"name": "search", "arguments": {"q": "x"}}],
        )
    ]
    out = _coerce_trajectory(msgs)
    assert out[0].meta.get("tool_calls") == [
        {"name": "search", "arguments": {"q": "x"}}
    ]


def test_none_returns_empty_list():
    assert _coerce_trajectory(None) == []


def test_invalid_role_raises():
    with pytest.raises(ValueError):
        _coerce_trajectory([{"role": "wizard", "content": "x"}])


def test_missing_role_dict_raises():
    with pytest.raises(ValueError):
        _coerce_trajectory([{"content": "x"}])


def test_object_missing_role_raises():
    class NoRole:
        content = "x"

    with pytest.raises(ValueError):
        _coerce_trajectory([NoRole()])


def test_render_trajectory_three_forms_match():
    from_turns = render_trajectory(_expected())
    from_dicts = render_trajectory(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    from_objs = render_trajectory(
        [
            FakeAgentMessage(role="user", content="hi"),
            FakeAgentMessage(role="assistant", content="hello"),
        ]
    )
    assert from_turns == from_dicts == from_objs == "[user] hi\n[assistant] hello"


def test_render_trajectory_includes_tool_calls():
    msgs = [
        FakeAgentMessage(
            role="assistant",
            content="thinking",
            tool_calls=[{"name": "search", "arguments": "x"}],
            tool_results=[{"content": "ok"}, {"content": "boom", "is_error": True}],
        )
    ]
    out = render_trajectory(msgs)
    assert "tool_call: search(x)" in out
    assert "tool_result: ok" in out
    assert "tool_error: boom" in out
