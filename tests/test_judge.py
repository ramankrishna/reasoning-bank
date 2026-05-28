"""judge_trajectory with a mock async LLM callable."""

from __future__ import annotations

import json

import pytest

from reasoning_bank.judge import judge_trajectory
from reasoning_bank.types import Turn


def _llm_returning(reply: str):
    async def llm(prompt: str, *, system: str | None = None) -> str:
        return reply

    return llm


async def test_judge_success():
    llm = _llm_returning(
        json.dumps({"outcome": "success", "rationale": "the agent solved it"})
    )
    outcome, rationale = await judge_trajectory(
        [Turn(role="assistant", content="done")], "t", llm
    )
    assert outcome == "success"
    assert "solved" in rationale


async def test_judge_failure():
    llm = _llm_returning(
        json.dumps({"outcome": "failure", "rationale": "agent gave up"})
    )
    outcome, _ = await judge_trajectory(
        [Turn(role="assistant", content="x")], "t", llm
    )
    assert outcome == "failure"


async def test_judge_invalid_outcome_raises():
    llm = _llm_returning(
        json.dumps({"outcome": "maybe", "rationale": "unclear"})
    )
    with pytest.raises(ValueError):
        await judge_trajectory([Turn(role="user", content="x")], "t", llm)


async def test_judge_non_object_payload_raises():
    llm = _llm_returning(json.dumps(["arr"]))
    with pytest.raises(ValueError):
        await judge_trajectory([Turn(role="user", content="x")], "t", llm)


async def test_judge_empty_response_raises():
    llm = _llm_returning("")
    with pytest.raises(ValueError):
        await judge_trajectory([Turn(role="user", content="x")], "t", llm)
