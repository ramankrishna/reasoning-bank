"""MaTTS with raw Anthropic. Runs k rollouts in parallel, contrast-distills.

Run: ANTHROPIC_API_KEY=sk-ant-... python examples/matts_anthropic.py
"""

from __future__ import annotations

import asyncio

from anthropic import AsyncAnthropic

from reasoning_bank import ReasoningBank, Turn, matts_run

client = AsyncAnthropic()


async def llm(prompt: str, *, system: str | None = None) -> str:
    msg = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=system or "",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return msg.content[0].text


async def main() -> None:
    bank = ReasoningBank(llm=llm, scope="matts-demo")
    task = "Prove that the square root of 2 is irrational."

    async def rollout() -> list[Turn]:
        answer = await llm(task)
        return [Turn("user", task), Turn("assistant", answer)]

    trajectories, memories = await matts_run(rollout, task=task, bank=bank, k=3)
    print(
        f"Ran {len(trajectories)} rollouts, "
        f"distilled {len(memories)} contrast memories"
    )
    for m in memories:
        print(f"  - {m.title}")


asyncio.run(main())
