"""ReasoningBank wrapping the raw Anthropic SDK. No fleet, no framework.

Requires: pip install anthropic
Run: ANTHROPIC_API_KEY=sk-ant-... python examples/raw_anthropic.py
"""

from __future__ import annotations

import asyncio

from anthropic import AsyncAnthropic

from reasoning_bank import ReasoningBank, Turn

client = AsyncAnthropic()


async def anthropic_llm(prompt: str, *, system: str | None = None) -> str:
    msg = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=system or "You are a helpful assistant.",
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


async def main() -> None:
    bank = ReasoningBank(llm=anthropic_llm, scope="demo")
    task = "What's an efficient way to find the median of a stream of numbers?"
    mems = await bank.retrieve(task, k=3)
    system = bank.format_as_system_block(mems) if mems else None
    answer = await anthropic_llm(task, system=system)
    trajectory = [Turn("user", task), Turn("assistant", answer)]
    print("Answer:", answer[:200])
    new = await bank.ingest_trajectory(trajectory, task=task)
    print(f"Distilled {len(new)} memories")


asyncio.run(main())
