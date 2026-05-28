# reasoning-bank

> Framework-agnostic implementation of ReasoningBank (Ouyang et al., ICLR 2026).
> Agents that learn from successful AND failed trajectories. Works with any agent loop.

[![PyPI](https://img.shields.io/pypi/v/reasoning-bank)](https://pypi.org/project/reasoning-bank/)
[![Python](https://img.shields.io/pypi/pyversions/reasoning-bank)](https://pypi.org/project/reasoning-bank/)
[![License](https://img.shields.io/github/license/ramankrishna/reasoning-bank)](LICENSE)

## What it is

ReasoningBank distills generalizable reasoning strategies from past agent
trajectories — both successes and failures — and retrieves them on future tasks.
Memory items are distilled *strategies*, not raw traces. Includes **MaTTS**
(memory-aware test-time scaling): run k rollouts in parallel, contrast them,
distill higher-quality memories.

Unlike the reference implementation (welded to WebArena/SWE-bench), this works
with **any** agent loop — raw Anthropic/OpenAI SDK, LangGraph, CrewAI, or your own.

## Install

```bash
pip install reasoning-bank
```

## 60-second example

```python
import asyncio
from anthropic import AsyncAnthropic
from reasoning_bank import ReasoningBank, Turn

client = AsyncAnthropic()

async def llm(prompt, *, system=None):
    msg = await client.messages.create(
        model="claude-sonnet-4-6", max_tokens=2000,
        system=system or "You are a helpful assistant.",
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

async def main():
    bank = ReasoningBank(llm=llm, scope="demo")
    task = "What's an efficient way to find the median of a stream?"

    # 1. Retrieve past learnings
    mems = await bank.retrieve(task, k=3)
    system = bank.format_as_system_block(mems) if mems else None

    # 2. Run your agent
    answer = await llm(task, system=system)
    trajectory = [Turn("user", task), Turn("assistant", answer)]

    # 3. Learn from the run
    await bank.ingest_trajectory(trajectory, task=task)

asyncio.run(main())
```

## How it plugs into your loop

Three calls:
1. `mems = await bank.retrieve(task)` — before your agent runs
2. Run your agent (optionally prepend `bank.format_as_system_block(mems)`)
3. `await bank.ingest_trajectory(trajectory, task)` — after, to learn

## MaTTS (memory-aware test-time scaling)

```python
from reasoning_bank import ReasoningBank, Turn, matts_run

async def rollout():
    answer = await llm(task)
    return [Turn("user", task), Turn("assistant", answer)]

trajectories, memories = await matts_run(rollout, task=task, bank=bank, k=4)
```

## Design

- **Neutral types**: pass any trajectory — `Turn` objects, dicts, or duck-typed objects with `.role`/`.content`
- **Neutral LLM**: provide an `async def llm(prompt, *, system=None) -> str`
- **Pluggable store** (SQLite+vec default) and **embedder** (MiniLM default)

## Paper

Ouyang et al., *ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory*, ICLR 2026.
[arXiv:2509.25140](https://arxiv.org/abs/2509.25140)

## Related

Used as the memory layer in [bottensor-fleet](https://github.com/ramankrishna/bottensor-fleet).

## License

Apache-2.0
