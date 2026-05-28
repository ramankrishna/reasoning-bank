# Quickstart

## Install

```bash
pip install reasoning-bank
```

The default install pulls in `sentence-transformers` (for the MiniLM embedder)
and `sqlite-vec` (for the local vector store). Python 3.11+ required.

## The three-call loop

Every integration boils down to:

```python
# 1. Retrieve relevant past learnings
mems = await bank.retrieve(task, k=3)

# 2. Run your agent (any framework, any loop)
system = bank.format_as_system_block(mems) if mems else None
answer = await your_agent(task, system=system)

# 3. Ingest the resulting trajectory
trajectory = [Turn("user", task), Turn("assistant", answer)]
await bank.ingest_trajectory(trajectory, task=task)
```

Step 3 is where the bank does its work: it judges the trajectory as success
or failure, distills generalizable strategies (different prompts for each
outcome), embeds them, merges with similar existing memories, and writes to
the store.

## Providing the LLM

`ReasoningBank` takes a neutral async callable. Anything with this signature
works:

```python
async def llm(prompt: str, *, system: str | None = None) -> str: ...
```

### With Anthropic

```python
from anthropic import AsyncAnthropic
client = AsyncAnthropic()

async def llm(prompt, *, system=None):
    msg = await client.messages.create(
        model="claude-sonnet-4-6", max_tokens=2000,
        system=system or "You are a helpful assistant.",
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text
```

### With OpenAI

```python
from openai import AsyncOpenAI
client = AsyncOpenAI()

async def llm(prompt, *, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await client.chat.completions.create(
        model="gpt-4o-mini", messages=messages,
    )
    return resp.choices[0].message.content
```

### With your own loop

The example at `examples/generic_loop.py` uses a deterministic stub LLM, so
you can see the full retrieve → run → ingest cycle with no API key.

## MaTTS — memory-aware test-time scaling

`matts_run` runs k rollouts in parallel, then asks the LLM to contrast them
and distill higher-quality memories than any single trajectory could yield.

```python
from reasoning_bank import matts_run, Turn

async def rollout():
    answer = await llm(task)
    return [Turn("user", task), Turn("assistant", answer)]

trajectories, memories = await matts_run(
    rollout, task=task, bank=bank, k=4,
)
```

`rollout_fn` is *yours* — it can do whatever your agent does (tool calls,
multi-turn reasoning, search). MaTTS doesn't care; it only sees the
trajectories you return.

## Trajectory formats

`ingest_trajectory` accepts:

- `list[Turn]` — `Turn("user", "...")`
- `list[dict]` — `{"role": "user", "content": "..."}`
- Anything with `.role` / `.content` attributes — coerced automatically

This means you can pass the message list straight out of most agent frameworks
without conversion.

## Scope

`scope` partitions memories. Use it to keep different tasks, agents, or
environments from polluting each other's retrieval:

```python
bank = ReasoningBank(llm=llm, scope="my-agent/web-tasks")
```

## Where memory lives

By default, memories are stored in a SQLite database at `./reasoning-bank.db`
in the current working directory. Override with the `store` argument to
`ReasoningBank` (see `concepts.md` for details).
