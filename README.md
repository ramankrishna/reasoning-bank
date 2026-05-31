# reasoning-bank

> A framework-agnostic implementation of ReasoningBank (Ouyang et al., ICLR 2026): agents that distill lessons from past trajectories and retrieve them on new tasks. Includes MaTTS (memory-aware test-time scaling).

[![PyPI](https://img.shields.io/pypi/v/reasoning-bank.svg)](https://pypi.org/project/reasoning-bank/)
[![Python](https://img.shields.io/pypi/pyversions/reasoning-bank.svg)](https://pypi.org/project/reasoning-bank/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2509.25140-b31b1b.svg)](https://arxiv.org/abs/2509.25140)

## What this is

ReasoningBank is a memory mechanism for LLM agents. After each task, it judges the trajectory, distills generalizable reasoning strategies from both successes and failures, and indexes them. On future tasks it retrieves the relevant strategies and injects them into the agent's context.

This package is a clean, framework-agnostic implementation. It works with any agent loop, the raw Anthropic or OpenAI SDK, or your own ReAct loop. It does not depend on any agent framework. (It is used as the memory layer in [bottensor-fleet](https://github.com/ramankrishna/bottensor-fleet), but does not require it.)

The reference implementation from the paper is welded to specific benchmark harnesses. This one is a standalone library you can attach to anything.

## Install

```
pip install reasoning-bank
```

## How it plugs into any agent loop

Three calls:

```python
from reasoning_bank import ReasoningBank, Turn

async def my_llm(prompt, *, system=None):
    # wrap your provider however you like
    ...

bank = ReasoningBank(llm=my_llm, scope="my-project")

# 1. retrieve relevant past lessons before your agent runs
memories = await bank.retrieve(task)
system_block = bank.format_as_system_block(memories) if memories else None

# 2. run your agent however you want, optionally prepending system_block
answer = await my_agent(task, system=system_block)

# 3. ingest the trajectory so the bank learns from it
trajectory = [Turn("user", task), Turn("assistant", answer)]
await bank.ingest_trajectory(trajectory, task=task)
```

The bank accepts any trajectory shape (its own `Turn` type, plain dicts, or duck-typed objects with `.role`/`.content`) and any async LLM callable. Pluggable embedder (MiniLM default) and store (SQLite+vec default, in-memory for tests).

## MaTTS

Memory-aware test-time scaling: run k rollouts in parallel, contrast them, distill higher-quality memories.

```python
from reasoning_bank import matts_run

async def rollout():
    answer = await my_agent(task)
    return [Turn("user", task), Turn("assistant", answer)]

trajectories, memories = await matts_run(rollout, task=task, bank=bank, k=3)
```

## Does it actually work?

This is the honest part, and the reason the repo includes a full experiment suite.

The machinery works end-to-end: it judges trajectories, distills sensible transferable lessons, retrieves them, and injects them, with no framework dependency. That is verified.

Whether it produces a measurable *performance lift* is a separate question, and the answer depends heavily on the task distribution and the model. I ran four controlled experiments to find out, including a SWE-bench-lite harness with a no-retry control, a naive-retry control, a per-instance bank, and a persistent cross-instance bank, scored by the official SWE-bench Docker evaluator.

Headline findings (Haiku 4.5, full methodology in [`experiments/`](experiments/)):

- On task suites where the base model is already at its capability ceiling, the bank cannot help, because there is no failure to learn from. Two early experiments hit this.
- On SWE-bench-lite (a real spread of difficulty, ~50% baseline), the persistent cross-instance bank produced **no measurable lift** over a no-retry baseline. The cross-task transfer hypothesis was not supported in this setup (n=45 clean cells on the persistent arm; late instances did not outperform early ones).
- The one consistently positive observation was **defensive**: naive retry (re-prompting with the raw error) *hurt* performance, and structured-reflection retry recovered it. The value was in how a failure is framed to the model, not in accumulating a memory bank.

In short: the implementation is faithful and the machinery is sound, but I did not find a regime in these experiments where cross-task memory accumulation produced net positive value. The full data, including infrastructure caveats and threats to validity, is in the experiments directory. Negative results are results.

## Design

- **Framework-agnostic:** neutral trajectory type and a plain async LLM callable at every boundary. No agent framework required.
- **Pluggable store:** SQLite + sqlite-vec by default; in-memory for tests.
- **Pluggable embedder:** sentence-transformers MiniLM by default.
- **Memory items are distilled strategies, not raw traces**, per the paper: a title, a description, and an actionable lesson, derived from both successes and failures.

## Paper

Ouyang et al., *ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory*, ICLR 2026. [arXiv:2509.25140](https://arxiv.org/abs/2509.25140)

This package is an independent implementation and is not affiliated with the paper's authors.

## License

Apache-2.0. Copyright 2026 Rama Krishna Bachu.
