"""ReasoningBank with a toy agent loop and a mock LLM. No provider key needed.

Run: python examples/generic_loop.py
"""

from __future__ import annotations

import asyncio
import hashlib

from reasoning_bank import InMemoryStore, ReasoningBank, Turn


class HashEmbedder:
    """Tiny deterministic embedder so the demo runs fully offline.

    Real deployments should use ``reasoning_bank.embedders.MiniLMEmbedder``
    or another semantic embedder; this hash-bucket trick is fine for a smoke
    demo where we just need stable vectors of a fixed dimension.
    """

    dim = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in text.lower().split():
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


async def mock_llm(prompt: str, *, system: str | None = None) -> str:
    if "judge" in prompt.lower():
        return '{"outcome":"success","rationale":"reached the goal"}'
    return (
        '[{"title":"Check units before computing",'
        '"description":"When a task mixes units, normalize first",'
        '"content":"Always convert to base units before arithmetic"}]'
    )


async def main() -> None:
    bank = ReasoningBank(
        llm=mock_llm, store=InMemoryStore(), embedder=HashEmbedder()
    )
    task = "Compute average speed for a 150-mile trip in 2.5 hours"
    trajectory = [
        Turn("user", task),
        Turn("assistant", "150 miles / 2.5 hours = 60 mph"),
    ]
    mems = await bank.retrieve(task, k=3)
    print(f"Retrieved {len(mems)} memories")
    new = await bank.ingest_trajectory(trajectory, task=task, outcome="success")
    print(f"Distilled {len(new)} new memories:")
    for m in new:
        print(f"  - {m.title}")
    mems2 = await bank.retrieve("Compute speed for a road trip", k=3)
    print(f"Retrieved {len(mems2)} on a related task")


asyncio.run(main())
