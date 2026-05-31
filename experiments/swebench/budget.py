"""Rolling 60-second token-budget governor.

The org's Anthropic rate limit is 50k input-tokens/min. We aim to stay under
45k/min so a single noisy step doesn't tip us over.

Usage::

    budget = TokenBudget(limit_per_min=45000)
    await budget.reserve(est_tokens)        # blocks until safe
    resp = await client.messages.create(...)
    budget.record(resp.usage.input_tokens)  # log actual after the call

The "reserve" step blocks until the rolling 60s sum of recorded tokens plus
`est_tokens` would fit under the limit. We use the agent's last observed
step usage as the estimate for the next step (kept on the governor).
"""
from __future__ import annotations

import asyncio
import time
from collections import deque


class TokenBudget:
    def __init__(self, limit_per_min: int = 45_000, window_s: float = 60.0):
        self.limit = limit_per_min
        self.window_s = window_s
        self._events: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()
        self.last_recorded: int = 0
        self.total_recorded: int = 0
        self.n_records: int = 0
        self.pause_count: int = 0
        self.total_pause_s: float = 0.0

    def _prune(self, now: float) -> int:
        while self._events and (now - self._events[0][0]) > self.window_s:
            self._events.popleft()
        return sum(t for _, t in self._events)

    async def reserve(self, est_tokens: int) -> None:
        est_tokens = max(0, int(est_tokens))
        while True:
            async with self._lock:
                now = time.monotonic()
                in_window = self._prune(now)
                if in_window + est_tokens <= self.limit:
                    return
                # Wait until the oldest event drops out of the window, then re-check.
                oldest_t = self._events[0][0]
                wait_s = (oldest_t + self.window_s) - now + 0.1
            wait_s = max(0.5, min(wait_s, 30.0))
            self.pause_count += 1
            self.total_pause_s += wait_s
            print(
                f"  [budget] pausing {wait_s:.1f}s "
                f"(in_window={in_window}, est={est_tokens}, limit={self.limit})",
                flush=True,
            )
            await asyncio.sleep(wait_s)

    def record(self, actual_tokens: int) -> None:
        actual_tokens = max(0, int(actual_tokens))
        self._events.append((time.monotonic(), actual_tokens))
        self.last_recorded = actual_tokens
        self.total_recorded += actual_tokens
        self.n_records += 1

    def stats(self) -> dict:
        return {
            "total_input_tokens": self.total_recorded,
            "n_calls": self.n_records,
            "pause_count": self.pause_count,
            "total_pause_s": round(self.total_pause_s, 2),
            "limit_per_min": self.limit,
        }
