"""Calibration run for the ReasoningBank experiment.

Goal: prove the puzzle suite has the right difficulty level for Haiku 4.5 at
temp 0.0 with no bank. Target band: 30-70% first-attempt accuracy.

- > 90% → puzzles too easy (ceiling). Don't run the experiment until fixed.
- < 10% → puzzles too hard (floor). Don't run the experiment until fixed.

Output: experiments/results/calibration.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from anthropic import AsyncAnthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.puzzles import PUZZLES, verify  # noqa: E402

MODEL = "claude-haiku-4-5-20251001"
TEMP = 0.0
MAX_TOKENS = 1024
SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Think step by step, but finish "
    "your response with a single line beginning with 'ANSWER:' followed by "
    "your final answer (a number or short string only — no units, no "
    "explanation on that line)."
)

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_PATH = OUT_DIR / "calibration.json"


def _extract_final_answer(text: str) -> str:
    """Pull whatever follows 'ANSWER:' on the last such line. Falls back to
    the last non-empty line of the response."""
    final = ""
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("ANSWER:"):
            final = s[len("ANSWER:") :].strip()
    if final:
        return final
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s:
            return s
    return ""


async def _ask(client: AsyncAnthropic, question: str) -> str:
    msg = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMP,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    parts = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = AsyncAnthropic()
    results = []
    correct = 0
    started = time.time()

    # Limit concurrency to avoid the API's per-account concurrent-connection cap.
    sem = asyncio.Semaphore(3)

    async def run_one(p):  # type: ignore[no-untyped-def]
        async with sem:
            t0 = time.time()
            try:
                raw = await _ask(client, p.question)
                err = None
            except Exception as exc:  # noqa: BLE001
                raw = ""
                err = f"{type(exc).__name__}: {exc}"
            answer = _extract_final_answer(raw)
            ok = bool(answer) and verify(p.id, answer)
            return {
                "id": p.id,
                "family": p.family,
                "expected": str(p.expected),
                "answer": answer,
                "correct": ok,
                "raw": raw,
                "error": err,
                "elapsed_s": round(time.time() - t0, 2),
            }

    rows = await asyncio.gather(*(run_one(p) for p in PUZZLES))
    for row in rows:
        results.append(row)
        correct += int(row["correct"])
        mark = "PASS" if row["correct"] else "FAIL"
        print(
            f"[{mark}] {row['id']:>22}  "
            f"expected={row['expected']:>8}  got={row['answer']!r}  "
            f"({row['elapsed_s']}s)"
        )

    n = len(results)
    rate = correct / n if n else 0.0
    elapsed = round(time.time() - started, 2)
    if rate > 0.9:
        verdict = "TOO_EASY"
    elif rate < 0.1:
        verdict = "TOO_HARD"
    else:
        verdict = "OK"

    summary = {
        "model": MODEL,
        "temperature": TEMP,
        "n": n,
        "correct": correct,
        "accuracy": rate,
        "verdict": verdict,
        "wall_clock_s": elapsed,
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print()
    print(
        f"Calibration: {correct}/{n} = {rate:.1%}  verdict={verdict}  "
        f"({elapsed}s)"
    )
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
