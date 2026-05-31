"""Phase 3 — calibration. Run all tasks once, bank-off, fresh sandbox per task.

Decision gate (printed at the end):
  - gotcha-hit rate > 30%       -> proceed to Phase 4
  - gotcha-hit rate == 0        -> ceiling (tasks too easy)  -> STOP
  - success rate < ~25%         -> floor (tasks too hard)    -> STOP
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from anthropic import AsyncAnthropic

from experiments.agentic.runner import dump_results, run_one, summarize
from experiments.agentic.tasks import TASKS

RESULTS_DIR = Path(__file__).resolve().parent / "results"


async def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    client = AsyncAnthropic()
    results = []
    for task in TASKS:
        print(f"[calibrate] running {task.id} (gotchas={sorted(task.gotchas)})")
        r = await run_one(task, arm="calibration", client=client)
        results.append(r)
        print(
            f"  success={r.success} steps={r.steps} hits={r.gotchas_hit}"
            f" err={r.error or ''}"
        )

    dump_results(results, RESULTS_DIR / "calibration.json")
    s = summarize(results)
    (RESULTS_DIR / "calibration_summary.json").write_text(
        json.dumps(s, indent=2)
    )
    print("\n=== CALIBRATION SUMMARY ===")
    print(json.dumps(s, indent=2))

    # Decision gate
    if s["any_gotcha_hit_rate"] == 0:
        verdict = "CEILING: agent hit no gotchas — tasks too easy"
    elif s["success_rate"] < 0.25:
        verdict = "FLOOR: agent failed most tasks — tasks too hard"
    elif s["any_gotcha_hit_rate"] > 0.30:
        verdict = "PROCEED: headroom present (gotcha-hit rate > 30%)"
    else:
        verdict = "MARGINAL: gotcha-hit rate is positive but <=30%"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
