"""Phase 4 — three-arm experiment.

Arm 1: bank OFF                — baseline
Arm 2: bank ON                 — retrieve, inject as system block, ingest after
Arm 3: bank ON + retrieval log — same as Arm 2, also records retrieved titles

Each (task, arm) pair gets a fresh tempfile.mkdtemp() sandbox seeded by
task.setup(). Independent InMemoryStore per arm. Temp 0.0 throughout.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

from anthropic import AsyncAnthropic

from reasoning_bank import InMemoryStore, ReasoningBank, Turn

from experiments.agentic.runner import RunResult, dump_results, run_one, summarize
from experiments.agentic.tasks import TASKS

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# LLM callable that the bank uses internally (judge / distill / merge).
# ---------------------------------------------------------------------------


def _bank_llm(client: AsyncAnthropic):
    async def llm(prompt: str, *, system: str | None = None) -> str:
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return msg.content[0].text
    return llm


# ---------------------------------------------------------------------------
# Memory injection block
# ---------------------------------------------------------------------------


def _format_memories(items) -> str:
    if not items:
        return ""
    lines = ["Lessons from prior similar tasks (apply them if relevant):"]
    for i, it in enumerate(items, 1):
        title = it.title or "(untitled)"
        body = it.content[:400] if hasattr(it, "content") else ""
        lines.append(f"  {i}. {title}\n     {body}")
    return "\n".join(lines)


def _traj_to_turns(trajectory: list[dict]) -> list[Turn]:
    return [
        Turn(
            role=t["role"],
            content=str(t.get("content", "")),
            meta=t.get("meta", {}),
        )
        for t in trajectory
    ]


# ---------------------------------------------------------------------------
# Arm runners
# ---------------------------------------------------------------------------


async def run_arm_off(client: AsyncAnthropic) -> list[RunResult]:
    out: list[RunResult] = []
    for task in TASKS:
        print(f"[arm1-off] {task.id}")
        r = await run_one(task, arm="bank_off", client=client)
        print(f"  success={r.success} steps={r.steps} hits={r.gotchas_hit}")
        out.append(r)
    return out


async def run_arm_on(
    client: AsyncAnthropic,
    *,
    arm_name: str,
    log_retrieval: bool,
) -> tuple[list[RunResult], list[dict]]:
    bank = ReasoningBank(
        llm=_bank_llm(client),
        store=InMemoryStore(),
        scope=f"agentic-{arm_name}",
    )
    out: list[RunResult] = []
    retrieval_log: list[dict] = []
    for task in TASKS:
        print(f"[{arm_name}] {task.id}")
        # retrieve
        retrieved = await bank.retrieve(task.prompt, k=5, min_confidence=0.0)
        sys_block = _format_memories(retrieved)
        retrieved_titles = [getattr(x, "title", "") or "" for x in retrieved]
        if log_retrieval:
            retrieval_log.append({
                "task_id": task.id,
                "n_retrieved": len(retrieved),
                "titles": retrieved_titles,
            })
        # run
        r = await run_one(
            task,
            arm=arm_name,
            client=client,
            system_block=sys_block or None,
            retrieved_titles=retrieved_titles,
        )
        print(
            f"  success={r.success} steps={r.steps} hits={r.gotchas_hit}"
            f" retrieved={len(retrieved)}"
        )
        out.append(r)
        # ingest — judge automatically based on trajectory
        try:
            await bank.ingest_trajectory(
                _traj_to_turns(r.trajectory),
                task=task.prompt,
                outcome="success" if r.success else "failure",
            )
        except Exception as e:
            print(f"  [ingest error] {type(e).__name__}: {e}")
    return out, retrieval_log


# ---------------------------------------------------------------------------
# Repeated-error metric
# ---------------------------------------------------------------------------


def repeated_error_count(results: list[RunResult]) -> int:
    """Count gotcha hits that occurred AFTER an earlier task in the same arm
    had hit the same family. The bank had a chance to learn — did it?"""
    seen_families: set[str] = set()
    repeats = 0
    for r in results:
        for fam in r.gotchas_hit:
            if fam in seen_families:
                repeats += 1
        seen_families.update(r.gotchas_hit)
    return repeats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    client = AsyncAnthropic()

    t0 = time.time()
    print("\n=== ARM 1: bank OFF ===")
    arm1 = await run_arm_off(client)
    dump_results(arm1, RESULTS_DIR / "arm1_off.json")

    print("\n=== ARM 2: bank ON ===")
    arm2, _ = await run_arm_on(client, arm_name="bank_on", log_retrieval=False)
    dump_results(arm2, RESULTS_DIR / "arm2_on.json")

    print("\n=== ARM 3: bank ON + retrieval log ===")
    arm3, retrieval_log = await run_arm_on(
        client, arm_name="bank_on_logged", log_retrieval=True
    )
    dump_results(arm3, RESULTS_DIR / "arm3_on_logged.json")
    (RESULTS_DIR / "arm3_retrieval_log.json").write_text(
        json.dumps(retrieval_log, indent=2)
    )

    elapsed = time.time() - t0
    summary = {
        "elapsed_seconds": round(elapsed, 1),
        "arm1_off": {
            **summarize(arm1),
            "repeated_error_count": repeated_error_count(arm1),
        },
        "arm2_on": {
            **summarize(arm2),
            "repeated_error_count": repeated_error_count(arm2),
        },
        "arm3_on_logged": {
            **summarize(arm3),
            "repeated_error_count": repeated_error_count(arm3),
        },
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
