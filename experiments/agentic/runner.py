"""Per-task runner with sandbox lifecycle. Shared by calibration and the
three-arm experiment."""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

from experiments.agentic.agent import run_agent
from experiments.agentic.tasks import Task, detect_gotcha_errors
from experiments.agentic.tools import Sandbox


@dataclass
class RunResult:
    task_id: str
    arm: str
    success: bool
    steps: int
    final_answer: str
    gotchas_target: list[str]
    gotchas_hit: list[str]
    trajectory: list[dict] = field(default_factory=list)
    retrieved_titles: list[str] = field(default_factory=list)
    error: str | None = None


def _count_steps(trajectory: list[dict]) -> int:
    return sum(1 for t in trajectory if t.get("role") == "tool")


async def run_one(
    task: Task,
    arm: str,
    *,
    client: AsyncAnthropic,
    system_block: str | None = None,
    retrieved_titles: list[str] | None = None,
    max_steps: int = 10,
) -> RunResult:
    """Run a single task in a fresh sandbox. Always destroys the sandbox."""
    sandbox_dir = Path(tempfile.mkdtemp(prefix=f"agentic_{task.id}_"))
    try:
        task.setup(sandbox_dir)
        sb = Sandbox(str(sandbox_dir))
        try:
            final, traj = await run_agent(
                task.prompt,
                sb,
                system_block=system_block,
                max_steps=max_steps,
                client=client,
            )
            ok = bool(task.verify(final, sandbox_dir, traj))
            hits = sorted(detect_gotcha_errors(traj, final))
            return RunResult(
                task_id=task.id,
                arm=arm,
                success=ok,
                steps=_count_steps(traj),
                final_answer=final,
                gotchas_target=sorted(task.gotchas),
                gotchas_hit=hits,
                trajectory=traj,
                retrieved_titles=retrieved_titles or [],
            )
        except Exception as e:
            return RunResult(
                task_id=task.id,
                arm=arm,
                success=False,
                steps=0,
                final_answer="",
                gotchas_target=sorted(task.gotchas),
                gotchas_hit=[],
                trajectory=[],
                retrieved_titles=retrieved_titles or [],
                error=f"{type(e).__name__}: {e}",
            )
    finally:
        shutil.rmtree(sandbox_dir, ignore_errors=True)


def dump_results(results: list[RunResult], path: Path) -> None:
    path.write_text(
        json.dumps([asdict(r) for r in results], indent=2, default=str)
    )


def summarize(results: list[RunResult]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n": 0}
    success = sum(1 for r in results if r.success)
    mean_steps = sum(r.steps for r in results) / n
    any_gotcha = sum(1 for r in results if r.gotchas_hit)
    hits_by_family: dict[str, int] = {}
    for r in results:
        for g in r.gotchas_hit:
            hits_by_family[g] = hits_by_family.get(g, 0) + 1
    return {
        "n": n,
        "success": success,
        "success_rate": success / n,
        "mean_steps": mean_steps,
        "any_gotcha_hit_rate": any_gotcha / n,
        "hits_by_family": hits_by_family,
    }
