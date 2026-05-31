"""Phase 3 arm logic for the resumable driver.

Interface: each arm function takes (instance, *, seed, budget, client, workdir,
run_id, bank=None) and returns a result dict with keys:

    arm:           one of arm1_noretry, arm2_naive, arm3a_bank_fresh,
                   arm3b_bank_persist
    instance_id:   the SWE-bench instance id
    seed:          the trial seed
    attempts:      how many agent runs were made (1..MAX_ATTEMPTS)
    resolved:      True iff any attempt passed the swebench eval
    first_pass:    True iff attempt 1 passed
    final_pass:    same as resolved (alias for clarity in analysis)
    patches:       list of per-attempt patches (truncated for storage)
    per_attempt:   list of {steps, input_tokens, output_tokens, resolved, error?}
    retrieved_pre: memories retrieved before attempt 1 (bank arms only)
    retrieved_retry: memories retrieved before attempt 2 (bank arms only)

The retry arms (2, 3a, 3b) run UP TO MAX_ATTEMPTS=3 agent runs and stop as
soon as one passes. Arm 1 always runs exactly once.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

from experiments.swebench.agent import RepoSandbox, run_agent
from experiments.swebench.harness import (
    prepare_workdir, extract_patch, write_predictions, run_swebench_eval,
)
from reasoning_bank import ReasoningBank
from reasoning_bank.types import Turn

MAX_ATTEMPTS = 3
PATCH_TRIM = 4000  # bytes of patch we save per attempt in the ledger


# --------- helpers ---------

def _traj_to_turns(trajectory: list[dict]) -> list[Turn]:
    return [
        Turn(role=t["role"], content=str(t.get("content", "")), meta=t.get("meta", {}))
        for t in trajectory
    ]


def _format_memories(items) -> str:
    if not items:
        return ""
    lines = ["Lessons from prior attempts on similar bugs (apply if relevant):"]
    for i, it in enumerate(items, 1):
        title = it.title or "(untitled)"
        body = (it.content or "")[:400]
        lines.append(f"  {i}. {title}\n     {body}")
    return "\n".join(lines)


def _eval_one(record: dict, instance: dict, run_id: str, workdir: Path) -> bool:
    """Run swebench harness on a single record and return resolved bool."""
    preds = workdir / "predictions.jsonl"
    write_predictions([record], preds)
    report = run_swebench_eval(preds, [instance["instance_id"]], run_id, max_workers=1)
    resolved = report.get("resolved_ids") or []
    return instance["instance_id"] in resolved


async def _single_attempt(
    instance: dict,
    *,
    attempt_workdir: Path,
    extra_system: str | None,
    client: AsyncAnthropic,
    budget: Any,
    problem_override: str | None = None,
) -> dict:
    """Run agent once, return a per-attempt dict with patch, stats, trajectory."""
    try:
        repo_dir = prepare_workdir(instance, attempt_workdir)
    except Exception as e:
        return {
            "patch": "", "trajectory": [],
            "stats": {"steps": 0, "input_tokens": 0, "output_tokens": 0},
            "error": f"prepare_failed: {type(e).__name__}: {e}",
        }
    sandbox = RepoSandbox(repo_dir)
    problem = problem_override if problem_override is not None else instance["problem_statement"]
    try:
        final_text, traj, stats = await run_agent(
            problem, sandbox,
            extra_system=extra_system,
            max_steps=int(os.environ.get("PHASE3_MAX_STEPS", "20")),
            client=client,
            budget=budget,
        )
    except Exception as e:
        return {
            "patch": "", "trajectory": [],
            "stats": {"steps": 0, "input_tokens": 0, "output_tokens": 0},
            "error": f"agent_failed: {type(e).__name__}: {e}",
        }
    patch = extract_patch(repo_dir)
    return {
        "patch": patch, "trajectory": traj, "stats": stats,
        "final_text": final_text,
    }


def _make_record(instance: dict, patch: str) -> dict:
    return {
        "instance_id": instance["instance_id"],
        "model_patch": patch,
        "model_name_or_path": "haiku-4-5-agent",
    }


def _feedback_prompt(prev_patch: str, attempt_n: int) -> str:
    body = prev_patch[:1500] or "(empty patch — you did not edit any file)"
    return (
        f"\n\n## Prior attempt #{attempt_n} feedback\n"
        f"Your previous patch did not fix the bug — tests still fail. "
        f"Look more carefully at the issue and try a different approach. "
        f"The previous patch was:\n\n{body}"
    )


# --------- arms ---------

async def arm1_noretry(
    instance: dict, *, seed: int, budget: Any, client: AsyncAnthropic,
    workdir: Path, run_id: str, bank: ReasoningBank | None = None,
) -> dict:
    random.seed(seed)
    wd = workdir / "a1"
    r = await _single_attempt(
        instance, attempt_workdir=wd, extra_system=None,
        client=client, budget=budget,
    )
    rec = _make_record(instance, r["patch"])
    if r.get("error"):
        resolved = False
    else:
        resolved = _eval_one(rec, instance, f"{run_id}_a1", wd)
    return {
        "arm": "arm1_noretry",
        "instance_id": instance["instance_id"],
        "seed": seed,
        "attempts": 1,
        "resolved": resolved,
        "first_pass": resolved,
        "final_pass": resolved,
        "patches": [r["patch"][:PATCH_TRIM]],
        "per_attempt": [{
            "steps": r["stats"].get("steps", 0),
            "input_tokens": r["stats"].get("input_tokens", 0),
            "output_tokens": r["stats"].get("output_tokens", 0),
            "patch_len": len(r["patch"]),
            "resolved": resolved,
            "error": r.get("error"),
        }],
    }


async def arm2_naive(
    instance: dict, *, seed: int, budget: Any, client: AsyncAnthropic,
    workdir: Path, run_id: str, bank: ReasoningBank | None = None,
) -> dict:
    random.seed(seed)
    per_attempt = []
    patches = []
    first_pass = False
    final_pass = False
    prev_patch = ""
    attempts_made = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts_made = attempt
        wd = workdir / f"a{attempt}"
        problem_override = None
        if attempt > 1:
            problem_override = instance["problem_statement"] + _feedback_prompt(prev_patch, attempt - 1)
        r = await _single_attempt(
            instance, attempt_workdir=wd, extra_system=None,
            client=client, budget=budget,
            problem_override=problem_override,
        )
        prev_patch = r["patch"]
        rec = _make_record(instance, r["patch"])
        if r.get("error"):
            this_resolved = False
        else:
            this_resolved = _eval_one(rec, instance, f"{run_id}_a{attempt}", wd)
        patches.append(r["patch"][:PATCH_TRIM])
        per_attempt.append({
            "steps": r["stats"].get("steps", 0),
            "input_tokens": r["stats"].get("input_tokens", 0),
            "output_tokens": r["stats"].get("output_tokens", 0),
            "patch_len": len(r["patch"]),
            "resolved": this_resolved,
            "error": r.get("error"),
        })
        if attempt == 1:
            first_pass = this_resolved
        if this_resolved:
            final_pass = True
            break
    return {
        "arm": "arm2_naive",
        "instance_id": instance["instance_id"],
        "seed": seed,
        "attempts": attempts_made,
        "resolved": final_pass,
        "first_pass": first_pass,
        "final_pass": final_pass,
        "patches": patches,
        "per_attempt": per_attempt,
    }


async def _bank_arm(
    instance: dict, *, seed: int, budget: Any, client: AsyncAnthropic,
    workdir: Path, run_id: str, bank: ReasoningBank, arm_label: str,
) -> dict:
    random.seed(seed)
    pre_memories = await bank.retrieve(instance["problem_statement"], k=5)
    retrieved_pre = len(pre_memories)
    extra_system = _format_memories(pre_memories) if pre_memories else None

    per_attempt = []
    patches = []
    first_pass = False
    final_pass = False
    retrieved_retry = 0
    attempts_made = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts_made = attempt
        wd = workdir / f"a{attempt}"
        r = await _single_attempt(
            instance, attempt_workdir=wd, extra_system=extra_system,
            client=client, budget=budget,
        )
        rec = _make_record(instance, r["patch"])
        if r.get("error"):
            this_resolved = False
        else:
            this_resolved = _eval_one(rec, instance, f"{run_id}_a{attempt}", wd)
        patches.append(r["patch"][:PATCH_TRIM])
        per_attempt.append({
            "steps": r["stats"].get("steps", 0),
            "input_tokens": r["stats"].get("input_tokens", 0),
            "output_tokens": r["stats"].get("output_tokens", 0),
            "patch_len": len(r["patch"]),
            "resolved": this_resolved,
            "error": r.get("error"),
        })
        if attempt == 1:
            first_pass = this_resolved

        # Ingest this attempt's trajectory whether pass or fail
        try:
            await bank.ingest_trajectory(
                _traj_to_turns(r["trajectory"]),
                task=instance["problem_statement"],
                outcome="success" if this_resolved else "failure",
            )
        except Exception as e:
            print(f"  [bank-ingest-error] {type(e).__name__}: {e}", flush=True)

        if this_resolved:
            final_pass = True
            break

        # Refresh memories for the next attempt
        next_memories = await bank.retrieve(instance["problem_statement"], k=5)
        if attempt == 1:
            retrieved_retry = len(next_memories)
        extra_system = _format_memories(next_memories) if next_memories else None

    return {
        "arm": arm_label,
        "instance_id": instance["instance_id"],
        "seed": seed,
        "attempts": attempts_made,
        "resolved": final_pass,
        "first_pass": first_pass,
        "final_pass": final_pass,
        "patches": patches,
        "per_attempt": per_attempt,
        "retrieved_pre": retrieved_pre,
        "retrieved_retry": retrieved_retry,
    }


async def arm3a_bank_fresh(
    instance: dict, *, seed: int, budget: Any, client: AsyncAnthropic,
    workdir: Path, run_id: str, bank: ReasoningBank,
) -> dict:
    return await _bank_arm(
        instance, seed=seed, budget=budget, client=client,
        workdir=workdir, run_id=run_id, bank=bank,
        arm_label="arm3a_bank_fresh",
    )


async def arm3b_bank_persist(
    instance: dict, *, seed: int, budget: Any, client: AsyncAnthropic,
    workdir: Path, run_id: str, bank: ReasoningBank,
) -> dict:
    return await _bank_arm(
        instance, seed=seed, budget=budget, client=client,
        workdir=workdir, run_id=run_id, bank=bank,
        arm_label="arm3b_bank_persist",
    )


def bank_llm_factory(client: AsyncAnthropic, budget: Any = None):
    """LLM callable for ReasoningBank's internal summarisation/merging calls."""
    async def llm(prompt: str, *, system: str | None = None) -> str:
        # Conservatively reserve a small budget for these synthesis calls.
        if budget is not None:
            await budget.reserve(2000)
        msg = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1500,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        if budget is not None:
            budget.record(msg.usage.input_tokens)
        # Anthropic responses can contain multiple text blocks; concatenate.
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts) if parts else ""
    return llm
