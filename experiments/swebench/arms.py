"""Phase 3: three-arm experiment.

Arm 1 (no_retry): run once, record pass/fail.
Arm 2 (naive_retry): if fail, retry once with raw error fed back.
Arm 3a (bank_retry_fresh): bank per-instance; if fail, ingest then retry with memories.
Arm 3b (bank_retry_persist): bank persisted across instances; otherwise same as 3a.

Each arm gets 3 seeds.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.swebench.harness import (
    prepare_workdir, extract_patch, run_one_instance,
    write_predictions, run_swebench_eval, _ensure_repo_cache,
)
from experiments.swebench.agent import run_agent, RepoSandbox

from reasoning_bank import ReasoningBank, InMemoryStore
from reasoning_bank.types import Turn


def _bank_llm(client: AsyncAnthropic):
    async def llm(prompt: str, *, system: str | None = None) -> str:
        msg = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1500,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return msg.content[0].text
    return llm


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


async def _check_one_with_swebench(record: dict, instance: dict, run_id: str,
                                    workdir: Path) -> bool:
    """Run swebench eval on a single record. Returns True iff resolved."""
    preds = workdir / "predictions.jsonl"
    write_predictions([record], preds)
    report = run_swebench_eval(preds, [instance["instance_id"]], run_id, max_workers=1)
    resolved = report.get("resolved_ids") or []
    return instance["instance_id"] in resolved


async def run_no_retry(instance: dict, *, client: AsyncAnthropic, workdir: Path,
                       run_id: str) -> dict:
    rec = await run_one_instance(instance, workdir=workdir, client=client, max_steps=30)
    resolved = await _check_one_with_swebench(rec, instance, run_id, workdir)
    return {"arm": "no_retry", "first_pass": resolved, "final_pass": resolved,
            "instance_id": instance["instance_id"], "patch_len": len(rec.get("model_patch","")),
            "steps": rec["stats"]["steps"], "input_tokens": rec["stats"]["input_tokens"],
            "output_tokens": rec["stats"]["output_tokens"], "error": rec.get("error")}


async def run_naive_retry(instance: dict, *, client: AsyncAnthropic, workdir: Path,
                          run_id: str) -> dict:
    # 1st attempt
    wd1 = workdir / "attempt1"
    rec1 = await run_one_instance(instance, workdir=wd1, client=client, max_steps=30)
    first_pass = await _check_one_with_swebench(rec1, instance, run_id + "_a1", wd1)
    if first_pass:
        return {"arm": "naive_retry", "first_pass": True, "final_pass": True,
                "instance_id": instance["instance_id"],
                "patch_len_1": len(rec1.get("model_patch","")), "patch_len_2": 0,
                "steps_1": rec1["stats"]["steps"], "steps_2": 0,
                "input_tokens": rec1["stats"]["input_tokens"],
                "output_tokens": rec1["stats"]["output_tokens"]}
    # Retry with raw "your patch didn't pass tests" feedback
    feedback = (
        "Your previous patch did not fix the bug — tests still fail. "
        "Look more carefully at the issue and try a different approach. "
        "The previous patch was:\n\n" + (rec1.get("model_patch", "")[:1500] or "(empty)")
    )
    wd2 = workdir / "attempt2"
    # Fresh repo for retry
    repo_dir = prepare_workdir(instance, wd2)
    sandbox = RepoSandbox(repo_dir)
    try:
        final_text, traj, stats = await run_agent(
            instance["problem_statement"] + "\n\n## Prior attempt feedback\n" + feedback,
            sandbox, max_steps=30, client=client,
        )
        rec2 = {
            "instance_id": instance["instance_id"],
            "model_patch": extract_patch(repo_dir),
            "model_name_or_path": "haiku-4-5-agent",
            "stats": stats,
        }
    except Exception as e:
        rec2 = {"instance_id": instance["instance_id"], "model_patch": "",
                "model_name_or_path": "haiku-4-5-agent",
                "stats": {"steps": 0, "input_tokens": 0, "output_tokens": 0},
                "error": f"agent_failed: {e}"}
    final_pass = await _check_one_with_swebench(rec2, instance, run_id + "_a2", wd2)
    return {"arm": "naive_retry", "first_pass": False, "final_pass": final_pass,
            "instance_id": instance["instance_id"],
            "patch_len_1": len(rec1.get("model_patch","")),
            "patch_len_2": len(rec2.get("model_patch","")),
            "steps_1": rec1["stats"]["steps"], "steps_2": rec2["stats"]["steps"],
            "input_tokens": rec1["stats"]["input_tokens"] + rec2["stats"]["input_tokens"],
            "output_tokens": rec1["stats"]["output_tokens"] + rec2["stats"]["output_tokens"]}


async def run_bank_retry(instance: dict, *, client: AsyncAnthropic, workdir: Path,
                         run_id: str, bank: ReasoningBank, arm_label: str) -> dict:
    """Bank-retry arm. Uses provided bank (fresh per-instance for 3a, shared for 3b)."""
    # Retrieve any pre-existing memories first
    pre_memories = await bank.retrieve(instance["problem_statement"], k=5)
    extra_system_1 = _format_memories(pre_memories) if pre_memories else None

    wd1 = workdir / "attempt1"
    repo_dir = prepare_workdir(instance, wd1)
    sandbox = RepoSandbox(repo_dir)
    try:
        final_text, traj, stats = await run_agent(
            instance["problem_statement"], sandbox,
            extra_system=extra_system_1, max_steps=30, client=client,
        )
        rec1 = {
            "instance_id": instance["instance_id"],
            "model_patch": extract_patch(repo_dir),
            "model_name_or_path": "haiku-4-5-agent",
            "stats": stats,
            "_trajectory": traj,
        }
    except Exception as e:
        rec1 = {"instance_id": instance["instance_id"], "model_patch": "",
                "model_name_or_path": "haiku-4-5-agent",
                "stats": {"steps": 0, "input_tokens": 0, "output_tokens": 0},
                "_trajectory": [], "error": f"agent_failed: {e}"}

    first_pass = await _check_one_with_swebench(rec1, instance, run_id + "_a1", wd1)

    if first_pass:
        # Ingest the success too
        await bank.ingest_trajectory(_traj_to_turns(rec1["_trajectory"]),
                                      task=instance["problem_statement"], outcome="success")
        return {"arm": arm_label, "first_pass": True, "final_pass": True,
                "instance_id": instance["instance_id"],
                "patch_len_1": len(rec1.get("model_patch","")), "patch_len_2": 0,
                "steps_1": rec1["stats"]["steps"], "steps_2": 0,
                "input_tokens": rec1["stats"]["input_tokens"],
                "output_tokens": rec1["stats"]["output_tokens"],
                "retrieved_pre": len(pre_memories), "retrieved_retry": 0}

    # Ingest the failure
    await bank.ingest_trajectory(_traj_to_turns(rec1["_trajectory"]),
                                  task=instance["problem_statement"], outcome="failure")

    # Retrieve fresh (now includes the failure lesson)
    post_memories = await bank.retrieve(instance["problem_statement"], k=5)
    extra_system_2 = _format_memories(post_memories) if post_memories else None

    wd2 = workdir / "attempt2"
    repo_dir2 = prepare_workdir(instance, wd2)
    sandbox2 = RepoSandbox(repo_dir2)
    try:
        final_text2, traj2, stats2 = await run_agent(
            instance["problem_statement"], sandbox2,
            extra_system=extra_system_2, max_steps=30, client=client,
        )
        rec2 = {
            "instance_id": instance["instance_id"],
            "model_patch": extract_patch(repo_dir2),
            "model_name_or_path": "haiku-4-5-agent",
            "stats": stats2,
            "_trajectory": traj2,
        }
    except Exception as e:
        rec2 = {"instance_id": instance["instance_id"], "model_patch": "",
                "model_name_or_path": "haiku-4-5-agent",
                "stats": {"steps": 0, "input_tokens": 0, "output_tokens": 0},
                "_trajectory": [], "error": f"agent_failed: {e}"}

    final_pass = await _check_one_with_swebench(rec2, instance, run_id + "_a2", wd2)
    # Ingest retry result
    await bank.ingest_trajectory(
        _traj_to_turns(rec2["_trajectory"]),
        task=instance["problem_statement"],
        outcome="success" if final_pass else "failure",
    )
    return {"arm": arm_label, "first_pass": False, "final_pass": final_pass,
            "instance_id": instance["instance_id"],
            "patch_len_1": len(rec1.get("model_patch","")),
            "patch_len_2": len(rec2.get("model_patch","")),
            "steps_1": rec1["stats"]["steps"], "steps_2": rec2["stats"]["steps"],
            "input_tokens": rec1["stats"]["input_tokens"] + rec2["stats"]["input_tokens"],
            "output_tokens": rec1["stats"]["output_tokens"] + rec2["stats"]["output_tokens"],
            "retrieved_pre": len(pre_memories), "retrieved_retry": len(post_memories)}
