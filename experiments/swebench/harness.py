"""Per-instance harness: clone repo at base_commit, run agent, extract patch."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

from experiments.swebench.agent import RepoSandbox, run_agent

REPO_CACHE = Path(os.environ.get("REPO_CACHE", "/tmp/swebench_repo_cache"))
REPO_CACHE.mkdir(exist_ok=True, parents=True)


def _git(cwd: Path, *args: str, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check, capture_output=True,
        text=True, timeout=timeout,
    )


def _ensure_repo_cache(repo: str) -> Path:
    """Maintain a bare-ish clone we can copy from. Avoids re-cloning each time."""
    safe = repo.replace("/", "__")
    cache = REPO_CACHE / safe
    if cache.exists() and (cache / ".git").exists():
        return cache
    print(f"  cloning {repo} into cache...")
    cache.parent.mkdir(exist_ok=True, parents=True)
    _git(REPO_CACHE, "clone", f"https://github.com/{repo}.git", str(cache), timeout=600)
    return cache


def prepare_workdir(instance: dict, workdir: Path) -> Path:
    """Clone the repo at base_commit. NOTE: we do NOT apply test_patch — that
    would leak the answer to the agent. swebench applies test_patch during eval."""
    workdir.mkdir(exist_ok=True, parents=True)
    repo_dir = workdir / "repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    cache = _ensure_repo_cache(instance["repo"])
    subprocess.run(["cp", "-R", str(cache), str(repo_dir)], check=True)
    _git(repo_dir, "fetch", "--all", "--quiet", check=False, timeout=300)
    _git(repo_dir, "checkout", "-f", instance["base_commit"], timeout=120)
    _git(repo_dir, "clean", "-fdx", timeout=60)
    return repo_dir


def extract_patch(repo_dir: Path) -> str:
    """git diff of agent's changes vs base_commit."""
    _git(repo_dir, "add", "-A", timeout=60)
    r = _git(repo_dir, "diff", "--cached", timeout=60, check=False)
    return r.stdout


async def run_one_instance(
    instance: dict,
    *,
    workdir: Path,
    extra_system: str | None = None,
    max_steps: int = 30,
    client: AsyncAnthropic | None = None,
    budget: Any = None,
) -> dict:
    """Run the agent on a single instance. Returns prediction record + stats."""
    t0 = time.time()
    instance_id = instance["instance_id"]
    try:
        repo_dir = prepare_workdir(instance, workdir)
    except Exception as e:
        return {
            "instance_id": instance_id,
            "model_patch": "",
            "model_name_or_path": "haiku-4-5-agent",
            "error": f"prepare_failed: {e}",
            "stats": {"steps": 0, "input_tokens": 0, "output_tokens": 0, "wall_s": time.time() - t0},
        }

    sandbox = RepoSandbox(repo_dir)
    try:
        final_text, trajectory, agent_stats = await run_agent(
            instance["problem_statement"], sandbox,
            extra_system=extra_system,
            max_steps=max_steps,
            client=client,
            budget=budget,
        )
    except Exception as e:
        return {
            "instance_id": instance_id,
            "model_patch": "",
            "model_name_or_path": "haiku-4-5-agent",
            "error": f"agent_failed: {type(e).__name__}: {e}",
            "stats": {"steps": 0, "input_tokens": 0, "output_tokens": 0, "wall_s": time.time() - t0},
        }

    patch = extract_patch(repo_dir)
    agent_stats["wall_s"] = time.time() - t0
    return {
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": "haiku-4-5-agent",
        "final_text": final_text[:1000],
        "trajectory": trajectory,
        "stats": agent_stats,
    }


def write_predictions(records: list[dict], path: Path) -> None:
    """Write predictions in the format swebench expects."""
    with path.open("w") as f:
        for rec in records:
            slim = {
                "instance_id": rec["instance_id"],
                "model_patch": rec.get("model_patch", ""),
                "model_name_or_path": rec.get("model_name_or_path", "haiku-4-5-agent"),
            }
            f.write(json.dumps(slim) + "\n")


def run_swebench_eval(predictions_path: Path, instance_ids: list[str], run_id: str,
                      max_workers: int = 2) -> dict:
    """Invoke swebench harness; return the parsed report json."""
    env = os.environ.copy()
    env.setdefault("DOCKER_HOST", "unix:///Users/ram/.colima/default/docker.sock")
    cmd = [
        ".venv/bin/python", "-m", "swebench.harness.run_evaluation",
        "-p", str(predictions_path),
        "-id", run_id,
        "-i", *instance_ids,
        "--max_workers", str(max_workers),
        "--cache_level", "instance",
    ]
    cwd = Path("/Users/ram/reasoning-bank")
    print(f"  running swebench eval for {len(instance_ids)} instances...")
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=7200)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
    # Report is written in cwd of process
    report_name = f"haiku-4-5-agent.{run_id}.json"
    report = cwd / report_name
    if report.exists():
        return json.loads(report.read_text())
    return {"error": "no report written", "stdout_tail": r.stdout[-500:], "stderr_tail": r.stderr[-500:]}
