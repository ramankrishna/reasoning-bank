"""Raw-Anthropic agent for SWE-bench. No fleet, no langchain.

The agent gets a sandboxed view of a cloned repo and an issue description.
It must produce a code change. We capture the change via git diff afterward.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from anthropic import RateLimitError, APIStatusError

MODEL = "claude-haiku-4-5"


async def _call_with_retry(client: AsyncAnthropic, **kwargs):
    """Wrap messages.create with simple exponential backoff for 429s."""
    delays = [5, 15, 30, 60, 90]
    for attempt, delay in enumerate(delays + [None]):
        try:
            return await client.messages.create(**kwargs)
        except (RateLimitError, APIStatusError) as e:
            if delay is None:
                raise
            await asyncio.sleep(delay)
        except Exception:
            raise


TOOLS_SPEC = [
    {
        "name": "list_dir",
        "description": "List entries in a directory (relative to the repo root). "
                       "Returns a newline-separated listing.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative path; default '.'"}},
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the repo. Optionally provide line offset/limit "
                       "to read a slice (1-indexed offset, default whole file up to 400 lines).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "description": "1-indexed line to start from"},
                "limit": {"type": "integer", "description": "Max lines to return"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "grep",
        "description": "Search the repo for a regex pattern. Returns matching lines "
                       "with file:line. Use this before reading large files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Directory to search; default repo root"},
                "max_results": {"type": "integer", "description": "Default 50"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exactly one occurrence of old_text with new_text in a file. "
                       "Fails if old_text is not unique. Use this to make code changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "write_file",
        "description": "Overwrite a file with new content. Creates parent dirs if needed. "
                       "Prefer edit_file when modifying existing files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
]


SYSTEM_BASE = (
    "You are a software engineer fixing a bug or implementing a small feature "
    "request in a Python repository.\n\n"
    "Workflow — be DECISIVE, don't over-explore:\n"
    "1. Use grep to locate the relevant code (1-3 calls max).\n"
    "2. Use read_file to look at the specific function/class (1-3 reads max).\n"
    "3. Call edit_file to make the fix. DO THIS as soon as you have a hypothesis — "
    "you can iterate if the first edit is wrong.\n"
    "4. Once you've made the edit(s), send a final message starting with 'DONE'.\n\n"
    "Critical rules:\n"
    "- Make the MINIMAL change. Often it's 1-5 lines.\n"
    "- After ~10 tool calls of exploration, you MUST attempt an edit. "
    "Reading more code without editing is wasted effort.\n"
    "- Do NOT modify test files (under tests/, testing/, test_*.py).\n"
    "- If totally stuck, send 'DONE' and explain — don't loop forever.\n"
    "- The issue may be a bug OR a feature request (e.g., 'add a shortcut option'). "
    "Both are valid; implement what's asked."
)


class RepoSandbox:
    """Filesystem-scoped tools over a local repo clone."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def _safe(self, path: str) -> Path:
        p = (self.root / path).resolve()
        try:
            p.relative_to(self.root)
        except ValueError as e:
            raise ValueError(f"path escapes sandbox: {path}") from e
        return p

    def list_dir(self, path: str = ".") -> str:
        d = self._safe(path)
        if not d.exists():
            return f"ERROR: not found: {path}"
        if not d.is_dir():
            return f"ERROR: not a directory: {path}"
        entries = []
        for e in sorted(d.iterdir()):
            if e.name.startswith(".git"):
                continue
            marker = "/" if e.is_dir() else ""
            entries.append(f"{e.name}{marker}")
        return "\n".join(entries) if entries else "(empty)"

    def read_file(self, path: str, offset: int = 1, limit: int = 200) -> str:
        p = self._safe(path)
        if not p.exists():
            return f"ERROR: not found: {path}"
        if p.is_dir():
            return f"ERROR: is a directory: {path}"
        try:
            lines = p.read_text(errors="replace").splitlines()
        except Exception as e:
            return f"ERROR: {e}"
        start = max(0, offset - 1)
        end = min(len(lines), start + limit)
        chunk = lines[start:end]
        numbered = "\n".join(f"{i+start+1}: {ln}" for i, ln in enumerate(chunk))
        total = len(lines)
        suffix = f"\n... ({total - end} more lines)" if end < total else ""
        return numbered + suffix

    def grep(self, pattern: str, path: str = ".", max_results: int = 50) -> str:
        d = self._safe(path)
        try:
            r = subprocess.run(
                ["grep", "-rn", "--include=*.py", "-E", pattern, str(d)],
                capture_output=True, text=True, timeout=20,
            )
        except subprocess.TimeoutExpired:
            return "ERROR: grep timed out"
        lines = r.stdout.splitlines()
        truncated = ""
        if len(lines) > max_results:
            truncated = f"\n... ({len(lines)-max_results} more matches)"
            lines = lines[:max_results]
        rel = []
        for ln in lines:
            ln = ln.replace(str(self.root) + "/", "")
            rel.append(ln)
        return "\n".join(rel) + truncated if rel else "(no matches)"

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        p = self._safe(path)
        if not p.exists():
            return f"ERROR: not found: {path}"
        try:
            text = p.read_text()
        except Exception as e:
            return f"ERROR: {e}"
        count = text.count(old_text)
        if count == 0:
            return "ERROR: old_text not found"
        if count > 1:
            return f"ERROR: old_text appears {count} times — needs to be unique"
        new = text.replace(old_text, new_text, 1)
        p.write_text(new)
        return f"edited {path} (replaced 1 occurrence)"

    def write_file(self, path: str, content: str) -> str:
        p = self._safe(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} bytes to {path}"


async def run_agent(
    problem_statement: str,
    sandbox: RepoSandbox,
    extra_system: str | None = None,
    max_steps: int = 30,
    max_tokens_per_step: int = 4000,
    client: AsyncAnthropic | None = None,
    budget: "TokenBudget | None" = None,
) -> tuple[str, list[dict], dict]:
    """Run the agent until it says 'DONE' or hits max_steps.

    Returns (final_text, neutral_trajectory, stats).
    """
    client = client or AsyncAnthropic()
    system = SYSTEM_BASE + ("\n\n" + extra_system if extra_system else "")

    user_prompt = f"## Issue\n\n{problem_statement}\n\n## Your task\n\nFind and fix the bug."
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    trajectory: list[dict] = [{"role": "user", "content": user_prompt}]

    tool_impls = {
        "list_dir": sandbox.list_dir,
        "read_file": sandbox.read_file,
        "grep": sandbox.grep,
        "edit_file": sandbox.edit_file,
        "write_file": sandbox.write_file,
    }

    final_text = ""
    stats = {"steps": 0, "input_tokens": 0, "output_tokens": 0, "stopped_by": "max_steps"}
    # Per-step floor delay to respect 50k tok/min rate limit. Set RB_STEP_DELAY=0 to disable.
    step_delay = float(os.environ.get("RB_STEP_DELAY", "0"))

    for step in range(max_steps):
        stats["steps"] = step + 1
        if step > 0 and step_delay > 0:
            await asyncio.sleep(step_delay)
        if budget is not None:
            est = budget.last_recorded if budget.last_recorded else 3000
            await budget.reserve(est)
        resp = await _call_with_retry(
            client,
            model=MODEL,
            max_tokens=max_tokens_per_step,
            temperature=0.0,
            system=system,
            tools=TOOLS_SPEC,
            messages=messages,
        )
        if budget is not None:
            budget.record(resp.usage.input_tokens)
        stats["input_tokens"] += resp.usage.input_tokens
        stats["output_tokens"] += resp.usage.output_tokens

        text_parts = [b.text for b in resp.content if b.type == "text"]
        if text_parts:
            text = "\n".join(text_parts)
            trajectory.append({"role": "assistant", "content": text})
            final_text = text

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            stats["stopped_by"] = "no_tool_calls"
            break

        # Did the model emit DONE in text? If so, stop.
        if final_text.strip().startswith("DONE") or "\nDONE" in final_text:
            stats["stopped_by"] = "done_marker"
            break

        messages.append({"role": "assistant", "content": resp.content})

        results = []
        for tu in tool_uses:
            try:
                fn = tool_impls[tu.name]
                out = fn(**(tu.input or {}))
            except TypeError as e:
                out = f"ERROR: bad arguments: {e}"
            except Exception as e:
                out = f"ERROR: {e}"
            if not isinstance(out, str):
                out = str(out)
            # cap each tool result aggressively (rate-limit pressure)
            if len(out) > 3500:
                out = out[:3500] + "\n... (truncated)"
            trajectory.append({
                "role": "tool",
                "content": f"{tu.name}({tu.input}) -> {out[:1500]}",
                "meta": {"tool": tu.name},
            })
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out})
        messages.append({"role": "user", "content": results})

    return final_text, trajectory, stats
