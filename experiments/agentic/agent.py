"""Raw-Anthropic ReAct agent with sandboxed tools. No fleet."""
from __future__ import annotations

import os
from typing import Any

from anthropic import AsyncAnthropic

from experiments.agentic.tools import Sandbox

MODEL = "claude-haiku-4-5-20251001"


TOOLS_SPEC = [
    {
        "name": "list_dir",
        "description": "List files and subdirectories under the given path "
                       "(relative to the working directory).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file (relative path). Returns "
                       "'ERROR: file not found: ...' if missing.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file (overwrites any existing file). "
                       "Parent directories are created if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_shell",
        "description": "Run a shell command (10s timeout). Returns "
                       "'[exit_code=N]\\nstdout:\\n...\\nstderr:\\n...'. "
                       "Use the exit_code to know if the command succeeded.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "run_python",
        "description": "Run python code in a fresh subprocess. NO STATE "
                       "PERSISTS between calls. Use print() to see values — "
                       "bare expressions return nothing to stdout.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
]


SYSTEM_BASE = (
    "You are an agent with filesystem, shell, and python tools. Complete the "
    "task efficiently. When you are done, send a final message that starts "
    "with the literal token 'ANSWER:' followed by your answer."
)


def _make_client() -> AsyncAnthropic:
    return AsyncAnthropic()


async def run_agent(
    task_prompt: str,
    sandbox: Sandbox,
    system_block: str | None = None,
    max_steps: int = 10,
    client: AsyncAnthropic | None = None,
) -> tuple[str, list[dict]]:
    """Run a ReAct loop. Returns (final_answer, neutral_trajectory).

    ``system_block`` is concatenated AFTER the base system instructions —
    pass retrieved memories here for the bank-on arm.
    """
    client = client or _make_client()
    if system_block:
        system = SYSTEM_BASE + "\n\n" + system_block
    else:
        system = SYSTEM_BASE

    messages: list[dict[str, Any]] = [{"role": "user", "content": task_prompt}]
    trajectory: list[dict] = [{"role": "user", "content": task_prompt}]

    tool_impls = {
        "list_dir": sandbox.list_dir,
        "read_file": sandbox.read_file,
        "write_file": sandbox.write_file,
        "run_shell": sandbox.run_shell,
        "run_python": sandbox.run_python,
    }

    final_text = ""
    for _step in range(max_steps):
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=1500,
            temperature=0.0,
            system=system,
            tools=TOOLS_SPEC,
            messages=messages,
        )

        text_parts = [b.text for b in resp.content if b.type == "text"]
        if text_parts:
            text = "\n".join(text_parts)
            trajectory.append({"role": "assistant", "content": text})
            final_text = text  # last assistant text wins

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            return final_text, trajectory

        # Echo the assistant message into the chat history so tool_use ids resolve.
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
            trajectory.append({
                "role": "tool",
                "content": f"{tu.name}({tu.input}) -> {out}",
                "meta": {"tool": tu.name},
            })
            results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": out,
            })
        messages.append({"role": "user", "content": results})

    return final_text or "ANSWER: (max steps reached)", trajectory
