"""Neutral, framework-agnostic trajectory representation.

A ``Trajectory`` is just ``list[Turn]``. ``Turn`` is a tiny dataclass with a
``role`` (``system|user|assistant|tool``), free-form ``content`` text, and a
``meta`` bag for anything else (tool_calls, tool_results, timing, ids…).

The bank never imports from a specific agent framework. Callers convert
their own message objects into Turns via ``_coerce_trajectory()``, which
accepts three input shapes:

1. ``list[Turn]`` — passed through.
2. ``list[dict]`` with at least a ``role`` and ``content`` key (extra keys
   land in ``meta``).
3. ``list`` of duck-typed objects exposing ``.role`` and ``.content`` attrs
   (anything else accessible via ``vars()`` or known attrs is preserved in
   ``meta``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
_ROLES: frozenset[str] = frozenset(("system", "user", "assistant", "tool"))


@dataclass
class Turn:
    role: Role
    content: str
    meta: dict[str, Any] = field(default_factory=dict)


Trajectory = list[Turn]


def _coerce_role(value: Any) -> Role:
    s = str(value).strip().lower()
    if s not in _ROLES:
        raise ValueError(
            f"turn role must be one of {sorted(_ROLES)}, got {value!r}"
        )
    return s  # type: ignore[return-value]


def _coerce_one(msg: Any) -> Turn:
    if isinstance(msg, Turn):
        return msg
    if isinstance(msg, dict):
        if "role" not in msg:
            raise ValueError(f"turn dict missing 'role' key: {msg!r}")
        role = _coerce_role(msg["role"])
        content = msg.get("content") or ""
        meta = {k: v for k, v in msg.items() if k not in ("role", "content")}
        return Turn(role=role, content=str(content), meta=meta)
    role_attr = getattr(msg, "role", None)
    if role_attr is None:
        raise ValueError(
            f"cannot coerce {type(msg).__name__} to Turn — missing .role"
        )
    role = _coerce_role(role_attr)
    content_attr = getattr(msg, "content", "") or ""
    meta: dict[str, Any] = {}
    for name in ("tool_calls", "tool_results", "name", "id"):
        val = getattr(msg, name, None)
        if val is not None:
            meta[name] = val
    return Turn(role=role, content=str(content_attr), meta=meta)


def _coerce_trajectory(trajectory: Any) -> Trajectory:
    """Coerce a list of messages into ``list[Turn]``.

    Accepts ``list[Turn]``, ``list[dict]`` (with role/content keys), or a list
    of duck-typed objects with ``.role`` / ``.content`` attributes.
    """
    if trajectory is None:
        return []
    try:
        iterator = iter(trajectory)
    except TypeError as exc:
        raise TypeError(
            f"trajectory must be iterable, got {type(trajectory).__name__}"
        ) from exc
    return [_coerce_one(msg) for msg in iterator]


def render_trajectory(trajectory: Any) -> str:
    """Render a trajectory (any accepted form) as readable text for prompts."""
    turns = _coerce_trajectory(trajectory)
    lines: list[str] = []
    for turn in turns:
        if turn.content:
            lines.append(f"[{turn.role}] {turn.content}")
        for tc in turn.meta.get("tool_calls") or []:
            name = (
                tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            )
            args = (
                tc.get("arguments")
                if isinstance(tc, dict)
                else getattr(tc, "arguments", None)
            )
            lines.append(f"[{turn.role}] tool_call: {name}({args})")
        for tr in turn.meta.get("tool_results") or []:
            content_val = (
                tr.get("content")
                if isinstance(tr, dict)
                else getattr(tr, "content", None)
            )
            is_error = (
                tr.get("is_error", False)
                if isinstance(tr, dict)
                else getattr(tr, "is_error", False)
            )
            prefix = "tool_error" if is_error else "tool_result"
            lines.append(f"[{turn.role}] {prefix}: {content_val}")
    return "\n".join(lines)
