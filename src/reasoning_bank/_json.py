"""Tolerant JSON parsing for LLM outputs.

LLMs occasionally wrap valid JSON in ```json ... ``` fences despite being
asked not to. ``parse_json_strict`` strips those fences before parsing and
raises a ``ValueError`` carrying the original text on failure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.IGNORECASE | re.MULTILINE)
_PROMPTS_DIR = Path(__file__).parent / "prompts"


def parse_json_strict(text: str) -> Any:
    """Parse JSON from an LLM response.

    Strips ``` ... ``` fences if present, then parses. Raises ``ValueError``
    with the original text if parsing fails.
    """
    if not text:
        raise ValueError("Empty LLM response — expected JSON")
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response was not valid JSON: {exc}\n--- raw response ---\n{text}"
        ) from exc


def load_prompt(name: str) -> str:
    """Load a prompt template by name from the packaged ``prompts/`` dir."""
    path = _PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")
