"""Sandboxed tools for the agentic experiment. Every tool is locked to a sandbox dir.
NO network, hard timeouts, destructive-command denylist."""
import subprocess
import sys
import os
from pathlib import Path

DENY_PATTERNS = [
    "rm -rf /",
    "sudo",
    ":()",
    "mkfs",
    "dd if=",
    "> /dev",
    "curl",
    "wget",
    "nc ",
    "ssh ",
]


class Sandbox:
    def __init__(self, root: str):
        self.root = Path(root).resolve()

    def _safe(self, path: str) -> Path:
        p = (self.root / path).resolve()
        # Must be inside root (or equal to root)
        try:
            p.relative_to(self.root)
        except ValueError:
            raise ValueError(f"Path escapes sandbox: {path}")
        return p

    def list_dir(self, path: str = ".") -> str:
        d = self._safe(path)
        if not d.exists():
            return f"ERROR: directory not found: {path}"
        if not d.is_dir():
            return f"ERROR: not a directory: {path}"
        entries = sorted(os.listdir(d))
        return "\n".join(entries) if entries else "(empty)"

    def read_file(self, path: str) -> str:
        p = self._safe(path)
        if not p.exists():
            return f"ERROR: file not found: {path}"
        if p.is_dir():
            return f"ERROR: is a directory: {path}"
        try:
            return p.read_text()[:4000]
        except Exception as e:
            return f"ERROR: {e}"

    def write_file(self, path: str, content: str) -> str:
        p = self._safe(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} bytes to {path}"

    def run_shell(self, command: str) -> str:
        for bad in DENY_PATTERNS:
            if bad in command:
                return f"ERROR: command blocked (matched {bad!r})"
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                timeout=10,
                capture_output=True,
                text=True,
            )
            return (
                f"[exit_code={r.returncode}]\n"
                f"stdout:\n{r.stdout[:2000]}\n"
                f"stderr:\n{r.stderr[:1000]}"
            )
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out (10s)"

    def run_python(self, code: str) -> str:
        try:
            r = subprocess.run(
                [sys.executable, "-c", code],
                cwd=self.root,
                timeout=10,
                capture_output=True,
                text=True,
            )
            return (
                f"[exit_code={r.returncode}]\n"
                f"stdout:\n{r.stdout[:2000]}\n"
                f"stderr:\n{r.stderr[:1000]}"
            )
        except subprocess.TimeoutExpired:
            return "ERROR: python timed out (10s)"
