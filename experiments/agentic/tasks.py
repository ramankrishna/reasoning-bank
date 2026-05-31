"""Task suite for the agentic experiment.

Twelve tasks across five gotcha families (G1..G5). Each task has:
  - id
  - prompt  (the user-facing task description)
  - gotchas (set of gotcha codes it primarily tests)
  - setup(sandbox_root: Path) -> None       — establish starting filesystem state
  - verify(final_answer: str,                — judge whether task was solved
           sandbox_root: Path,
           trajectory: list[dict]) -> bool
  - The gotcha-error detector is shared across tasks (detect_gotcha_errors).

Gotcha families:
  G1 — discovery-before-action:   filenames not given, must list_dir first
  G2 — state-doesnt-persist:      each run_python is a fresh process
  G3 — stdout-vs-return:          run_python captures stdout, not values
  G4 — exit-code-matters:         non-zero exit => failure even if stdout looks ok
  G5 — append-needs-existence:    file may be missing; check before assuming
"""
from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------


@dataclass
class Task:
    id: str
    prompt: str
    gotchas: set[str]
    setup: Callable[[Path], None]
    verify: Callable[[str, Path, list[dict]], bool]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(p: Path, content: str, executable: bool = False) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    if executable:
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _ans_norm(s: str) -> str:
    return s.lower().strip()


def _ans_has(s: str, *needles: str) -> bool:
    n = _ans_norm(s)
    return any(needle.lower() in n for needle in needles)


# ---------------------------------------------------------------------------
# G1 — discovery-before-action
# ---------------------------------------------------------------------------


def _setup_secret(root: Path) -> None:
    _write(root / "config.dat", "TOKEN=alpha-9er\nMODE=prod\n")
    _write(root / "readme.txt", "Welcome. No secrets here.\n")


def _verify_secret(answer: str, root: Path, traj: list[dict]) -> bool:
    return "alpha-9er" in answer


def _setup_birthday(root: Path) -> None:
    _write(root / "profile.json", '{"name": "Ada", "birthday": "1985-03-22"}\n')
    _write(root / "notes.md", "# Notes\nNothing important.\n")


def _verify_birthday(answer: str, root: Path, traj: list[dict]) -> bool:
    return "1985-03-22" in answer


def _setup_count(root: Path) -> None:
    _write(root / "events.log", "\n".join([f"event {i}" for i in range(1, 8)]) + "\n")


def _verify_count(answer: str, root: Path, traj: list[dict]) -> bool:
    return bool(re.search(r"\b7\b", answer))


# ---------------------------------------------------------------------------
# G2 — state-doesnt-persist
# ---------------------------------------------------------------------------


def _setup_noop(root: Path) -> None:
    pass


def _verify_fib(answer: str, root: Path, traj: list[dict]) -> bool:
    return "6765" in answer


def _verify_squares_sum(answer: str, root: Path, traj: list[dict]) -> bool:
    return "385" in answer


# ---------------------------------------------------------------------------
# G3 — stdout-vs-return
# ---------------------------------------------------------------------------


def _verify_sha(answer: str, root: Path, traj: list[dict]) -> bool:
    # sha256(b'reasoning').hexdigest()
    expected = "ae16d2bbf81717eba2c1fbab5b4bf32c2c2cb2eef33c7c20b7c9aac9b8f63b22"
    # Recompute to avoid copy errors:
    import hashlib
    expected = hashlib.sha256(b"reasoning").hexdigest()
    return expected in answer.lower()


def _verify_sum_range(answer: str, root: Path, traj: list[dict]) -> bool:
    return "5050" in answer


# ---------------------------------------------------------------------------
# G4 — exit-code-matters
# ---------------------------------------------------------------------------


def _setup_build_fail(root: Path) -> None:
    _write(
        root / "build.sh",
        "#!/usr/bin/env bash\necho 'BUILD OK'\necho 'Artifacts produced.'\nexit 1\n",
        executable=True,
    )


def _verify_build_status(answer: str, root: Path, traj: list[dict]) -> bool:
    # Correct answer: build did NOT succeed (because exit 1).
    a = _ans_norm(answer)
    # Failure synonyms — present?
    fail_words = ["fail", "failed", "did not succeed", "didn't succeed",
                  "no", "false", "unsuccess", "exit code 1", "exit 1",
                  "non-zero", "nonzero", "broken"]
    if any(w in a for w in fail_words):
        return True
    return False


def _setup_tests_fail(root: Path) -> None:
    _write(
        root / "run_tests.sh",
        "#!/usr/bin/env bash\necho 'Tests passed!'\nexit 2\n",
        executable=True,
    )


def _verify_tests_status(answer: str, root: Path, traj: list[dict]) -> bool:
    a = _ans_norm(answer)
    fail_words = ["did not pass", "didn't pass", "fail", "no", "not all",
                  "exit code 2", "exit 2", "non-zero", "nonzero", "false"]
    # Make sure they didn't ALSO say passed unqualifiedly
    if any(w in a for w in fail_words):
        return True
    return False


def _setup_grep_miss(root: Path) -> None:
    _write(root / "data.txt", "alpha\nbeta\ngamma\n")


def _verify_grep_miss(answer: str, root: Path, traj: list[dict]) -> bool:
    a = _ans_norm(answer)
    # Correct: "no", file does not contain foo.
    neg = ["no", "does not contain", "doesn't contain", "not contain",
           "not found", "no occurrences", "absent", "no match"]
    pos = ["yes,", "yes.", "contains 'foo'", "contains foo", "found foo"]
    if any(p in a for p in pos):
        return False
    return any(n in a for n in neg)


# ---------------------------------------------------------------------------
# G5 — append-needs-existence
# ---------------------------------------------------------------------------


def _setup_no_log(root: Path) -> None:
    pass  # log.txt absent on purpose


def _verify_append_create(answer: str, root: Path, traj: list[dict]) -> bool:
    p = root / "log.txt"
    if not p.exists():
        return False
    return "first entry" in p.read_text()


def _setup_log_exists(root: Path) -> None:
    _write(root / "log.txt", "line one\nline two\n")


def _verify_append_existing(answer: str, root: Path, traj: list[dict]) -> bool:
    p = root / "log.txt"
    if not p.exists():
        return False
    content = p.read_text()
    return "line one" in content and "line two" in content and "new entry" in content


# ---------------------------------------------------------------------------
# Task list
# ---------------------------------------------------------------------------


TASKS: list[Task] = [
    Task(
        id="T01_secret_token",
        prompt="There is a secret token stored somewhere in your current "
               "working directory. Find it and report it exactly.",
        gotchas={"G1"},
        setup=_setup_secret,
        verify=_verify_secret,
    ),
    Task(
        id="T02_birthday",
        prompt="What is the user's birthday? The answer is somewhere in the "
               "current working directory. Report it in YYYY-MM-DD form.",
        gotchas={"G1"},
        setup=_setup_birthday,
        verify=_verify_birthday,
    ),
    Task(
        id="T03_count_lines",
        prompt="How many lines are in the log file in the current directory? "
               "Report just the number.",
        gotchas={"G1"},
        setup=_setup_count,
        verify=_verify_count,
    ),
    Task(
        id="T04_fib_split",
        prompt="Using the run_python tool: in one call, define an iterative "
               "function fib(n) that returns the nth Fibonacci number "
               "(fib(0)=0, fib(1)=1). In a SEPARATE second run_python call, "
               "call fib(20) and report the result.",
        gotchas={"G2"},
        setup=_setup_noop,
        verify=_verify_fib,
    ),
    Task(
        id="T05_squares_sum",
        prompt="Using run_python: build the list data = [i*i for i in range(1,11)] "
               "in one call, then in a SEPARATE second run_python call print sum(data). "
               "Report the value.",
        gotchas={"G2"},
        setup=_setup_noop,
        verify=_verify_squares_sum,
    ),
    Task(
        id="T06_sha_digest",
        prompt="Use run_python to compute the SHA-256 hex digest of the byte "
               "string b'reasoning' (hashlib.sha256(b'reasoning').hexdigest()). "
               "Report only the hex digest.",
        gotchas={"G3"},
        setup=_setup_noop,
        verify=_verify_sha,
    ),
    Task(
        id="T07_sum_range",
        prompt="Use run_python to evaluate sum(range(1, 101)). Report the value.",
        gotchas={"G3"},
        setup=_setup_noop,
        verify=_verify_sum_range,
    ),
    Task(
        id="T08_build_status",
        prompt="Run ./build.sh from your shell and report whether the build "
               "succeeded.",
        gotchas={"G4"},
        setup=_setup_build_fail,
        verify=_verify_build_status,
    ),
    Task(
        id="T09_test_status",
        prompt="Run ./run_tests.sh and report whether all tests passed.",
        gotchas={"G4"},
        setup=_setup_tests_fail,
        verify=_verify_tests_status,
    ),
    Task(
        id="T10_grep_miss",
        prompt="Use the shell tool to grep for the literal string 'foo' in "
               "data.txt. Does the file contain 'foo'? Answer yes or no.",
        gotchas={"G4"},
        setup=_setup_grep_miss,
        verify=_verify_grep_miss,
    ),
    Task(
        id="T11_append_create",
        prompt="Append the line 'first entry' to log.txt. After you are done, "
               "report 'done'.",
        gotchas={"G5"},
        setup=_setup_no_log,
        verify=_verify_append_create,
    ),
    Task(
        id="T12_append_existing",
        prompt="Append the line 'new entry' to log.txt, preserving any existing "
               "lines. After you are done, report 'done'.",
        gotchas={"G5"},
        setup=_setup_log_exists,
        verify=_verify_append_existing,
    ),
]


# ---------------------------------------------------------------------------
# Gotcha-error detection (trajectory-based)
# ---------------------------------------------------------------------------


def detect_gotcha_errors(trajectory: list[dict], final_answer: str = "") -> set[str]:
    """Inspect a trajectory and return the gotcha codes that were *tripped*.

    A gotcha is "tripped" if the trajectory shows the canonical mistake
    signature, regardless of whether the agent later recovered. Recovery is
    still a learning signal — the bank should distill the lesson either way.

    Signatures:
      G1: any read_file call returned 'file not found' (agent guessed wrong)
      G2: any run_python output contains 'NameError' (split state across calls)
      G3: any run_python call whose code lacks 'print(' and whose output had
          empty stdout — agent ran a bare expression and got nothing back
      G4: any run_shell with non-zero exit_code AND the final answer affirms
          success (so the agent overlooked the failure)
      G5: agent issued a read_file or shell-cat against log.txt that returned
          'file not found', and then proceeded to act as if it existed
          (e.g. write_file or shell append) without first creating it
    """
    final = _ans_norm(final_answer)
    hits: set[str] = set()

    py_results: list[tuple[str, str]] = []   # (code, output)
    shell_results: list[tuple[str, str]] = []  # (command, output)
    read_results: list[tuple[str, str]] = []   # (path, output)

    for turn in trajectory:
        if turn.get("role") != "tool":
            continue
        tool = (turn.get("meta") or {}).get("tool")
        content = turn.get("content", "")
        # The recorded content is f"{name}({input}) -> {output}".
        # We split on the first " -> " to extract the output side.
        if " -> " in content:
            call_str, output = content.split(" -> ", 1)
        else:
            call_str, output = content, ""

        if tool == "read_file":
            # input dict appears between ({ and }) in call_str
            m = re.search(r"'path':\s*'([^']*)'", call_str)
            path = m.group(1) if m else ""
            read_results.append((path, output))
        elif tool == "run_python":
            m = re.search(r"'code':\s*'(.*)'\}\)$", call_str)
            code = m.group(1) if m else call_str
            py_results.append((code, output))
        elif tool == "run_shell":
            m = re.search(r"'command':\s*'(.*)'\}\)$", call_str)
            cmd = m.group(1) if m else call_str
            shell_results.append((cmd, output))

    # G1 — any read_file returned "file not found"
    for _path, out in read_results:
        if "file not found" in out.lower():
            hits.add("G1")
            break

    # G2 — any run_python output contained NameError
    for _code, out in py_results:
        if "NameError" in out:
            hits.add("G2")
            break

    # G3 — bare expression with empty stdout
    for code, out in py_results:
        if "print(" in code:
            continue
        # Extract stdout block
        m = re.search(r"stdout:\n(.*?)\nstderr:", out, flags=re.DOTALL)
        stdout = m.group(1).strip() if m else ""
        if stdout == "":
            # Only count if the run executed (some signal present)
            if "exit_code=0" in out or "stderr:" in out:
                hits.add("G3")
                break

    # G4 — non-zero exit + final answer affirms success
    success_words = [
        "build succeeded", "succeeded", "all tests passed",
        "tests passed", "build ok", "yes,", "yes.",
        "passed", "success", "successfully", "contains foo",
        "contains 'foo'",
    ]
    failed_words = [
        "did not succeed", "didn't succeed", "fail", "failed", "no,",
        "no.", "not pass", "exit code 1", "exit code 2", "non-zero",
        "nonzero", "does not contain", "doesn't contain",
    ]
    affirms_success = (
        any(w in final for w in success_words)
        and not any(w in final for w in failed_words)
    )
    for _cmd, out in shell_results:
        m = re.search(r"exit_code=(-?\d+)", out)
        if m and m.group(1) != "0":
            if affirms_success:
                hits.add("G4")
            break

    # G5 — saw file-missing for log.txt and proceeded anyway
    saw_missing_log = any(
        "log.txt" in p and "file not found" in out.lower()
        for p, out in read_results
    )
    if saw_missing_log:
        # Did the agent then 'act as if it existed' = write_file/append?
        # In our framing, simply trying to read first and failing IS the
        # gotcha signature for append-needs-existence — they assumed it was
        # there. We surface it regardless of recovery.
        hits.add("G5")

    return hits


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def by_id(task_id: str) -> Task:
    for t in TASKS:
        if t.id == task_id:
            return t
    raise KeyError(task_id)
