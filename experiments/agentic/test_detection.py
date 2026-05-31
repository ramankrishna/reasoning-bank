"""Unit tests for detect_gotcha_errors and Task.verify (with correct trajectories).

Run with:  uv run python -m experiments.agentic.test_detection
"""
from __future__ import annotations

import tempfile
import shutil
from pathlib import Path

from experiments.agentic.tasks import (
    TASKS,
    by_id,
    detect_gotcha_errors,
)


# ---------------------------------------------------------------------------
# Synthetic trajectory helpers
# ---------------------------------------------------------------------------


def tt_user(s: str) -> dict:
    return {"role": "user", "content": s}


def tt_asst(s: str) -> dict:
    return {"role": "assistant", "content": s}


def tt_tool(name: str, inp: dict, out: str) -> dict:
    return {
        "role": "tool",
        "content": f"{name}({inp}) -> {out}",
        "meta": {"tool": name},
    }


# ---------------------------------------------------------------------------
# Synthetic trajectories
# ---------------------------------------------------------------------------


def case_g1_hit() -> tuple[list[dict], str, set[str]]:
    traj = [
        tt_user("Find the secret."),
        tt_tool("read_file", {"path": "secrets.txt"},
                "ERROR: file not found: secrets.txt"),
        tt_tool("list_dir", {}, "config.dat\nreadme.txt"),
        tt_tool("read_file", {"path": "config.dat"}, "TOKEN=alpha-9er\n"),
        tt_asst("ANSWER: alpha-9er"),
    ]
    return traj, "ANSWER: alpha-9er", {"G1"}


def case_g1_clean() -> tuple[list[dict], str, set[str]]:
    traj = [
        tt_user("Find the secret."),
        tt_tool("list_dir", {}, "config.dat\nreadme.txt"),
        tt_tool("read_file", {"path": "config.dat"}, "TOKEN=alpha-9er\n"),
        tt_asst("ANSWER: alpha-9er"),
    ]
    return traj, "ANSWER: alpha-9er", set()


def case_g2_hit() -> tuple[list[dict], str, set[str]]:
    traj = [
        tt_tool("run_python", {"code": "def fib(n): ...\n"},
                "[exit_code=0]\nstdout:\n\nstderr:\n"),
        tt_tool("run_python", {"code": "print(fib(20))"},
                "[exit_code=1]\nstdout:\n\nstderr:\nNameError: name 'fib' is not defined\n"),
        tt_asst("ANSWER: failed"),
    ]
    return traj, "ANSWER: failed", {"G2", "G3"}  # bare def call empty-stdout would NOT hit G3 because code does have print? But first call code is the def; let's not assert G3 here.


def case_g2_clean() -> tuple[list[dict], str, set[str]]:
    traj = [
        tt_tool("run_python", {"code": "def fib(n): ...\nprint(fib(20))"},
                "[exit_code=0]\nstdout:\n6765\n\nstderr:\n"),
        tt_asst("ANSWER: 6765"),
    ]
    return traj, "ANSWER: 6765", set()


def case_g3_hit() -> tuple[list[dict], str, set[str]]:
    traj = [
        tt_tool("run_python", {"code": "sum(range(1,101))"},
                "[exit_code=0]\nstdout:\n\nstderr:\n"),
        tt_asst("ANSWER: unknown"),
    ]
    return traj, "ANSWER: unknown", {"G3"}


def case_g3_clean() -> tuple[list[dict], str, set[str]]:
    traj = [
        tt_tool("run_python", {"code": "print(sum(range(1,101)))"},
                "[exit_code=0]\nstdout:\n5050\n\nstderr:\n"),
        tt_asst("ANSWER: 5050"),
    ]
    return traj, "ANSWER: 5050", set()


def case_g4_hit() -> tuple[list[dict], str, set[str]]:
    traj = [
        tt_tool("run_shell", {"command": "./build.sh"},
                "[exit_code=1]\nstdout:\nBUILD OK\n\nstderr:\n"),
        tt_asst("Build succeeded."),
    ]
    return traj, "The build succeeded.", {"G4"}


def case_g4_clean() -> tuple[list[dict], str, set[str]]:
    traj = [
        tt_tool("run_shell", {"command": "./build.sh"},
                "[exit_code=1]\nstdout:\nBUILD OK\n\nstderr:\n"),
        tt_asst("The build did not succeed (exit code 1)."),
    ]
    return traj, "The build did not succeed (exit code 1).", set()


def case_g5_hit() -> tuple[list[dict], str, set[str]]:
    traj = [
        tt_tool("read_file", {"path": "log.txt"},
                "ERROR: file not found: log.txt"),
        tt_tool("write_file", {"path": "log.txt", "content": "first entry\n"},
                "wrote 12 bytes to log.txt"),
        tt_asst("done"),
    ]
    return traj, "done", {"G1", "G5"}  # also G1 by signature (read failed)


def case_g5_clean() -> tuple[list[dict], str, set[str]]:
    traj = [
        tt_tool("list_dir", {}, "(empty)"),
        tt_tool("write_file", {"path": "log.txt", "content": "first entry\n"},
                "wrote 12 bytes to log.txt"),
        tt_asst("done"),
    ]
    return traj, "done", set()


SYNTHETIC_CASES = [
    ("g1_hit", case_g1_hit),
    ("g1_clean", case_g1_clean),
    ("g2_hit", case_g2_hit),
    ("g2_clean", case_g2_clean),
    ("g3_hit", case_g3_hit),
    ("g3_clean", case_g3_clean),
    ("g4_hit", case_g4_hit),
    ("g4_clean", case_g4_clean),
    ("g5_hit", case_g5_hit),
    ("g5_clean", case_g5_clean),
]


def test_detection() -> None:
    failures: list[str] = []
    for name, factory in SYNTHETIC_CASES:
        traj, final, expected_subset = factory()
        got = detect_gotcha_errors(traj, final)
        if not expected_subset.issubset(got):
            failures.append(
                f"{name}: expected SUPERSET of {expected_subset}, got {got}"
            )
        if name.endswith("_clean") and got:
            # Clean cases should hit *nothing*
            failures.append(f"{name}: clean trajectory but detected {got}")
        print(f"  {name}: detected={got} (expected ⊇ {expected_subset})")
    if failures:
        raise SystemExit("DETECTION FAILURES:\n  " + "\n  ".join(failures))
    print("DETECTION OK")


# ---------------------------------------------------------------------------
# Verifier round-trip: ensure each task.verify() passes on a correct
# trajectory we construct here (we don't need to invoke the LLM).
# ---------------------------------------------------------------------------


def _correct_trajectories() -> dict[str, tuple[str, list[dict]]]:
    """Map task.id -> (final_answer, trajectory) that should pass verify()."""
    return {
        "T01_secret_token": ("The secret is alpha-9er.", []),
        "T02_birthday": ("The user's birthday is 1985-03-22.", []),
        "T03_count_lines": ("The log file has 7 lines.", []),
        "T04_fib_split": ("fib(20) = 6765", []),
        "T05_squares_sum": ("sum = 385", []),
        "T06_sha_digest": (
            "ae16d2bbf81717eba2c1fbab5b4bf32c2c2cb2eef33c7c20b7c9aac9b8f63b22",
            [],
        ),
        "T07_sum_range": ("The value is 5050.", []),
        "T08_build_status": ("The build did not succeed (exit code 1).", []),
        "T09_test_status": ("Tests did not pass (exit code 2).", []),
        "T10_grep_miss": ("No, the file does not contain 'foo'.", []),
        "T11_append_create": ("done", []),  # filesystem mutation happens in setup-fixture below
        "T12_append_existing": ("done", []),
    }


def test_verifiers() -> None:
    import hashlib
    correct_sha = hashlib.sha256(b"reasoning").hexdigest()
    answers = _correct_trajectories()
    answers["T06_sha_digest"] = (correct_sha, [])

    failures: list[str] = []
    for task in TASKS:
        sandbox = Path(tempfile.mkdtemp())
        try:
            task.setup(sandbox)
            # For T11/T12, simulate a correct end-state on disk.
            if task.id == "T11_append_create":
                (sandbox / "log.txt").write_text("first entry\n")
            if task.id == "T12_append_existing":
                # setup wrote "line one\nline two\n"; correct end-state appends
                (sandbox / "log.txt").write_text("line one\nline two\nnew entry\n")
            final, traj = answers[task.id]
            ok = task.verify(final, sandbox, traj)
            if not ok:
                failures.append(f"{task.id}: verify failed on correct answer {final!r}")
            print(f"  {task.id}: verify_correct={ok}")
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)
    if failures:
        raise SystemExit("VERIFIER FAILURES:\n  " + "\n  ".join(failures))
    print("VERIFIER OK")


if __name__ == "__main__":
    print("== detection ==")
    test_detection()
    print("\n== verifiers ==")
    test_verifiers()
