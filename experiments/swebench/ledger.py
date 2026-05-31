"""Cell ledger for resumable Phase 3.

A "cell" is one (instance, arm, seed) trial. The driver checks the ledger
before running a cell; if it has a complete record on disk, it skips. Any
unfinished cell (no file, or file missing status=complete) is retried on
the next session.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

LEDGER_DIR = Path("experiments/swebench/run_data/phase3_cells")


def cell_id(instance_id: str, arm: str, seed: int) -> str:
    return f"{arm}__{instance_id}__s{seed}"


def cell_path(instance_id: str, arm: str, seed: int) -> Path:
    return LEDGER_DIR / f"{cell_id(instance_id, arm, seed)}.json"


def is_done(instance_id: str, arm: str, seed: int) -> bool:
    p = cell_path(instance_id, arm, seed)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
        return data.get("status") == "complete"
    except (json.JSONDecodeError, OSError):
        return False


def save_cell(instance_id: str, arm: str, seed: int, payload: dict[str, Any]) -> Path:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    final_path = cell_path(instance_id, arm, seed)
    payload = dict(payload)
    payload.setdefault("status", "complete")
    payload.setdefault("cell_id", cell_id(instance_id, arm, seed))
    payload.setdefault("instance_id", instance_id)
    payload.setdefault("arm", arm)
    payload.setdefault("seed", seed)
    fd, tmp_name = tempfile.mkstemp(
        prefix=final_path.name + ".",
        suffix=".tmp",
        dir=str(LEDGER_DIR),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_name, final_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return final_path


def load_cell(instance_id: str, arm: str, seed: int) -> dict[str, Any] | None:
    p = cell_path(instance_id, arm, seed)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def all_cells(
    instances: list[dict], arms: list[str], seeds: list[int]
) -> Iterable[tuple[dict, str, int]]:
    for arm in arms:
        for seed in seeds:
            for inst in instances:
                yield inst, arm, seed


def pending_cells(
    instances: list[dict], arms: list[str], seeds: list[int]
) -> Iterable[tuple[dict, str, int]]:
    for inst, arm, seed in all_cells(instances, arms, seeds):
        if not is_done(inst["instance_id"], arm, seed):
            yield inst, arm, seed


def progress(instances: list[dict], arms: list[str], seeds: list[int]) -> tuple[int, int]:
    total = 0
    done = 0
    for inst, arm, seed in all_cells(instances, arms, seeds):
        total += 1
        if is_done(inst["instance_id"], arm, seed):
            done += 1
    return done, total
