"""Phase 3 driver — resumable, chunked.

Each (instance, arm, seed) cell is independent and written atomically to the
ledger as soon as it finishes. Re-running the script picks up where the last
session stopped: any cell with a completed file is skipped.

Run with:

    SESSION_MAX_MINUTES=90 .venv/bin/python experiments/swebench/run_phase3_resumable.py

Env vars:
    SESSION_MAX_MINUTES   wall-clock cap (default 60); a SIGINT or the cap
                          ends the session cleanly between cells.
    PHASE3_INSTANCES      cap how many of the 30 instances to consider
                          (default: all). Useful for dry-runs.
    PHASE3_ARMS           comma list of arm names to enable (default all 4)
    PHASE3_SEEDS          comma list of seeds (default 0,1,2)
    PHASE3_INSTANCES_FILE override path to phase3_instances.json
    PHASE3_BANK_DB_PATH   override path to the persistent SQLite bank
    PHASE3_BUDGET_LIMIT   per-minute input-token cap (default 45000)
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from importlib.util import find_spec
from pathlib import Path

assert find_spec("fleet") is None, "fleet must not be importable"

# Ensure repo root is on sys.path for absolute imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from anthropic import AsyncAnthropic

from experiments.swebench.arms_resumable import (
    arm1_noretry, arm2_naive, arm3a_bank_fresh, arm3b_bank_persist,
    bank_llm_factory,
)
from experiments.swebench.budget import TokenBudget
from experiments.swebench.harness import _ensure_repo_cache
from experiments.swebench import ledger as L

from reasoning_bank import ReasoningBank, InMemoryStore
from reasoning_bank.stores import SQLiteVecStore

ROOT = Path(__file__).resolve().parent
RUN_DATA = ROOT / "run_data"
RUN_DATA.mkdir(parents=True, exist_ok=True)


ARM_FUNCS = {
    "arm1_noretry": arm1_noretry,
    "arm2_naive": arm2_naive,
    "arm3a_bank_fresh": arm3a_bank_fresh,
    "arm3b_bank_persist": arm3b_bank_persist,
}


def _load_instances() -> list[dict]:
    path = Path(os.environ.get("PHASE3_INSTANCES_FILE",
                               ROOT / "phase3_instances.json"))
    data = json.loads(path.read_text())
    n_cap = os.environ.get("PHASE3_INSTANCES")
    if n_cap:
        data = data[: int(n_cap)]
    return data


def _list_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [s.strip() for s in raw.split(",") if s.strip()]


_stop_requested = False


def _install_signal_handler() -> None:
    def handler(signum, frame):
        global _stop_requested
        _stop_requested = True
        print("\n[signal] stop requested — will exit after the current cell", flush=True)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


async def _run_cell(
    inst: dict, arm: str, seed: int, *,
    client: AsyncAnthropic, budget: TokenBudget, bank_llm,
    persistent_bank: ReasoningBank | None,
    workroot: Path, run_id_base: str,
) -> dict:
    iid = inst["instance_id"]
    workdir = workroot / arm / iid / f"s{seed}"
    workdir.mkdir(parents=True, exist_ok=True)
    rcid = f"{run_id_base}_{arm}_{iid}_s{seed}"
    fn = ARM_FUNCS[arm]
    if arm == "arm1_noretry":
        return await fn(inst, seed=seed, budget=budget, client=client,
                        workdir=workdir, run_id=rcid)
    if arm == "arm2_naive":
        return await fn(inst, seed=seed, budget=budget, client=client,
                        workdir=workdir, run_id=rcid)
    if arm == "arm3a_bank_fresh":
        bank = ReasoningBank(
            llm=bank_llm,
            store=InMemoryStore(),
            scope=f"p3-{iid}-s{seed}",
        )
        return await fn(inst, seed=seed, budget=budget, client=client,
                        workdir=workdir, run_id=rcid, bank=bank)
    if arm == "arm3b_bank_persist":
        assert persistent_bank is not None
        return await fn(inst, seed=seed, budget=budget, client=client,
                        workdir=workdir, run_id=rcid, bank=persistent_bank)
    raise ValueError(arm)


async def main() -> None:
    _install_signal_handler()

    instances = _load_instances()
    arms = _list_env("PHASE3_ARMS", list(ARM_FUNCS))
    seeds = [int(s) for s in _list_env("PHASE3_SEEDS", ["0", "1", "2"])]
    session_max_min = float(os.environ.get("SESSION_MAX_MINUTES", "60"))
    budget_limit = int(os.environ.get("PHASE3_BUDGET_LIMIT", "45000"))

    print(f"phase3-resumable: {len(instances)} instances × {len(arms)} arms "
          f"× {len(seeds)} seeds = {len(instances)*len(arms)*len(seeds)} cells")
    print(f"arms: {arms}")
    print(f"seeds: {seeds}")
    print(f"session cap: {session_max_min} min")
    print(f"budget: {budget_limit} input-tokens/min")

    done, total = L.progress(instances, arms, seeds)
    print(f"ledger: {done}/{total} cells already done")
    pending = [c for c in L.pending_cells(instances, arms, seeds)]
    print(f"pending: {len(pending)} cells")

    if not pending:
        print("nothing to do.")
        return

    # Repo cache warm-up — done once even if no cells run.
    seen = set()
    for inst in instances:
        if inst["repo"] not in seen:
            try:
                _ensure_repo_cache(inst["repo"])
            except Exception as e:
                print(f"  [warn] could not cache {inst['repo']}: {e}", flush=True)
            seen.add(inst["repo"])

    client = AsyncAnthropic()
    budget = TokenBudget(limit_per_min=budget_limit)
    bank_llm = bank_llm_factory(client, budget=budget)

    # Set up the persistent (3b) bank lazily — one DB for the whole project.
    persistent_bank = None
    if "arm3b_bank_persist" in arms:
        db_path = Path(os.environ.get(
            "PHASE3_BANK_DB_PATH",
            RUN_DATA / "phase3_bank.sqlite",
        ))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        persistent_bank = ReasoningBank(
            llm=bank_llm,
            store=SQLiteVecStore(db_path=str(db_path)),
            scope="phase3-3b-global",
        )
        print(f"persistent bank: {db_path}")

    run_id_base = "p3_" + time.strftime("%Y%m%d_%H%M%S")
    workroot = RUN_DATA / "phase3_work"
    workroot.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    session_deadline = t_start + session_max_min * 60.0
    completed_this_session = 0

    # Cell ordering rule:
    #  - For arm3b_bank_persist, the bank only learns if cells run in a FIXED
    #    instance order. We sort 3b cells by (seed, instance_index) and keep
    #    them grouped; other arms can run in the natural order produced by
    #    pending_cells.
    arm3b_cells = []
    other_cells = []
    inst_order = {inst["instance_id"]: i for i, inst in enumerate(instances)}
    for inst, arm, seed in pending:
        if arm == "arm3b_bank_persist":
            arm3b_cells.append((inst, arm, seed))
        else:
            other_cells.append((inst, arm, seed))
    arm3b_cells.sort(key=lambda c: (c[2], inst_order[c[0]["instance_id"]]))
    # Run non-3b cells first (they're independent), then 3b. That way if we
    # ever get partial 3b progress we can still meaningfully analyse the
    # other arms.
    ordered = other_cells + arm3b_cells

    for inst, arm, seed in ordered:
        if _stop_requested:
            print("[stop] session stop requested.", flush=True)
            break
        if time.monotonic() >= session_deadline:
            print(f"[stop] session cap of {session_max_min} min reached.", flush=True)
            break
        # Skip if it became done somehow (e.g. external write)
        if L.is_done(inst["instance_id"], arm, seed):
            continue
        iid = inst["instance_id"]
        print(f"\n=== cell {arm} | {iid} | seed={seed} "
              f"(elapsed {(time.monotonic()-t_start)/60:.1f} min, "
              f"done {completed_this_session}) ===", flush=True)
        t_cell = time.monotonic()
        try:
            result = await _run_cell(
                inst, arm, seed,
                client=client, budget=budget, bank_llm=bank_llm,
                persistent_bank=persistent_bank,
                workroot=workroot, run_id_base=run_id_base,
            )
        except KeyboardInterrupt:
            print("[interrupt] cell aborted, leaving pending.", flush=True)
            break
        except Exception as e:
            print(f"[cell-error] {type(e).__name__}: {e}", flush=True)
            # Leave the cell pending — it'll be retried next session.
            continue
        result["wall_s"] = round(time.monotonic() - t_cell, 2)
        result["run_id_base"] = run_id_base
        L.save_cell(iid, arm, seed, result)
        completed_this_session += 1
        print(
            f"  -> resolved={result.get('resolved')} "
            f"attempts={result.get('attempts')} "
            f"wall={result['wall_s']}s "
            f"budget_pauses={budget.pause_count}",
            flush=True,
        )

    # Session summary
    done, total = L.progress(instances, arms, seeds)
    elapsed_min = (time.monotonic() - t_start) / 60.0
    print("\n=== SESSION REPORT ===")
    print(f"Cells this session:   {completed_this_session} completed")
    print(f"Total progress:       {done}/{total}")
    if completed_this_session > 0:
        rate = elapsed_min / completed_this_session
        remaining = total - done
        eta_min = remaining * rate
        print(f"Cells/min observed:   {1/rate:.2f}")
        print(f"ETA at this rate:     ~{eta_min/60:.1f} hours wall-clock")
    bstats = budget.stats()
    print(f"Budget stats:         "
          f"{bstats['total_input_tokens']} input tokens over "
          f"{bstats['n_calls']} calls, "
          f"{bstats['pause_count']} pauses ({bstats['total_pause_s']}s)")


if __name__ == "__main__":
    asyncio.run(main())
