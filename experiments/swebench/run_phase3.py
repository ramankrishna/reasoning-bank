"""Phase 3 driver. Runs three arms over N instances × 3 seeds.

Each (instance, arm, seed) cell runs sequentially within an arm; arms run
sequentially (Arm 1, then 2, then 3a, then 3b). 3b shares a bank across seeds
within a seed-replay block.

NOTE: API non-determinism comes from temperature=0.0 + provider variation. We
re-run 3 times and report mean.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from anthropic import AsyncAnthropic

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.swebench.arms import (
    run_no_retry, run_naive_retry, run_bank_retry, _bank_llm,
)
from experiments.swebench.harness import _ensure_repo_cache
from reasoning_bank import ReasoningBank, InMemoryStore

ROOT = Path(__file__).parent
RUN_ID = "phase3_" + time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "run_data" / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_instances(n: int) -> list[dict]:
    """Take the probe 10 + first 20 of remaining by repo round-robin."""
    probe = json.loads((ROOT / "probe_instances.json").read_text())
    full = json.loads((ROOT / "all_instances.json").read_text())
    have = {p["instance_id"] for p in probe}
    by_repo: dict[str, list[dict]] = {}
    for ex in full:
        if ex["instance_id"] in have:
            continue
        by_repo.setdefault(ex["repo"], []).append(ex)
    for r in by_repo:
        by_repo[r] = sorted(by_repo[r], key=lambda x: len(x.get("patch", "")))
    extras: list[dict] = []
    idx = 0
    while len(extras) < (n - len(probe)):
        added = False
        for repo in sorted(by_repo):
            if idx < len(by_repo[repo]) and len(extras) < (n - len(probe)):
                extras.append(by_repo[repo][idx])
                added = True
        if not added:
            break
        idx += 1
    return probe + extras


async def main() -> None:
    n_instances = int(os.environ.get("PHASE3_N", "30"))
    n_seeds = int(os.environ.get("PHASE3_SEEDS", "3"))
    instances = _load_instances(n_instances)
    print(f"using {len(instances)} instances, {n_seeds} seeds")
    print(f"run_id={RUN_ID}")

    seen = set()
    for inst in instances:
        if inst["repo"] not in seen:
            _ensure_repo_cache(inst["repo"])
            seen.add(inst["repo"])

    client = AsyncAnthropic()
    bank_llm = _bank_llm(client)

    all_results: dict[str, list[dict]] = {
        "arm1_no_retry": [],
        "arm2_naive_retry": [],
        "arm3a_bank_fresh": [],
        "arm3b_bank_persist": [],
    }

    for seed in range(n_seeds):
        seed_dir = OUT_DIR / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Arm 1 — no retry
        for inst in instances:
            iid = inst["instance_id"]
            print(f"[seed{seed}][arm1] {iid}")
            wd = seed_dir / "arm1" / iid
            wd.mkdir(parents=True, exist_ok=True)
            r = await run_no_retry(inst, client=client, workdir=wd,
                                   run_id=f"{RUN_ID}_s{seed}_arm1")
            r["seed"] = seed
            all_results["arm1_no_retry"].append(r)
            (seed_dir / "arm1_results.jsonl").open("a").write(json.dumps(r) + "\n")
            print(f"  -> {r['first_pass']}")

        # Arm 2 — naive retry
        for inst in instances:
            iid = inst["instance_id"]
            print(f"[seed{seed}][arm2] {iid}")
            wd = seed_dir / "arm2" / iid
            wd.mkdir(parents=True, exist_ok=True)
            r = await run_naive_retry(inst, client=client, workdir=wd,
                                      run_id=f"{RUN_ID}_s{seed}_arm2")
            r["seed"] = seed
            all_results["arm2_naive_retry"].append(r)
            (seed_dir / "arm2_results.jsonl").open("a").write(json.dumps(r) + "\n")
            print(f"  first={r['first_pass']} final={r['final_pass']}")

        # Arm 3a — bank, fresh per instance
        for inst in instances:
            iid = inst["instance_id"]
            print(f"[seed{seed}][arm3a] {iid}")
            wd = seed_dir / "arm3a" / iid
            wd.mkdir(parents=True, exist_ok=True)
            bank = ReasoningBank(llm=bank_llm, store=InMemoryStore(),
                                 scope=f"phase3-s{seed}-{iid}")
            r = await run_bank_retry(inst, client=client, workdir=wd,
                                     run_id=f"{RUN_ID}_s{seed}_arm3a",
                                     bank=bank, arm_label="bank_fresh")
            r["seed"] = seed
            all_results["arm3a_bank_fresh"].append(r)
            (seed_dir / "arm3a_results.jsonl").open("a").write(json.dumps(r) + "\n")
            print(f"  first={r['first_pass']} final={r['final_pass']}")

        # Arm 3b — bank, persistent across instances within this seed
        bank = ReasoningBank(llm=bank_llm, store=InMemoryStore(),
                             scope=f"phase3-s{seed}-persist")
        for inst in instances:
            iid = inst["instance_id"]
            print(f"[seed{seed}][arm3b] {iid}")
            wd = seed_dir / "arm3b" / iid
            wd.mkdir(parents=True, exist_ok=True)
            r = await run_bank_retry(inst, client=client, workdir=wd,
                                     run_id=f"{RUN_ID}_s{seed}_arm3b",
                                     bank=bank, arm_label="bank_persist")
            r["seed"] = seed
            all_results["arm3b_bank_persist"].append(r)
            (seed_dir / "arm3b_results.jsonl").open("a").write(json.dumps(r) + "\n")
            print(f"  first={r['first_pass']} final={r['final_pass']} retr_pre={r['retrieved_pre']}")

    (OUT_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))
    # Summary
    summary: dict = {}
    for arm, rows in all_results.items():
        if not rows:
            continue
        n = len(rows)
        first = sum(1 for r in rows if r["first_pass"]) / n
        final = sum(1 for r in rows if r["final_pass"]) / n
        summary[arm] = {"n": n, "first_pass_rate": first, "final_pass_rate": final}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
