"""Phase 2: run the 10 probe instances with the agent, no bank, no retry."""
from __future__ import annotations

import asyncio
import functools
import json
import os
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)  # type: ignore[assignment]

from anthropic import AsyncAnthropic

# Make experiments.swebench importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.swebench.harness import (
    prepare_workdir, extract_patch, run_one_instance,
    write_predictions, run_swebench_eval,
)

ROOT = Path(__file__).parent
PROBE = ROOT / "probe_instances.json"
RUN_ID = "probe_" + time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "run_data" / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)


async def run_all() -> None:
    probe = json.loads(PROBE.read_text())
    # Allow narrowing to first N for a quick signal
    n_limit = int(os.environ.get("PROBE_N", str(len(probe))))
    probe = probe[:n_limit]
    print(f"running {len(probe)} probe instances, run_id={RUN_ID}")

    # Pre-warm repo caches sequentially to avoid concurrent clones blowing things up
    from experiments.swebench.harness import _ensure_repo_cache
    seen = set()
    for inst in probe:
        if inst["repo"] not in seen:
            _ensure_repo_cache(inst["repo"])
            seen.add(inst["repo"])

    client = AsyncAnthropic()
    records: list[dict] = []

    # Concurrency=1 to respect 50k input tok/min rate limit.
    sem = asyncio.Semaphore(1)

    async def run_with_limit(inst: dict) -> dict:
        async with sem:
            iid = inst["instance_id"]
            wd = OUT_DIR / iid
            print(f"  [{iid}] starting agent...")
            t0 = time.time()
            max_steps = int(os.environ.get("PROBE_MAX_STEPS", "20"))
            rec = await run_one_instance(inst, workdir=wd, client=client, max_steps=max_steps)
            print(f"  [{iid}] done in {time.time()-t0:.1f}s, "
                  f"patch_len={len(rec.get('model_patch',''))}, "
                  f"steps={rec['stats']['steps']}")
            # Save the full record (with trajectory) for later analysis
            (wd / "record.json").write_text(json.dumps(rec, default=str))
            return rec

    records = await asyncio.gather(*[run_with_limit(inst) for inst in probe])

    # Write predictions for swebench
    preds_path = OUT_DIR / "predictions.jsonl"
    write_predictions(records, preds_path)
    print(f"\nwrote predictions to {preds_path}")

    # Save summary
    summary = {
        "run_id": RUN_ID,
        "n_instances": len(records),
        "n_with_patch": sum(1 for r in records if r.get("model_patch")),
        "n_errored": sum(1 for r in records if r.get("error")),
        "instances": [
            {
                "instance_id": r["instance_id"],
                "patch_len": len(r.get("model_patch", "")),
                "steps": r["stats"]["steps"],
                "input_tokens": r["stats"]["input_tokens"],
                "output_tokens": r["stats"]["output_tokens"],
                "wall_s": r["stats"]["wall_s"],
                "error": r.get("error"),
            }
            for r in records
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Now run swebench evaluation
    instance_ids = [r["instance_id"] for r in records]
    print(f"\nrunning swebench eval...")
    report = run_swebench_eval(preds_path, instance_ids, RUN_ID, max_workers=2)
    (OUT_DIR / "swebench_report.json").write_text(json.dumps(report, indent=2))
    print("=" * 60)
    print(json.dumps(report, indent=2)[:2000])


if __name__ == "__main__":
    asyncio.run(run_all())
