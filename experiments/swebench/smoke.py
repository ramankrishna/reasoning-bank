"""Smoke test: run the agent on ONE instance, check we get a non-empty patch."""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.swebench.harness import run_one_instance

ROOT = Path(__file__).parent


async def main() -> None:
    probe = json.loads((ROOT / "probe_instances.json").read_text())
    # Pick the smallest patch as smoke
    inst = min(probe, key=lambda x: len(x.get("patch", "")))
    print(f"smoke instance: {inst['instance_id']} (patch_len={len(inst['patch'])})")
    print(f"problem statement (first 500 chars):\n{inst['problem_statement'][:500]}\n")

    wd = ROOT / "run_data" / "smoke"
    wd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rec = await run_one_instance(inst, workdir=wd, max_steps=30)
    (wd / "record.json").write_text(json.dumps(rec, default=str))
    print(f"\ndone in {time.time()-t0:.1f}s")
    print(f"steps: {rec['stats']['steps']}")
    print(f"stopped_by: {rec['stats']['stopped_by']}")
    print(f"input tokens: {rec['stats']['input_tokens']}")
    print(f"output tokens: {rec['stats']['output_tokens']}")
    print(f"patch length: {len(rec.get('model_patch',''))}")
    print(f"final text: {rec.get('final_text','')[:200]}")
    if rec.get("model_patch"):
        print(f"\n=== PATCH ===\n{rec['model_patch'][:3000]}")
    if rec.get("error"):
        print(f"\nERROR: {rec['error']}")


if __name__ == "__main__":
    asyncio.run(main())
