"""Finish the partial probe by gathering existing records + running swebench eval."""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
RUN_DIR = ROOT / "run_data" / "probe_20260528_165740"


def main() -> None:
    records = []
    for p in sorted(RUN_DIR.glob("*/record.json")):
        rec = json.loads(p.read_text())
        records.append(rec)
    print(f"loaded {len(records)} records")

    preds_path = RUN_DIR / "predictions.jsonl"
    with preds_path.open("w") as f:
        for rec in records:
            slim = {
                "instance_id": rec["instance_id"],
                "model_patch": rec.get("model_patch", ""),
                "model_name_or_path": "haiku-4-5-agent",
            }
            f.write(json.dumps(slim) + "\n")
    print(f"wrote {preds_path}")

    summary = {
        "n_instances": len(records),
        "n_with_patch": sum(1 for r in records if r.get("model_patch")),
        "n_errored": sum(1 for r in records if r.get("error")),
        "instances": [
            {
                "instance_id": r["instance_id"],
                "patch_len": len(r.get("model_patch", "")),
                "steps": r["stats"].get("steps", 0),
                "input_tokens": r["stats"].get("input_tokens", 0),
                "output_tokens": r["stats"].get("output_tokens", 0),
                "wall_s": r["stats"].get("wall_s", 0),
                "error": r.get("error"),
            }
            for r in records
        ],
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    instance_ids = [r["instance_id"] for r in records]
    env = os.environ.copy()
    env.setdefault("DOCKER_HOST", "unix:///Users/ram/.colima/default/docker.sock")
    cmd = [
        ".venv/bin/python", "-m", "swebench.harness.run_evaluation",
        "-p", str(preds_path),
        "-id", "probe_eval_partial",
        "-i", *instance_ids,
        "--max_workers", "2",
        "--cache_level", "instance",
    ]
    print("running swebench eval...")
    print(" ".join(cmd))
    cwd = Path("/Users/ram/reasoning-bank")
    r = subprocess.run(cmd, cwd=cwd, env=env, timeout=7200)
    print("swebench done, returncode =", r.returncode)


if __name__ == "__main__":
    main()
