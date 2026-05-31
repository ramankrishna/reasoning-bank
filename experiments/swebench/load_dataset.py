"""Load SWE-bench-lite and pick the probe set."""
import json
from pathlib import Path
from datasets import load_dataset

OUT = Path(__file__).parent / "probe_instances.json"


def main() -> None:
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    print(f"loaded {len(ds)} instances")

    # Stratify by repo
    by_repo: dict[str, list[dict]] = {}
    for ex in ds:
        by_repo.setdefault(ex["repo"], []).append(ex)
    print("repos:", {k: len(v) for k, v in by_repo.items()})

    # Pick probe set: one per repo until we hit 10, smallest patches first
    probe: list[dict] = []
    repos_seen: set[str] = set()
    pool = []
    for repo, items in by_repo.items():
        items = sorted(items, key=lambda x: len(x.get("patch", "")))
        pool.extend(items)
    pool = sorted(pool, key=lambda x: (x["repo"], len(x.get("patch", ""))))

    # Round-robin: smallest patch from each repo, then 2nd smallest, etc.
    rr: dict[str, list[dict]] = {}
    for ex in pool:
        rr.setdefault(ex["repo"], []).append(ex)
    idx = 0
    while len(probe) < 10:
        added = False
        for repo, items in rr.items():
            if idx < len(items) and len(probe) < 10:
                probe.append(items[idx])
                added = True
        if not added:
            break
        idx += 1

    print(f"probe set: {len(probe)} instances")
    for ex in probe:
        print(f"  - {ex['instance_id']} ({ex['repo']}) patch_len={len(ex.get('patch',''))}")

    # Serialize (only the fields we need; HF Dataset row -> dict)
    keep_fields = [
        "instance_id", "repo", "base_commit", "patch", "test_patch",
        "problem_statement", "hints_text", "created_at", "version",
        "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit",
    ]
    out = []
    for ex in probe:
        out.append({k: ex.get(k) for k in keep_fields})
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")

    # Also write the FULL dataset to a json (for Phase 3 — pick 20 more)
    full_out = Path(__file__).parent / "all_instances.json"
    full = []
    for ex in ds:
        full.append({k: ex.get(k) for k in keep_fields})
    full_out.write_text(json.dumps(full, indent=2))
    print(f"wrote {full_out} ({len(full)} instances)")


if __name__ == "__main__":
    main()
