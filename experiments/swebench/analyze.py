"""Analyse the cell ledger.

Reports, per arm:
  - per-seed pass rate, mean ± std across seeds
  - attempt-1 pass rate
  - mean attempts-to-pass (over resolved cells)
  - mean total input tokens per cell

Then the key contrasts:
  - Arm3a - Arm2 delta in final-pass rate (THE main signal)
  - Arm3b first-pass rate over instance order (does the persistent bank
    lift attempt-1?)
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.swebench import ledger as L

ROOT = Path(__file__).resolve().parent


def _load_all_cells() -> list[dict]:
    cells = []
    if not L.LEDGER_DIR.exists():
        return cells
    for f in L.LEDGER_DIR.glob("*.json"):
        try:
            cells.append(json.loads(f.read_text()))
        except Exception as e:
            print(f"skip {f}: {e}", file=sys.stderr)
    return cells


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def main() -> None:
    cells = _load_all_cells()
    print(f"loaded {len(cells)} cells from {L.LEDGER_DIR}")
    if not cells:
        return

    by_arm = defaultdict(list)
    for c in cells:
        by_arm[c["arm"]].append(c)

    print("\n=== per-arm summary ===")
    print(f"{'arm':22s}  {'n':>4s}  {'first%':>7s}  {'final%':>7s}  "
          f"{'mean_atts':>9s}  {'tok/cell':>10s}")
    summary = {}
    for arm in sorted(by_arm):
        rows = by_arm[arm]
        n = len(rows)
        first_rate = sum(1 for r in rows if r.get("first_pass")) / n
        final_rate = sum(1 for r in rows if r.get("final_pass") or r.get("resolved")) / n
        passed = [r for r in rows if r.get("final_pass") or r.get("resolved")]
        mean_atts = (sum(r.get("attempts", 1) for r in passed) / len(passed)) if passed else float("nan")
        toks = []
        for r in rows:
            per = r.get("per_attempt") or []
            toks.append(sum(a.get("input_tokens", 0) for a in per))
        mean_tok = sum(toks) / n if toks else 0
        summary[arm] = {
            "n": n,
            "first_pass_rate": first_rate,
            "final_pass_rate": final_rate,
            "mean_attempts_to_pass": (None if math.isnan(mean_atts) else round(mean_atts, 2)),
            "mean_input_tokens": int(mean_tok),
        }
        ma = f"{mean_atts:.2f}" if not math.isnan(mean_atts) else "—"
        print(f"{arm:22s}  {n:>4d}  {first_rate*100:>6.1f}%  "
              f"{final_rate*100:>6.1f}%  {ma:>9s}  {int(mean_tok):>10d}")

    print("\n=== final-pass rate per seed (across instances) ===")
    by_arm_seed = defaultdict(lambda: defaultdict(list))
    for c in cells:
        by_arm_seed[c["arm"]][c["seed"]].append(c)
    print(f"{'arm':22s}  {'seed':>4s}  {'n':>4s}  {'final%':>7s}")
    seed_rates = defaultdict(list)
    for arm in sorted(by_arm_seed):
        for seed in sorted(by_arm_seed[arm]):
            rs = by_arm_seed[arm][seed]
            rate = sum(1 for r in rs if r.get("final_pass") or r.get("resolved")) / len(rs)
            seed_rates[arm].append(rate)
            print(f"{arm:22s}  {seed:>4d}  {len(rs):>4d}  {rate*100:>6.1f}%")
    print("\n=== final-pass mean ± std across seeds ===")
    for arm in sorted(seed_rates):
        m, s = _mean_std(seed_rates[arm])
        print(f"  {arm:22s}  {m*100:>5.1f}% ± {s*100:.1f}  (n={len(seed_rates[arm])} seeds)")

    print("\n=== KEY DELTAS ===")
    if "arm2_naive" in summary and "arm3a_bank_fresh" in summary:
        d = summary["arm3a_bank_fresh"]["final_pass_rate"] - summary["arm2_naive"]["final_pass_rate"]
        print(f"  Arm3a - Arm2 (final-pass): {d*100:+.1f} pp")
    if "arm1_noretry" in summary and "arm2_naive" in summary:
        d = summary["arm2_naive"]["final_pass_rate"] - summary["arm1_noretry"]["final_pass_rate"]
        print(f"  Arm2 - Arm1 (lift from naive retry): {d*100:+.1f} pp")
    if "arm3a_bank_fresh" in summary and "arm3b_bank_persist" in summary:
        d = summary["arm3b_bank_persist"]["final_pass_rate"] - summary["arm3a_bank_fresh"]["final_pass_rate"]
        print(f"  Arm3b - Arm3a (persistence lift): {d*100:+.1f} pp")

    print("\n=== Arm3b first-pass over instance order ===")
    if "arm3b_bank_persist" in by_arm_seed:
        instances_path = Path(ROOT / "phase3_instances.json")
        if instances_path.exists():
            inst_list = json.loads(instances_path.read_text())
            inst_order = {inst["instance_id"]: i for i, inst in enumerate(inst_list)}
        else:
            inst_order = {}
        for seed in sorted(by_arm_seed["arm3b_bank_persist"]):
            rs = by_arm_seed["arm3b_bank_persist"][seed]
            ordered = sorted(rs, key=lambda r: inst_order.get(r["instance_id"], 0))
            seq = "".join("Y" if r.get("first_pass") else "n" for r in ordered)
            print(f"  seed={seed}: {seq}")

    out = ROOT / "run_data" / "phase3_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
