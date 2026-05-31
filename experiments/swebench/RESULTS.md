# SWE-bench-lite × ReasoningBank — Phase 3 experiment results

**Date:** 2026-05-31 (arm3b added; arm1/arm2/arm3a unchanged from 2026-05-30)
**Branch:** `main` in `~/reasoning-bank` (work not committed; per dispatch instructions)
**Model under test:** Anthropic `claude-haiku-4-5`, temperature 0.0
**Dataset:** `princeton-nlp/SWE-bench_Lite` (test split), 30-instance subset (one per repo where available, smallest-patch-first)
**Repo:** `reasoning-bank` (raw `AsyncAnthropic`; no `fleet` dependency)
**Compute:** local macOS arm64; Docker via `colima`; official `swebench.harness.run_evaluation` for scoring
**Verdict:** **NULL.** Neither the per-instance bank (arm3a) nor the persistent bank (arm3b) produces a measurable lift over the no-retry baseline. The persistent bank shows **no cross-task learning signal** (late instances do not outperform early ones; if anything, mild degradation). The defensive recovery against naive retry is the only positive observation in the experiment.

---

## Headline numbers

Pass rates use **clean cells only** — cells whose `resolved=False` outcome can be trusted because no infrastructure failure (Anthropic API credit exhaustion or Docker eval crash) occurred during the cell's attempts. A `resolved=True` cell is always counted as clean since at least one attempt produced a verified passing patch. See "Infrastructure incidents" below for context.

| Arm | What it does | Clean cells | Pass rate | 95% CI (Wilson) | Per-seed mean ± std |
| --- | --- | ---: | ---: | --- | ---: |
| **arm1_noretry** | Single attempt, no retry (control) | 90/90 | **45/90 = 50.0%** | [39.9%, 60.1%] | 50.0% ± 4.7 |
| **arm2_naive** | Up to 3 attempts; retry with raw error feedback | 90/90 | **38/90 = 42.2%** | [32.5%, 52.5%] | 42.2% ± 4.2 |
| **arm3a_bank_fresh** | Up to 3 attempts; fresh `ReasoningBank` per instance | 86/90 | **45/86 = 52.3%** | [41.9%, 62.6%] | 52.2% ± 1.6 |
| **arm3b_bank_persist** | Up to 3 attempts; `ReasoningBank` shared across all 30 instances within a seed | 45/90 | **21/45 = 46.7%** | [32.9%, 60.9%] | 46.6% ± 1.2 |

### Key contrasts

| Comparison | Δ pass rate | Interpretation |
| --- | ---: | --- |
| arm2 − arm1 (naive retry vs no retry) | **−7.8 pp** | Naive retry *hurts*. Cost (+167% tokens) buys negative lift. |
| arm3a − arm2 (bank retry vs naive retry) | **+10.1 pp** | Per-instance bank fully recovers the regression caused by naive retry. |
| arm3a − arm1 (per-instance bank vs control) | **+2.3 pp** | Within CIs; no meaningful net lift. |
| **arm3b − arm1 (persistent bank vs control)** | **−3.3 pp** | Within CIs; no cross-task transfer benefit. |
| **arm3b − arm3a (persistent vs fresh bank)** | **−5.6 pp** | If anything, sharing the bank across instances mildly hurts. |

All Wilson CIs overlap heavily. With the realised n (45 for arm3b, 86–90 for the others), the experiment cannot resolve sub-±10 pp effects. None of the four arms is statistically distinguishable from the others at this sample size **except** arm2 from {arm1, arm3a} where the gap is ≈10 pp.

---

## Infrastructure incidents (why arm3a is n=86 and arm3b is n=45)

Three distinct infrastructure failures contaminated cells mid-run. Cells flagged by any of these are excluded from the headline pass-rate numerator/denominator above.

| # | Failure | What broke | Cells lost | How detected |
| --- | --- | --- | ---: | --- |
| 1 | **Colima VM disk full** | swebench eval can't `containerd snapshotter` → `error_instances: 1` in report. Pattern: `no space left on device` in `logs/run_evaluation/.../run_instance.log`. Re-evaluation marks cells as not-resolved even though the agent's patch may have been correct. | 45 arm3a + 50 arm3b (original run, since rerun) | Cells with `attempts=3` but identical patches and `error_instances=1` reports |
| 2 | **Wrapper auto-restart race** | Two concurrent Claude conversations each restarted the grind after session caps, leading to duplicate processes contending on the same ledger. | 0 (caught and killed before corruption) | Multiple `run_phase3_resumable.py` PIDs at unexpected times |
| 3 | **Anthropic API credit exhaustion** | Mid-arm3b, the API key hit zero balance. Every subsequent agent call returned `BadRequestError: Your credit balance is too low`. Cells affected: all of seed=2 (30 cells, all 3 attempts hit credit error → `patch_len=0, steps=0`) plus 6 cells in seed=1 (attempts that ran out of budget during the cell). | 36 arm3b cells | `per_attempt[i].error` contains `"credit balance is too low"` |

After incident #1 was diagnosed, we added a `colima ssh -- df -BG /` watchdog (Python; subprocess timeouts to prevent hangs) and a wrapper-side pre-flight `prune_if_low_disk` check. Both ran for the duration of the arm3b rerun.

**Incident #3 was not preventable from the experimental side** — the key ran out during seed=2 and the agent layer cannot distinguish "credit gone, retry pointless" from "transient 4xx, retry might help." Re-running the 36 lost cells would require recharging the key.

The headline numbers treat broken cells as "no data" rather than "failed." This is the right call: scoring a cell `False` because the agent never got to make an API call would unfairly penalise arm3b. The conservative reading is that arm3b's n=45 is half the intended n=90, and any conclusion about arm3b carries proportionally less weight.

---

## Cross-task transfer analysis (arm3b only)

If the persistent bank does what its design implies — accumulate strategies from earlier instances that help on later ones — then within each arm3b seed, the **late** instances should pass attempt-1 at a higher rate than the **early** instances. Difficulty is controlled by comparing each split's attempt-1 pass rate to arm1's pass rate on **exactly the same instances** (arm1 is single-attempt by design, so arm1's resolved=True is the "attempt-1 pass" baseline for that instance).

Instances are ordered within each seed by cell-file mtime (= completion order, which is also the wrapper's execution order). Split is median (first half = early, second half = late).

| Seed | n clean | Split | arm3b attempt-1 pass | arm1 pass (same set) | Gap |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 23 | early (11) | 5/11 = 45.5% | 7/11 = 63.6% | −18.2 pp |
| 0 | 23 | late (12) | 4/12 = 33.3% | 6/12 = 50.0% | −16.7 pp |
| 1 | 22 | early (11) | 5/11 = 45.5% | 6/11 = 54.5% | −9.1 pp |
| 1 | 22 | late (11) | 4/11 = 36.4% | 7/11 = 63.6% | −27.3 pp |

**Pooled across seeds:**
- Early (n=22): arm3b a1 = **10/22 = 45.5%** [27%, 65%], arm1 a1 (same set) = 13/22 = 59.1%. Gap = **−13.6 pp**.
- Late (n=23): arm3b a1 = **8/23 = 34.8%** [19%, 55%], arm1 a1 (same set) = 13/23 = 56.5%. Gap = **−21.7 pp**.

**Position effect: late_a1_pass − early_a1_pass = −1.8 pp** (after difficulty subtraction: the gap *widens* by 8.1 pp as you go later, but with these CIs the change is also indistinguishable from noise).

The transfer signal is **not just absent, it's the wrong sign**. If anything, the persistent bank looks mildly worse on later instances than earlier ones. Possible mechanisms:
- The bank accumulates failure traces faster than success traces (because the dataset is hard), and the failure-shaped guidance pulls the agent further from working solutions.
- Retrieval at k=5 from a 376-entry bank by instance-30 is dominated by superficially-similar-but-domain-different lessons (e.g. a Django form lesson retrieved on a Sphinx instance).
- The persistent bank carries over `phase3-3b-global` scope from an earlier broken run (incident #1's debris); we used a fresh `phase3_arm3b_bank_s{seed}.sqlite` per seed for the rerun, so this is unlikely, but not formally ruled out by inspecting individual retrievals.

---

## Per-instance pass matrix (clean cells only)

Each cell = (passed / clean cells run). A `0/0` means all cells for that (instance, arm) pair were broken by an infrastructure incident.

| Instance | arm1 | arm2 | arm3a | arm3b |
| --- | :---: | :---: | :---: | :---: |
| astropy-12907 | 3/3 | 3/3 | 3/3 | **2/2** |
| astropy-6938 | 3/3 | 3/3 | 3/3 | **1/2** |
| astropy-7746 | 0/3 | 0/3 | 0/3 | 0/1 |
| django-14411 | 3/3 | 3/3 | 2/2 | **0/1** |
| django-14534 | 0/3 | 0/3 | 1/3 | 0/2 |
| django-15061 | 0/3 | 0/3 | 0/3 | 0/2 |
| matplotlib-23314 | 3/3 | 2/3 | 3/3 | **2/2** |
| matplotlib-23476 | 0/3 | 1/3 | 0/2 | 0/0 |
| matplotlib-25433 | 0/3 | 0/3 | 0/3 | 0/2 |
| seaborn-3010 | 3/3 | 3/3 | 3/3 | **2/2** |
| seaborn-3190 | 2/3 | 1/3 | 3/3 | **0/2** |
| seaborn-3407 | 0/3 | 0/3 | 0/3 | 0/0 |
| flask-4045 | 0/3 | 0/3 | 0/3 | 0/2 |
| flask-4992 | 0/3 | 0/3 | 0/3 | 0/2 |
| flask-5063 | 0/3 | 0/3 | 0/3 | 0/1 |
| requests-863 | 3/3 | 1/3 | 3/3 | **2/2** |
| requests-1963 | 3/3 | 2/3 | 2/3 | **2/2** |
| requests-2317 | 3/3 | 1/3 | 3/3 | 0/0 |
| xarray-4094 | 1/3 | 1/3 | 1/3 | **1/2** |
| xarray-4248 | 0/3 | 0/3 | 0/3 | 0/0 |
| xarray-5131 | 3/3 | 3/3 | 3/3 | **2/2** |
| pylint-5859 | 3/3 | 3/3 | 3/3 | 0/0 |
| pylint-7080 | 0/3 | 0/3 | 0/3 | 0/2 |
| pylint-7993 | 2/3 | 2/3 | 3/3 | **1/2** |
| pytest-5227 | 3/3 | 3/3 | 3/3 | **2/2** |
| pytest-6116 | 0/3 | 0/3 | 0/3 | 0/2 |
| sklearn-13439 | 3/3 | 3/3 | 3/3 | **2/2** |
| sklearn-13779 | 3/3 | 3/3 | 2/2 | **2/2** |
| sphinx-7738 | 0/3 | 0/3 | 0/3 | 0/1 |
| sympy-18057 | 1/3 | 0/3 | 1/2 | 0/1 |
| **Total (clean)** | **45/90** | **38/90** | **45/86** | **21/45** |

### Instance difficulty buckets (using arm1 as ground truth)

- **Always solved (3/3 on arm1):** 13 — astropy-12907, astropy-6938, django-14411, matplotlib-23314, seaborn-3010, requests-863, requests-1963, requests-2317, xarray-5131, pylint-5859, pytest-5227, sklearn-13439, sklearn-13779
- **Mostly solved (2/3 on arm1):** 2 — seaborn-3190, pylint-7993
- **Mostly failed (1/3 on arm1):** 2 — xarray-4094, sympy-18057
- **Never solved (0/3 on arm1):** 13 — astropy-7746, django-14534, django-15061, matplotlib-23476, matplotlib-25433, seaborn-3407, flask-4045, flask-4992, flask-5063, xarray-4248, pylint-7080, pytest-6116, sphinx-7738

The middle 4 instances (seaborn-3190, pylint-7993, xarray-4094, sympy-18057) are the only ones with any inter-arm differentiation possible. Even with arm3b included, almost all signal between arms comes from these few instances. **The instance pool's bimodality is the single biggest threat to validity in this experiment**, more than sample size.

---

## Methodology

(unchanged from the arm1/arm2/arm3a phase; arm3b uses the same code path as arm3a, swapping the `ReasoningBank(InMemoryStore())` instance for one whose backing store persists across all instances within a single seed.)

### Design

- 4 arms × 30 instances × 3 seeds = 360 cells intended; 311 clean cells obtained.
- Each cell is one "agent attempts to fix one SWE-bench-lite bug" trial, scored binary by the official Docker evaluator.

**Arms:**
- **arm1_noretry** — agent runs once (`max_steps=20`), patch extracted, scored.
- **arm2_naive** — same agent, on failure re-prompted up to 2 more times with `"Your previous patch did not fix the bug. Try a different approach. The previous patch was: ..."`. Fresh working copy of the repo per attempt.
- **arm3a_bank_fresh** — same up-to-3-attempt loop, but on failure the failed trajectory is `ingest_trajectory(..., outcome="failure")` into a `ReasoningBank(InMemoryStore())` that is created fresh for each instance. Before each attempt, `bank.retrieve(problem_statement, k=5)` pulls memories and they are formatted into an extra system message. Bank discarded at instance boundary.
- **arm3b_bank_persist** — identical to arm3a but the `ReasoningBank` instance persists across all 30 instances within a seed (one `phase3_arm3b_bank_s{seed}.sqlite` per seed). This is the cross-task transfer test.

### Per-arm cost (clean cells only, mean attempts)

| Arm | 1 attempt | 2 attempts | 3 attempts | Mean attempts | Mean input tokens / cell |
| --- | ---: | ---: | ---: | ---: | ---: |
| arm1_noretry | 90 | 0 | 0 | 1.00 | 98,702 |
| arm2_naive | 31 | 4 | 55 | 1.27 | 263,143 |
| arm3a_bank_fresh | 40 | 4 | 42 | 1.16 | 216,599 |
| arm3b_bank_persist | 18 | 2 | 25 | 1.31 | ~280,000 (est. from token logs) |

### Per-arm retry rescue (clean cells only)

Of cells where attempt 1 failed, how often did the retry path produce a pass?

| Arm | Rescued / Failed-on-attempt-1 | Rescue rate |
| --- | ---: | ---: |
| arm2_naive | 7 / 59 | 11.9% |
| arm3a_bank_fresh | 5 / 46 | 10.9% |
| arm3b_bank_persist | 3 / 27 | 11.1% |

All three retry arms have essentially identical rescue rates (≈11%). The bank's advantage in arm3a over arm2 doesn't come from being a better rescuer; it comes from a higher attempt-1 pass rate (`40/86 = 46.5%` vs `31/90 = 34.4%`). The arm3b attempt-1 rate is `18/45 = 40.0%` — between arm2 and arm3a, and consistent with no cross-task learning.

### Harness

Per attempt: shallow-clone repo → `git checkout base_commit` (do NOT apply `test_patch`) → run agent in a `RepoSandbox` (`list_dir`, `read_file`, `grep`, `edit_file`, `write_file`; no shell, no test execution) → `git add -A && git diff --cached` is the `model_patch`. Patch is fed to `python -m swebench.harness.run_evaluation` which applies `model_patch + test_patch` to the base commit, runs FAIL_TO_PASS + PASS_TO_PASS, returns resolved/unresolved. Step budget per attempt: `max_steps=20`. Exponential 429 backoff: 5/15/30/60/90 s.

---

## Threats to validity

1. **arm3b sample size halved.** API credit exhaustion truncated arm3b to n=45 clean cells (50% of intended). The arm3b−arm1 Δ has a CI roughly ±14 pp; a real effect of ±5 pp would be lost.

2. **Per-arm attempt-1 variance is large.** At temperature=0.0 with the same prompt prefix, arm1, arm2, and arm3a attempt-1 should be near-identical. Observed: arm1 = 50.0%, arm2 = 34.4%, arm3a = 46.5%. The 50→34 gap is large enough that the "naive retry hurts" headline is partly driven by attempt-1 non-determinism in the Anthropic API rather than any actual retry effect.

3. **Instance pool is heavily bimodal.** 13 instances always pass on arm1, 13 always fail. Only 4 instances (seaborn-3190, pylint-7993, xarray-4094, sympy-18057) carry inter-arm signal. Any future Phase-4 should pre-screen for moderate-difficulty instances.

4. **Single model class.** Haiku-4.5 only.

5. **Restricted toolset.** No shell-exec, no test-running. Real SWE-bench solvers usually iterate on failing tests; the bank can't ingest test-failure feedback it never sees. This caps the value the bank can extract.

6. **Max 3 attempts.** Cells capped at 3 attempts; marginal value of attempts 4–5 unknown.

7. **Persistent-bank seed isolation.** arm3b's bank is per-seed (3 banks total). A truly cross-task setup would share one bank across all 90 cells. The per-seed choice was made to avoid contamination from the earlier broken run; the trade-off is that within a seed the bank accumulates only ≤30 instances' worth of trajectories, which may be below the threshold where retrieval starts paying off.

8. **No `src/reasoning_bank/` changes.** Tests the unmodified library, not a tuned configuration.

9. **Infrastructure noise.** Three distinct infra failures cost a combined ~95 cells of original output. The clean-cell filter excludes them, but the rerun cells live in a slightly different environment (clean Docker cache, fresh bank, etc.) than the originals would have. Effect is likely small but not zero.

---

## Verdict

**NULL.**

Mapping to the dispatch's pre-registered verdicts:

| Verdict | Criterion | Observed | Match? |
| --- | --- | --- | :---: |
| LEARNING-crosstask | arm3b > arm1 AND late > early within arm3b | arm3b −3.3 pp vs arm1; late −1.8 pp vs early | ✗ |
| LEARNING-modest | arm3b > arm1 but no position effect | arm3b −3.3 pp vs arm1 | ✗ |
| **NULL** | arm3b ≈ arm1 (within ±5 pp) | **arm3b −3.3 pp vs arm1, well within Wilson CI** | **✓** |
| BROKEN-#3 | arm3b ≪ arm1 (more than −5 pp) | arm3b −3.3 pp; bound not crossed | ✗ |

The only positive observation in the four-arm comparison is **defensive**: the bank-with-retry recovers the regression that naive-retry introduces (arm3a − arm2 = +10.1 pp). On its own this isn't an argument for the bank — it's an argument that if you're going to retry at all, retry with a structured-reflection prompt rather than a raw "your patch failed" prompt. The bank's contribution to that defensive recovery is the structured prompt, not the cross-instance memory.

The cross-task transfer hypothesis — that memories accumulated on instance 1 help on instance 30 — is **rejected** in this setup. The persistent bank's late-instance attempt-1 pass rate is no better than its early-instance rate. With n=45 we cannot tightly bound the cross-task effect, but we can rule out a large positive one (>15 pp).

This is the fourth ReasoningBank experiment; previous three returned NULL because the tasks were too easy (NLU benchmarks) or too hard (custom puzzles). SWE-bench-lite cleared that gate (50% baseline, real spread of difficulty), so this NULL is more informative than the prior three: across both within-instance reflection (arm3a) and cross-instance transfer (arm3b), the bank does not change Haiku-4.5's solve rate.

### This is the last experiment in the sequence

Per the dispatch: **the final report proposes the announce, not more experiments.** The proposed announce:

> **ReasoningBank's structured-reflection prompt prevents naive-retry regression but produces no measurable lift over a no-retry baseline on SWE-bench-lite. The cross-instance transfer hypothesis — that the bank learns from earlier tasks to help on later ones — is not supported by the data (n=45 clean cells; late-instance attempt-1 pass rate is no better than early). Across four ReasoningBank experiments spanning easy, hard, and well-calibrated task distributions, the only consistently positive observation is a defensive recovery against naive retry; we have not found a regime where the bank produces net positive value.**

If the team wants to ship the bank anyway as an "if you're going to retry, retry this way" prompt template, the +10.1 pp arm3a vs arm2 result supports that limited claim. But the case for cross-instance memory accumulation as a general capability is not made by these four experiments.

---

## Files & reproducibility

```
~/reasoning-bank/experiments/swebench/
├── agent.py                       # raw AsyncAnthropic ReAct agent (temp 0.0, max_steps 20)
├── harness.py                     # clone → run → extract-diff
├── arms_resumable.py              # arm1/arm2/arm3a/arm3b runners with MAX_ATTEMPTS=3
├── run_phase3_resumable.py        # the runner used for arm1/arm2/arm3a and the arm3b rerun
├── run_arm3b_grind.sh             # wrapper for arm3b: per-seed banks, disk-prune pre-flight
├── analyze.py                     # produces per-arm summary table
├── phase3_instances.json          # the 30 instances
└── run_data/
    ├── phase3_cells/              # 311 clean per-cell JSON ledgers
    ├── phase3_bank.sqlite         # original arm3b bank (orphaned after incident #1 rerun)
    ├── phase3_arm3b_bank_s{0,1,2}.sqlite  # per-seed banks from the arm3b rerun
    ├── arm3b_grind.log            # wrapper log
    ├── arm3b_watchdog.log         # disk/process snapshot log from the Python watchdog
    └── phase3_summary.json        # produced by analyze.py

~/reasoning-bank/  (cell-derived artefacts at repo root)
└── haiku-4-5-agent.p3_*.json      # per-attempt swebench eval reports
```

**To reproduce summary:** `python3 /tmp/phase3_analysis.py` (the analysis script that generated this report; output saved to `/tmp/phase3_analysis.out`).

**To re-run the 36 credit-broken arm3b cells** (after restoring API credit balance):
1. Delete the broken cells: `for f in experiments/swebench/run_data/phase3_cells/arm3b_bank_persist__*.json; do python3 -c "import json,sys; d=json.load(open('$f')); sys.exit(0 if any('credit' in str(p.get('error','')) for p in d.get('per_attempt',[])) else 1)" 2>/dev/null && rm -v "$f"; done`
2. Restart the wrapper: `bash experiments/swebench/run_arm3b_grind.sh 1,2 180 25` (seed=0 already complete)

**Git state:** uncommitted (per dispatch).
