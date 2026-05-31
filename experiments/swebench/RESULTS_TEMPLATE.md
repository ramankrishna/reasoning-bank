# SWE-bench-lite × ReasoningBank — experiment

**Status:** {VERDICT}
**Date:** 2026-05-28
**Model:** Anthropic Haiku 4.5 (`claude-haiku-4-5`), temperature 0.0
**Dataset:** SWE-bench-lite test split (300 instances)
**Repo under test:** `reasoning-bank` (no `fleet` import, raw `AsyncAnthropic`)

## Why this experiment

After three NULL results on simpler agentic benchmarks (the bank didn't change
outcomes because the underlying tasks were too easy to fail or too easy to
solve), SWE-bench-lite was chosen because it is hard enough that even Sonnet-class
models score under 50%. Haiku should fail often — giving the bank-retry arm
something to actually learn from.

## Method

### Harness

- Per-instance: clone the repo at `base_commit` to a local working dir.
- Run a raw `AsyncAnthropic` ReAct agent (`agent.py`) with tools
  `list_dir / read_file / grep / edit_file / write_file`. NO shell-exec or
  test-running — the agent must reason about correctness from the issue text
  and the code alone.
- After the agent stops, take `git diff` of the working dir as the
  `model_patch`.
- Hand the prediction to the official `swebench.harness.run_evaluation`
  Docker-based evaluator, which applies `test_patch + model_patch` to the
  base commit, runs the FAIL_TO_PASS and PASS_TO_PASS tests, and resolves
  pass/fail.

### Constraints (enforced)

- `from importlib.util import find_spec; assert find_spec('fleet') is None` ✓
- `src/reasoning_bank/` is unmodified ✓
- Docker provided by colima (Docker Desktop install fails on macOS sudo-cli-plugins step)
- Concurrency = 1 (org rate limit: 50,000 input tokens/min)

### Phase 1d — gold-patch sanity check

3 instances (`astropy-12907`, `django-14534`, `flask-4045`) had their gold
patches submitted to the swebench evaluator. All 3 resolved ✓. This confirms
the Docker harness, swebench evaluator, image build, and test runner all
work end-to-end.

### Phase 2 — fail-fast probe (BINDING)

10 instances across 10 different repos (one per repo, smallest patch first):
`astropy-12907`, `django-14534`, `matplotlib-23314`, `seaborn-3010`,
`flask-4045`, `requests-863`, `xarray-4094`, `pylint-7080`, `pytest-6116`,
`scikit-learn-13439`.

Decision rule (set up-front, binding):
- solve rate ≥ 80% → STOP, NULL-#2 (ceiling)
- solve rate = 0% → STOP, NULL-#1 (floor)
- 10–70% → proceed to Phase 3

## Results

### Phase 2

{PHASE_2_TABLE}

**Solve rate:** {PHASE_2_RATE}/10
**Verdict:** {PHASE_2_VERDICT}

### Phase 3 (if reached)

{PHASE_3_TABLES}

## Threats to validity

- Haiku 4.5 is a small model. Larger models would likely have a different
  baseline and thus a different bank-effect curve. We are not claiming this
  generalises.
- Agent tools are read/edit-only — the agent cannot run tests or use a
  debugger. This makes solve rate floor lower than a real-world coding
  agent. Bank-retry effects might be larger or smaller with a richer tool
  set.
- We use `swebench.harness.run_evaluation` which downloads/builds per-instance
  Docker images. Image-pulling timing is not part of the model latency.
- `temperature=0.0` does not give bit-for-bit reproducibility on the
  Anthropic API; the 3 seeds are seed-1, 2, 3 of repeated runs.
- Bank-retry shares the SAME model with no retrieval-vs-bank ablation. Some
  effect could come from "retry with more context in system prompt" rather
  than from the lessons themselves.

## Files

- `experiments/swebench/agent.py` — raw Anthropic agent
- `experiments/swebench/harness.py` — per-instance Docker orchestration
- `experiments/swebench/arms.py` — Phase-3 arm logic
- `experiments/swebench/run_probe.py` — Phase-2 runner
- `experiments/swebench/run_phase3.py` — Phase-3 runner
- `experiments/swebench/run_data/probe_<id>/` — Phase-2 records
- `experiments/swebench/run_data/phase3_<id>/` — Phase-3 records
