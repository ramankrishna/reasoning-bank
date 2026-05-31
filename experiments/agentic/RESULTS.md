# Agentic ReasoningBank experiment — results

**Verdict: NULL-#2 — no meaningful learning signal detected at this scale.**
The bank distills the *right* lessons, but the agent (Haiku 4.5) is good
enough at the task suite that the lessons rarely have somewhere to bind.
Success rate is at ceiling and step counts move within noise.

---

## 1. Methodology

### 1.1 Code surface (no fleet)
- `experiments/agentic/tools.py` — `Sandbox` class. Filesystem + `run_shell` +
  `run_python`, each locked to a per-task tempdir. Hard 10 s timeout, no
  network, denylist for destructive patterns. `run_python` uses
  `subprocess.run([sys.executable, "-c", code])`, so **no state persists
  across calls** — this is the load-bearing G2 hazard.
- `experiments/agentic/tasks.py` — 12 tasks across 5 gotcha families
  (G1–G5). Each task: `setup(root)`, `verify(answer, root, traj)`, and a
  shared `detect_gotcha_errors(traj, answer)` that scans the trajectory for
  the canonical mistake signature of each family.
- `experiments/agentic/agent.py` — raw-Anthropic `AsyncAnthropic` ReAct
  loop. Tools registered via the `tools=` spec. Temp 0.0. Optional
  `system_block` parameter — the bank injects retrieved memories there.
- `experiments/agentic/runner.py` — per-task lifecycle: fresh
  `tempfile.mkdtemp()`, `task.setup`, agent run, `task.verify`,
  `detect_gotcha_errors`, then `shutil.rmtree` regardless of outcome.
  Identical starting state for every arm.
- `experiments/agentic/experiment.py` — three-arm driver. Independent
  `InMemoryStore` per arm.

`reasoning_bank` and `anthropic` are the only deps. `assert
importlib.util.find_spec('fleet') is None` confirmed at session start.

### 1.2 Gotcha families and detection signatures
| Family | Description | Detector signature |
|---|---|---|
| G1 | Discovery-before-action: filenames not given | `read_file` returned `"file not found"` (guessed) |
| G2 | State doesn't persist between `run_python` calls | any `run_python` output contains `NameError` |
| G3 | `run_python` captures stdout, not values | code lacks `print(` AND empty stdout |
| G4 | Non-zero exit ≠ success even if stdout looks good | shell exited non-zero AND final answer affirms success |
| G5 | File may not exist before append/edit | `read_file` showed log.txt missing |

Detector and verifier were unit-tested on 10 synthetic trajectories and
on 12 hand-constructed correct trajectories. All passed.

### 1.3 Sandbox safety (pre-flight)
All 10 sandbox isolation checks passed before any LLM call:
- path-escape via `../../../etc/passwd` → blocked
- `sudo rm -rf /` → blocked (denylist)
- `curl …` → blocked (network denylist)
- write outside sandbox via `../` → blocked
- 15 s shell sleep → killed at 10 s
- 15 s python sleep → killed at 10 s
- `run_python` showed `NameError` for a variable defined in the previous
  call → G2 hazard is real
- bare expression produced empty stdout → G3 hazard is real
- read of nonexistent file produced `"file not found"` → G1 hazard is real

### 1.4 Three arms
| Arm | What it does |
|---|---|
| Arm 1 — bank OFF | baseline; no system_block, no retrieval, no ingest |
| Arm 2 — bank ON | `bank.retrieve(prompt, k=5)` → format as system_block → run → `bank.ingest_trajectory(traj, prompt, outcome)` |
| Arm 3 — bank ON + retrieval log | identical to Arm 2 + write retrieved titles per task to `arm3_retrieval_log.json` |

- All three arms iterate tasks in the same order T01 → T12.
- Each arm has its own fresh `InMemoryStore`; no cross-arm contamination.
- LLM (agent and bank-internal judge/distill) is
  `claude-haiku-4-5-20251001`, temp 0.0.
- Embedder is the bank's default MiniLM (sentence-transformers,
  384-dim).
- Each (task, arm) gets a fresh sandbox via `tempfile.mkdtemp()` + setup,
  destroyed afterwards.

---

## 2. Phase 3 — calibration

Single bank-off pass over all 12 tasks:

```
n=12  success=12/12  mean_steps=2.00
any_gotcha_hit_rate=0.25  (3/12 tasks tripped at least one gotcha)
hits_by_family: G1=1, G2=2, G3=1, G5=1
```

- Ceiling check: gotchas WERE hit (3/12) — not a ceiling.
- Floor check: success rate = 100% — not a floor.
- Verdict: MARGINAL but neither stop condition fires. We proceed to
  Phase 4 understanding that *success rate is already at ceiling* and
  the bank's potential lift will only be visible in mean steps and
  gotcha-hit rate.

---

## 3. Phase 4 — three-arm results

### 3.1 Summary

| Metric | Arm 1 (off) | Arm 2 (on) | Arm 3 (on+log) |
|---|---|---|---|
| Success rate | 12/12 | 12/12 | 12/12 |
| Mean steps  | 1.92  | 1.92  | 1.83 |
| Any-gotcha-hit rate | 16.7% (2/12) | 25.0% (3/12) | 16.7% (2/12) |
| Repeated-error count | 0 | 1 | 0 |
| Hits by family | G1=1, G2=1, G3=1, G5=1 | G2=2, G3=1, G1=1, G5=1 | G2=1, G1=1, G5=1 |

Elapsed: 215.9 s for all 36 task runs.

### 3.2 Per-task hits (failures *of process*, not of outcome — all tasks succeed)

| Task | Family | Arm 1 hits | Arm 2 hits | Arm 3 hits |
|---|---|---|---|---|
| T01 secret_token | G1 | — | — | — |
| T02 birthday | G1 | — | — | — |
| T03 count_lines | G1 | — | — | — |
| T04 fib_split | G2 | — | G2 | G2 |
| T05 squares_sum | G2 | G2, G3 | G2, G3 | — |
| T06 sha_digest | G3 | — | — | — |
| T07 sum_range | G3 | — | — | — |
| T08 build_status | G4 | — | — | — |
| T09 test_status | G4 | — | — | — |
| T10 grep_miss | G4 | — | — | — |
| T11 append_create | G5 | G1, G5 | G1, G5 | G1, G5 |
| T12 append_existing | G5 | — | — | — |

### 3.3 What the bank actually distilled

Inspection of `arm3_retrieval_log.json` shows the bank captured all five
gotcha families as discrete, well-named lessons:

| Family | Distilled lesson title (representative) |
|---|---|
| G1 | "List directory contents before reading files" |
| G2 | "Rebuild state in isolated tool calls" |
| G2 | "Understand tool execution scope and state isolation" |
| G3 | "Include print() in Python code for tool output capture" |
| G4 | "Prioritize exit codes over stdout messages" |
| G4 | "Recognize that build scripts may report success before detecting errors" |
| G5 | "Handle missing files by creating them when appropriate" |

This is the strongest positive signal in the experiment: **the bank's
distillation correctly named the gotcha mechanisms in its own words.**

---

## 4. Verdict: NULL-#2

**No meaningful learning lift between bank-off and bank-on.**

Why "null" and not "broken":
- The bank ran end-to-end without error.
- It retrieved 0–5 memories per task (only the first task retrieves 0,
  because the bank starts empty).
- It distilled lessons that map exactly onto the gotcha families.
- The retrieval-log inspection shows reasonable top-k relevance (G2
  lessons surface for T05, G4 lessons surface for T09, etc.).

Why the lift is invisible:
1. **Ceiling.** Haiku 4.5 solves 12/12 even without the bank. There is
   no headroom on the success metric.
2. **Sticky-instruction confound on G2.** T04 and T05 *literally tell the
   agent to use two separate `run_python` calls*. The bank's "rebuild
   state in isolated tool calls" memory is retrieved for T05 in both
   bank-on arms, and Arm 3 does combine the calls and skips the gotcha
   — but Arm 2 doesn't. With N=1 per arm per task, this is a coin flip,
   not a trend.
3. **N=2 per family is too few.** The bank can only learn from the
   first task in a family and apply to the second. With only one
   transfer attempt per family, a single coin flip dominates the
   per-family signal.
4. **Determinism is imperfect.** Arm 2 and Arm 3 are identical except
   for whether the retrieval log is written to disk — yet they
   disagree on T05 (Arm 2 hit G2+G3, Arm 3 hit none) and on T09 (Arm 2
   used 1 step, Arm 3 used 1 step, but baseline used 2). Temperature
   0.0 does not guarantee bitwise determinism in the Anthropic API,
   so a 1-task swing between identical arms tells us the experiment's
   per-task variance is roughly equal to its claimed bank effect.

Together these say: at this N, the bank's true effect (if any) is
below the noise floor. We cannot tell whether the bank is helping,
hurting, or neutral from these 36 runs alone.

---

## 5. Threats to validity

- **Sticky-instruction confound** (above). G2 tasks explicitly instruct
  the agent to split calls, which fights the very lesson the bank
  would inject. Should be re-worded ("you may use multiple steps")
  before re-running.
- **N=12 with N=2 per family** is far too small to separate signal
  from API non-determinism.
- **Ceiling on success rate.** With 12/12 baseline, the only
  differentiators left are step count and gotcha-hit count — both
  noisy at this N. Real lift on this suite requires either harder
  tasks or a weaker base model.
- **G5 detection is conservative.** It fires on the first
  read-failure of log.txt, which is in fact a reasonable thing to do
  before appending (most agents read first to know existing content).
  Counting it as a "mistake" overstates the G5 hit rate.
- **Embedder warm-up cost.** MiniLM is loaded lazily on the first
  ingest. Not a correctness issue, but cosmetic noise in the log.
- **`run_python` runs `sys.executable -c`** — that's the host venv,
  not an isolated interpreter. The denylist + `cwd=sandbox` + 10 s
  timeout cover ordinary safety, but a determined attacker inside
  the LLM could still `import os` and read environment variables. We
  did not see this happen.

---

## 6. What would change the verdict

To turn NULL into either LEARNING or BROKEN, one of these is needed:

- **More tasks per family** (e.g. 5 G2 tasks, 5 G3 tasks) — then a
  bank arm that suppresses 4/5 second-round hits is a clear signal.
- **Harder tasks** that Haiku 4.5 fails without help (longer
  trajectories, ambiguous filenames, scripts that look correct).
- **Multiple seeds.** Even without a temperature lever, run each arm
  3× and report mean ± std. A 0.16 → 0.25 jump on a single seed is
  not a finding.
- **Stronger transfer probe.** Run T05 with the bank pre-seeded with
  exactly the "rebuild state" memory and nothing else. If T05 still
  hits G2, the lesson is being ignored — bank effect is null. If it
  doesn't, the bank effect *exists* but is invisible against
  arm-level noise.

---

## 7. Files

```
experiments/agentic/
├── RESULTS.md                  (this file)
├── tools.py                    (sandbox)
├── tasks.py                    (12 tasks + detect_gotcha_errors)
├── test_detection.py           (detector + verifier unit tests)
├── agent.py                    (raw-Anthropic ReAct loop)
├── runner.py                   (per-task lifecycle)
├── calibrate.py                (Phase 3 driver)
├── experiment.py               (Phase 4 three-arm driver)
└── results/
    ├── calibration.json
    ├── calibration_summary.json
    ├── arm1_off.json
    ├── arm2_on.json
    ├── arm3_on_logged.json
    ├── arm3_retrieval_log.json
    └── summary.json
```

Everything reproducible with:

```
uv run python -m experiments.agentic.test_detection
uv run python -m experiments.agentic.calibrate
uv run python -m experiments.agentic.experiment
```

(All three need `ANTHROPIC_API_KEY` set.)

Per dispatch instructions: NOT committed to the repo.
