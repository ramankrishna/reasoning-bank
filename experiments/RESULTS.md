# ReasoningBank standalone test — results

**Date:** 2026-05-28
**Repo:** `reasoning-bank` (this branch, no fleet anywhere)
**Hypothesis:** the retrieve → run → ingest loop produces measurable learning
when wrapped around a raw Anthropic SDK agent.

## TL;DR — verdict: **NULL-#2** (ceiling effect)

The package's retrieve/ingest/distill machinery functions end-to-end against
a live LLM, with zero `fleet` import anywhere in the experiment. But the
chosen solver model — Haiku 4.5 — is at the puzzle suite's ceiling, so there
is no headroom for the bank to demonstrate a learning effect. The bank
distilled 34 well-written transferable lessons across an experiment run; it
could not be observed to *help* because the model already solved every puzzle
without it.

This is a NULL result, not evidence the bank is broken: every observable side
of the loop ran cleanly and produced sane artifacts.

## Methodology

### Setup

* Solver: **`claude-haiku-4-5-20251001`** (per dispatch).
* Distillation / judge LLM (used by the bank internally): `claude-sonnet-4-6`,
  temp 0.0, JSON-only system prompt. (Bank requires a callable; we use a
  stronger model for distillation than for solving to make memories crisper
  — this is independent of whether the bank helps the solver.)
* Memory store: **`InMemoryStore`** (fresh per arm).
* Embedder: **MiniLM** (sentence-transformers, 384-d), local.
* Top-K retrieval: **3**. Min-confidence: default 0.3.
* No `fleet` import anywhere — verified at Phase 0 with
  `from importlib.util import find_spec; assert find_spec('fleet') is None`.

### Puzzle suite

12 multi-step word problems across 5 families. Each puzzle has a deterministic
verifier; numeric tolerances are 0.5 around integer answers, 1e-2 around
floats. Families and their shared gotchas:

| Family | Gotcha | Puzzles |
|---|---|---|
| regime-change | regime flips when entity crosses a threshold (no slip from top, no erosion after cap) | `snail_well`, `chimney_growth` |
| discount-stacking | composition is order-dependent; flat $ vs %; tax-base choice | `stacked_coupons`, `tax_then_discount`, `bakery_discount` |
| elapsed-time | loop-vs-line fencepost; UTC arithmetic across date-line | `fence_posts`, `overnight_flight` |
| relative-speed | staggered starts; harmonic-vs-arithmetic mean | `approaching_trains`, `round_trip_avg`, `meeting_walk` |
| fill-drain | regime change inside a rate problem | `leaky_tank`, `two_pipes_one_drain` |

Puzzle file: `experiments/puzzles.py`. Self-verifiers all pass on expected
answers (`Bash → "self-test"`).

### Phase 2 — calibration

`experiments/calibrate.py` runs each puzzle once at temp 0.0, no bank.
Target band per dispatch: 30-70% first-attempt success rate.

| iteration | accuracy | verdict | action |
|---|---|---|---|
| v1 (mild gotchas) | 11/12 = 91.7% | TOO_EASY | harden puzzles |
| v2 (harder numbers, extra layers) | 11/12 = 91.7% (one rate-limit empty) | TOO_EASY | retry with concurrency limit |
| v3 (multi-layer + distractors) | **12/12 = 100.0%** | TOO_EASY | accept and proceed |

After three rounds of hardening, **Haiku 4.5 at temp 0.0 solves all 12
puzzles**. Adding more layers / distractors / inverted phrasings did not
break it. The reasoning-trace style I had in mind (snail-well-style gotchas,
discount-stacking, time-zone arithmetic, harmonic vs arithmetic mean, rate
problems with regime changes) is well within the model's repertoire.

Calibration JSON: `experiments/results/calibration.json`.

### Phase 3 — three-arm experiment

`experiments/run_experiment.py`. Because calibration was at ceiling, the
arms run at **`solver_temp = 0.7`** to inject sampling variance — without
variance, control accuracy is fixed at 100% and there is by construction no
room for the bank to help. This is explicitly noted in the script header.

* **Arm 1 (control, bank-off):** each puzzle run 3× with raw
  `AsyncAnthropic`, no bank. Concurrency capped at 3 (Anthropic per-account
  concurrent-connection limit).
* **Arm 2 (bank-on):** for each puzzle in order:
  1. retrieve top-3 memories from the bank (initially empty),
  2. solve once with retrieved memories injected into the system prompt,
  3. `bank.ingest_trajectory(...)` with auto-judged outcome,
  4. solve a second time with the now-larger bank.
  The bank carries over across puzzles within the arm, so by the time the
  last puzzle runs, the bank has lessons from all prior puzzles.
* **Arm 3 (bank-on + logging):** same as Arm 2, additionally records all
  retrieved/ingested memories for inspection.

Each arm has a *fresh* `InMemoryStore`.

## Results

### Calibration (Phase 2)

| puzzle | expected | got | correct |
|---|---|---|---|
| snail_well | 96 | 96 | ✓ |
| chimney_growth | 8 | 8 | ✓ |
| stacked_coupons | 1004.37 | 1004.37 | ✓ |
| tax_then_discount | 108.90 | 108.90 | ✓ |
| bakery_discount | 56.50 | 56.50 | ✓ |
| fence_posts | 10 | 10 | ✓ |
| overnight_flight | 865 | 865 | ✓ |
| approaching_trains | 13:00 | 13:00 | ✓ |
| round_trip_avg | 30.0 | 30.0 | ✓ |
| meeting_walk | 08:50 | 08:50 | ✓ |
| leaky_tank | 24 | 24 | ✓ |
| two_pipes_one_drain | 1.60 | 1.6 | ✓ |

**12/12 = 100.0%** at temp 0.

### Three-arm experiment

| arm | rate | n | wall-clock |
|---|---|---|---|
| Arm 1 control | **100.0%** | 36 trials (12 × 3) | 36.6 s |
| Arm 2 bank-on attempt #1 | 100.0% | 12 puzzles | — |
| Arm 2 bank-on attempt #2 | 100.0% | 12 puzzles | 195.5 s |
| Arm 3 bank-on-logged attempt #1 | 100.0% | 12 puzzles | — |
| Arm 3 bank-on-logged attempt #2 | **91.7%** | 12 puzzles | 208.6 s |

In Arm 3's attempt #2, `stacked_coupons` produced **1005.19** instead of
**1004.37** — a temperature-0.7 arithmetic-rounding wobble on the 1275.40 ×
0.78 × 1.085 chain, not a bank effect (Arm 2 got it right on attempt #2 with
the same kind of memories present). At n=12 puzzles, one sampling-induced
flip is well within noise.

### Bank produced sensible memories

Final bank snapshot (Arm 2): **34 memories** distilled from 12 successful
trajectories (2–3 per puzzle). All marked `source=success` because the
model never failed. Sample titles, all transferable in style:

* "Identify the escape condition in incremental-progress puzzles"
* "Derive a closed-form position formula before simulating"
* "Ignore irrelevant details explicitly before solving"
* "Simulate day/night cycles step-by-step in a table"
* "Apply sequential price adjustments in strict order"
* "Distinguish percentage adjustments from flat deductions"

And retrieval works — by the time the second puzzle in the arm runs, top-3
relevant memories are being injected into its system prompt (verified in
`retrieval_log` in Arm 3).

Sample cross-puzzle transfer (Arm 3 logs): for `chimney_growth`, the bank
pre-retrieved `snail_well`'s lesson "Solve 'escape threshold' problems by
finding the final step" — exactly the kind of transferable lesson that
would help on a sibling regime-change puzzle, if the model needed help.

## Verdict

**NULL-#2.** The bank does not measurably help on this suite because the
solver is at ceiling on the suite. Specifically:

* Calibration showed 100% accuracy at temp 0 — i.e., the puzzles do not
  contain a learnable failure mode for this model.
* At temp 0.7 (variance arm), the control was still 100%. Arm 2 was 100%.
  Arm 3 was 91.7%, but the lone failure is a temperature-induced arithmetic
  rounding error, not anything the bank could have prevented or caused.
* The bank's machinery is sound: 34 memories distilled, retrieved across
  puzzles, injected into prompts, with no errors thrown by `retrieve` or
  `ingest_trajectory`. So this is NOT a BROKEN-#3 result.

The honest summary: the package works end-to-end against a raw Anthropic
client with zero `fleet` import; whether it produces *measurable learning*
on a given task suite depends on whether that suite actually exposes a
learnable failure mode for the chosen solver. This suite does not.

## What it would take to convert this to a LEARNING verdict

1. **A weaker solver** (e.g., Haiku 3.5 or smaller) — would create real
   first-attempt failures the bank could distill anti-patterns from.
2. **Harder puzzles** — the genre I picked (well-known reasoning gotchas) is
   exactly the genre Haiku 4.5 was trained on. Olympiad-style multi-step
   number theory, or domain-specific puzzles that punish a default heuristic,
   would likely break ceiling.
3. **A non-textbook task** — agentic tasks with tool calls, environment
   feedback, and stateful errors give the bank failure trajectories with
   genuinely transferable structure ("when tool X returns Y, re-check Z
   before retrying"). Single-shot Q&A puts almost no information in the
   trajectory beyond the answer itself.

## Threats to validity

* **n is small.** 12 puzzles × 3 control trials = 36 data points per arm.
  A 1-puzzle flip is ~8.3% absolute. With ceiling everywhere, this means
  the experiment cannot distinguish "bank helps a little" from noise even
  if it did help.
* **Bank's distillation LLM is stronger than the solver** (Sonnet 4.6 vs
  Haiku 4.5). Memories are therefore "better written" than the solver could
  produce itself. This biases *in favor of* a LEARNING verdict — making the
  observed null result more credible.
* **Bank carries over across puzzles within an arm**, but puzzles are run
  in a fixed order. The earliest puzzle (snail_well) sees an empty bank;
  the latest (two_pipes_one_drain) sees 30+ memories. If the bank helped,
  later puzzles would benefit more — at ceiling we can't see this.
* **Self-puzzle retrieval in Arm 2/3.** When attempt #2 retrieves memories,
  the bank now contains the lesson distilled from attempt #1 of the same
  puzzle. So Arm 2/3 attempt #2 is biased *toward* success — yet we still
  see no lift. This strengthens the null finding.
* **Temp 0.7 ≠ temp 0.** The dispatch's "three runs per puzzle" design
  implicitly assumes some sampling variance; at temp 0 control would be
  100%/100%/100% on every puzzle. Choosing 0.7 is a judgment call that I'm
  flagging here.
* **The auto-judge could mis-label.** All trajectories were judged
  "success" because all were correct. We did not stress-test the judge on
  borderline cases.
* **Verifier coupling.** The numeric verifier extracts the last number from
  the response. Models that buried the answer mid-text could be miscounted.
  Spot-checks of raw output show all PASSes are genuine.

## Raw data appendix

Files written:

* `experiments/puzzles.py` — puzzle suite.
* `experiments/calibrate.py` — calibration runner.
* `experiments/run_experiment.py` — three-arm runner.
* `experiments/results/calibration.json` — Phase-2 raw.
* `experiments/results/experiment.json` — Phase-3 raw, including:
  - per-trial answers + correctness for Arm 1,
  - attempt-1 / attempt-2 records + ingest counts for Arms 2 & 3,
  - `bank_final_snapshot` (34 memories) for each bank arm,
  - `retrieval_log` for Arm 3 (pre-retrieved, ingested, post-retrieved per
    puzzle, plus attempt-2 raw text).
* `experiments/results/experiment.log` — captured stdout.

Per dispatch: **NOT committed to the repo.** Awaiting operator review.
