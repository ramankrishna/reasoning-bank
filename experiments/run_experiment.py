"""Three-arm experiment for ReasoningBank.

Arm 1 (control):  bank-off. Each puzzle run N_TRIALS times. Records pass rate.
Arm 2 (bank-on):  For each puzzle, run once -> ingest trajectory -> run again
                  with retrieved memories injected into the system prompt.
                  Records first-attempt and post-bank rates.
Arm 3 (bank-on +
       logging):  Same as Arm 2, but also logs what the bank retrieved and
                  what it distilled.

Everything uses raw `anthropic.AsyncAnthropic` — no fleet, no agent framework.
Memory store is `InMemoryStore`, isolated per arm.

NOTE ON TEMPERATURE
-------------------
Calibration showed Haiku 4.5 at temp 0.0 is at ceiling (12/12) on the puzzle
suite. To create the *headroom* the experiment needs, we run the arms at
temperature TEMP = 0.7. This makes per-trial outcomes stochastic, so even a
puzzle the model usually solves can fail sometimes, giving the bank actual
trajectories to learn from. Calibration is still single-shot temp=0 (the
reference deterministic baseline).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from anthropic import AsyncAnthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.puzzles import PUZZLES, verify  # noqa: E402

from reasoning_bank import (  # noqa: E402
    InMemoryStore,
    ReasoningBank,
    Turn,
)

MODEL = "claude-haiku-4-5-20251001"
SOLVER_TEMP = 0.7  # injects variance so we have failures to learn from
INDUCTION_TEMP = 0.0  # crisp/distilled memories
INDUCTION_MODEL = "claude-sonnet-4-6"  # stronger model for distillation/judging
MAX_TOKENS_SOLVE = 1024
MAX_TOKENS_INDUCT = 2048
N_TRIALS = 3
N_PUZZLE_SEMAPHORE = 3  # concurrent solver calls per arm
BANK_K = 3
BASE_SOLVER_SYSTEM = (
    "You are a careful reasoning assistant. Think step by step, but finish "
    "your response with a single line beginning with 'ANSWER:' followed by "
    "your final answer (a number or short string only — no units, no "
    "explanation on that line)."
)


OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_PATH = OUT_DIR / "experiment.json"


def _extract_final_answer(text: str) -> str:
    final = ""
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("ANSWER:"):
            final = s[len("ANSWER:") :].strip()
    if final:
        return final
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s:
            return s
    return ""


async def _solve(
    client: AsyncAnthropic,
    question: str,
    system: str,
    temperature: float = SOLVER_TEMP,
) -> str:
    msg = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_SOLVE,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    parts = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def _make_induction_llm(client: AsyncAnthropic):
    """The bank's distill/judge LLM. Lower temp + JSON-safe."""

    async def llm(prompt: str, *, system: str | None = None) -> str:
        msg = await client.messages.create(
            model=INDUCTION_MODEL,
            max_tokens=MAX_TOKENS_INDUCT,
            temperature=INDUCTION_TEMP,
            system=system or "You produce ONLY a JSON response, no prose.",
            messages=[{"role": "user", "content": prompt}],
        )
        parts = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts)

    return llm


# ---------- Arm 1: control ----------


async def arm_control(client: AsyncAnthropic) -> dict:
    sem = asyncio.Semaphore(N_PUZZLE_SEMAPHORE)
    trials: list[dict] = []
    started = time.time()

    async def one(puzzle, trial_idx):  # type: ignore[no-untyped-def]
        async with sem:
            t0 = time.time()
            try:
                raw = await _solve(client, puzzle.question, BASE_SOLVER_SYSTEM)
                err = None
            except Exception as exc:  # noqa: BLE001
                raw = ""
                err = f"{type(exc).__name__}: {exc}"
            answer = _extract_final_answer(raw)
            ok = bool(answer) and verify(puzzle.id, answer)
            return {
                "puzzle_id": puzzle.id,
                "family": puzzle.family,
                "trial": trial_idx,
                "answer": answer,
                "correct": ok,
                "error": err,
                "elapsed_s": round(time.time() - t0, 2),
            }

    coros = [
        one(p, t)
        for p in PUZZLES
        for t in range(N_TRIALS)
    ]
    trials = await asyncio.gather(*coros)
    elapsed = round(time.time() - started, 2)
    return {
        "arm": "control",
        "n_trials_per_puzzle": N_TRIALS,
        "trials": trials,
        "wall_clock_s": elapsed,
    }


# ---------- Arms 2 & 3: bank-on ----------


async def arm_bank(
    client: AsyncAnthropic,
    *,
    log_retrievals: bool,
) -> dict:
    """For each puzzle, in order:
       1) attempt #1 with empty (or partially-populated) bank
       2) ingest the trajectory (success or failure)
       3) attempt #2 with bank retrieval injected into system prompt.

    Bank is shared across puzzles within an arm — by the time later puzzles
    run, the bank contains memories from earlier puzzles. This is the
    cross-puzzle transfer signal the experiment is looking for.

    Bank is *fresh* (empty InMemoryStore) at the start of the arm.
    """

    started = time.time()
    induct_llm = _make_induction_llm(client)
    bank = ReasoningBank(
        llm=induct_llm,
        store=InMemoryStore(),
        scope="experiment",
    )
    trials: list[dict] = []
    retrieval_log: list[dict] = []

    for p in PUZZLES:
        # ----- attempt 1: pre-ingest (bank may have items from earlier puzzles)
        pre_mems = await bank.retrieve(p.question, k=BANK_K)
        pre_system = BASE_SOLVER_SYSTEM
        if pre_mems:
            block = bank.format_as_system_block(pre_mems)
            pre_system = f"{BASE_SOLVER_SYSTEM}\n\n{block}"

        t0 = time.time()
        try:
            raw1 = await _solve(client, p.question, pre_system)
            err1 = None
        except Exception as exc:  # noqa: BLE001
            raw1 = ""
            err1 = f"{type(exc).__name__}: {exc}"
        ans1 = _extract_final_answer(raw1)
        ok1 = bool(ans1) and verify(p.id, ans1)
        elapsed1 = round(time.time() - t0, 2)

        # ----- ingest attempt 1 trajectory
        trajectory1 = [
            Turn("user", p.question),
            Turn("assistant", raw1),
        ]
        try:
            ingested = await bank.ingest_trajectory(
                trajectory1,
                task=p.question,
                outcome="success" if ok1 else "failure",
            )
            ingest_err = None
        except Exception as exc:  # noqa: BLE001
            ingested = []
            ingest_err = f"{type(exc).__name__}: {exc}"

        # ----- attempt 2: post-ingest (bank now contains lessons from attempt 1)
        post_mems = await bank.retrieve(p.question, k=BANK_K)
        post_system = BASE_SOLVER_SYSTEM
        if post_mems:
            block = bank.format_as_system_block(post_mems)
            post_system = f"{BASE_SOLVER_SYSTEM}\n\n{block}"

        t1 = time.time()
        try:
            raw2 = await _solve(client, p.question, post_system)
            err2 = None
        except Exception as exc:  # noqa: BLE001
            raw2 = ""
            err2 = f"{type(exc).__name__}: {exc}"
        ans2 = _extract_final_answer(raw2)
        ok2 = bool(ans2) and verify(p.id, ans2)
        elapsed2 = round(time.time() - t1, 2)

        record = {
            "puzzle_id": p.id,
            "family": p.family,
            "attempt1_answer": ans1,
            "attempt1_correct": ok1,
            "attempt1_error": err1,
            "attempt1_elapsed_s": elapsed1,
            "ingest_error": ingest_err,
            "n_memories_ingested": len(ingested),
            "attempt2_answer": ans2,
            "attempt2_correct": ok2,
            "attempt2_error": err2,
            "attempt2_elapsed_s": elapsed2,
            "n_pre_mems": len(pre_mems),
            "n_post_mems": len(post_mems),
        }
        trials.append(record)
        print(
            f"  [{p.id:>22}] a1={'P' if ok1 else 'F'} "
            f"(ans={ans1!r:>10}) "
            f"-> ingested {len(ingested)} mems "
            f"-> a2={'P' if ok2 else 'F'} "
            f"(ans={ans2!r:>10})  "
            f"pre_k={len(pre_mems)}, post_k={len(post_mems)}"
        )

        if log_retrievals:
            retrieval_log.append({
                "puzzle_id": p.id,
                "pre_retrieved": [
                    {
                        "id": m.id,
                        "title": m.title,
                        "description": m.description,
                        "content": m.content,
                        "source": m.source,
                    }
                    for m in pre_mems
                ],
                "ingested": [
                    {
                        "id": m.id,
                        "title": m.title,
                        "description": m.description,
                        "content": m.content,
                        "source": m.source,
                    }
                    for m in ingested
                ],
                "post_retrieved": [
                    {
                        "id": m.id,
                        "title": m.title,
                        "description": m.description,
                        "content": m.content,
                        "source": m.source,
                    }
                    for m in post_mems
                ],
                "attempt2_raw": raw2,
            })

    elapsed = round(time.time() - started, 2)
    # Final bank snapshot
    bank_snapshot = []
    try:
        for m in await bank.list(scope="experiment"):
            bank_snapshot.append({
                "id": m.id,
                "title": m.title,
                "description": m.description,
                "content": m.content,
                "source": m.source,
                "use_count": m.use_count,
                "confidence": m.confidence,
            })
    except Exception as exc:  # noqa: BLE001
        bank_snapshot = [{"error": str(exc)}]

    result = {
        "arm": "bank-on-logged" if log_retrievals else "bank-on",
        "trials": trials,
        "bank_final_snapshot": bank_snapshot,
        "wall_clock_s": elapsed,
    }
    if log_retrievals:
        result["retrieval_log"] = retrieval_log
    return result


# ---------- summarize ----------


def _summarize_control(arm: dict) -> dict:
    by_puzzle: dict[str, list[int]] = defaultdict(list)
    for t in arm["trials"]:
        by_puzzle[t["puzzle_id"]].append(int(t["correct"]))
    rates = {pid: sum(v) / len(v) for pid, v in by_puzzle.items()}
    overall = sum(int(t["correct"]) for t in arm["trials"]) / len(arm["trials"])
    return {
        "per_puzzle_rate": rates,
        "overall_rate": overall,
        "n_trials": len(arm["trials"]),
    }


def _summarize_bank(arm: dict) -> dict:
    a1 = [int(t["attempt1_correct"]) for t in arm["trials"]]
    a2 = [int(t["attempt2_correct"]) for t in arm["trials"]]
    return {
        "n_puzzles": len(arm["trials"]),
        "attempt1_rate": sum(a1) / len(a1) if a1 else 0,
        "attempt2_rate": sum(a2) / len(a2) if a2 else 0,
        "delta": (sum(a2) - sum(a1)) / len(a1) if a1 else 0,
        "n_total_memories": (
            len(arm.get("bank_final_snapshot", []))
            if isinstance(arm.get("bank_final_snapshot"), list)
            else 0
        ),
    }


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = AsyncAnthropic()

    print("=== ARM 1: control (bank-off) ===")
    arm1 = await arm_control(client)
    s1 = _summarize_control(arm1)
    print(f"control overall: {s1['overall_rate']:.1%}  "
          f"({s1['n_trials']} trials)")
    print()

    print("=== ARM 2: bank-on ===")
    arm2 = await arm_bank(client, log_retrievals=False)
    s2 = _summarize_bank(arm2)
    print(
        f"bank-on  attempt1={s2['attempt1_rate']:.1%}  "
        f"attempt2={s2['attempt2_rate']:.1%}  delta={s2['delta']:+.1%}"
    )
    print()

    print("=== ARM 3: bank-on + logged ===")
    arm3 = await arm_bank(client, log_retrievals=True)
    s3 = _summarize_bank(arm3)
    print(
        f"bank-on-logged  attempt1={s3['attempt1_rate']:.1%}  "
        f"attempt2={s3['attempt2_rate']:.1%}  delta={s3['delta']:+.1%}"
    )
    print()

    payload = {
        "config": {
            "model": MODEL,
            "induction_model": INDUCTION_MODEL,
            "solver_temp": SOLVER_TEMP,
            "induction_temp": INDUCTION_TEMP,
            "n_trials_control": N_TRIALS,
            "bank_k": BANK_K,
        },
        "puzzles": [
            {"id": p.id, "family": p.family, "expected": str(p.expected)}
            for p in PUZZLES
        ],
        "arm1_control": arm1,
        "arm2_bank": arm2,
        "arm3_bank_logged": arm3,
        "summary": {
            "control": s1,
            "bank_on": s2,
            "bank_on_logged": s3,
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
