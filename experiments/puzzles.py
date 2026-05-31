"""Multi-step logic puzzles for the ReasoningBank experiment.

Five families, shared gotchas. Each puzzle layers two-to-three traps:

* a familiar-looking surface that nudges the reader toward a canonical answer,
* a subtle modifier that flips one part of the calculation,
* and an embedded distractor or boundary condition that often gets dropped.

Each entry exposes a deterministic ``verify(answer)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


_NUM_RE = re.compile(r"-?\d+\.?\d*")
_HHMM_RE = re.compile(r"\b([0-2]?\d):([0-5]\d)\b")


def _extract_number(text: str) -> float | None:
    matches = _NUM_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _num_eq(expected: float, tol: float = 1e-2) -> Callable[[str], bool]:
    def check(answer: str) -> bool:
        num = _extract_number(answer)
        if num is None:
            return False
        if abs(expected - round(expected)) < 1e-9:
            return abs(num - expected) < 0.5
        return abs(num - expected) < tol

    return check


def _time_eq(hh: int, mm: int) -> Callable[[str], bool]:
    def check(answer: str) -> bool:
        matches = list(_HHMM_RE.finditer(answer))
        if not matches:
            return False
        m = matches[-1]
        return int(m.group(1)) == hh and int(m.group(2)) == mm

    return check


@dataclass
class Puzzle:
    id: str
    family: str
    question: str
    expected: Any
    verifier: Callable[[str], bool]


PUZZLES: list[Puzzle] = [
    # ---------- regime-change ----------
    Puzzle(
        id="snail_well",
        family="regime-change",
        question=(
            "A snail starts at the bottom of a well that is 100 feet deep. "
            "Each calendar day, the snail climbs upward at a rate of 5 feet "
            "during the daytime period; each calendar night it slides back "
            "DOWN at a rate of 4 feet. The snail stops moving forever the "
            "moment it first crosses the rim of the well at any point during "
            "a daytime climb. Important: the diameter of the well is 6 feet "
            "(this is irrelevant). After the snail's escape, three observers "
            "claim it took: 100 days, 96 days, and 97 days respectively. "
            "Which observer is correct? Answer ONLY with the integer number "
            "of days, do not include 'days' or any other text on the answer "
            "line."
        ),
        expected=96,
        verifier=_num_eq(96),
    ),
    Puzzle(
        id="chimney_growth",
        family="regime-change",
        question=(
            "A team builds a chimney. Each weekday (Mon-Fri) they add 7 m of "
            "new height during the day, and each weekday night the wind "
            "erodes 5 m off the top. The team does NOT work on weekends "
            "(Saturday and Sunday); during weekends the wind still erodes "
            "5 m PER NIGHT (i.e., 10 m total across the weekend). The target "
            "is 30 meters; once the chimney reaches 30 m during a weekday "
            "build, a cap is installed and erosion stops. The chimney starts "
            "at 0 m on a Monday morning. On which calendar day-number (Day "
            "1 = the first Monday, Day 2 = Tuesday, ...) does the chimney "
            "first reach 30 m? Answer with a single integer."
        ),
        # Pattern across one week:
        # Mon-build/Mon-night: +7-5=+2. Tue: +2. Wed: +2. Thu: +2. Fri-build/Fri-night: +2.
        # Then Sat-night: -5. Sun-night: -5. (Could clamp at 0; assume can clamp.)
        # Trajectory:
        # Day 1 Mon: 0 -> 7 -> 2.
        # Day 2 Tue: 2 -> 9 -> 4.
        # Day 3 Wed: 4 -> 11 -> 6.
        # Day 4 Thu: 6 -> 13 -> 8.
        # Day 5 Fri: 8 -> 15 -> 10.
        # Day 6 Sat: 10 -> 5 (night erosion).
        # Day 7 Sun: 5 -> 0 (night erosion). (Doesn't go negative.)
        # Day 8 Mon: 0 -> 7 -> 2.
        # ... same as day 1.
        # So end-of-Sunday height resets to 0 each week!
        # That means it NEVER escapes. Bad puzzle.
        # Recompute with no Sunday floor: end-of-Sun heights:
        # Week 1 end-of-Sun: 0.
        # Week 2 end-of-Sun: 0. Stuck.
        # SKIP, replace with cleaner version (below).
        expected=-1,  # placeholder, will fix in next puzzle
        verifier=_num_eq(-1),
    ),
    # ---------- discount-stacking ----------
    Puzzle(
        id="stacked_coupons",
        family="discount-stacking",
        question=(
            "A laptop is listed at $1,275.40. The store applies a 22% "
            "holiday discount. Then an 8.5% sales tax is applied to the "
            "post-discount price. Then a flat $75 mail-in rebate is "
            "subtracted at the very end. The receipt also lists a $0.00 "
            "shipping line and a $0.00 gift-wrap line (irrelevant). What is "
            "the final amount the customer ultimately pays, in dollars, "
            "rounded to the nearest cent?"
        ),
        # 1275.40 * 0.78 = 994.812. * 1.085 = 1079.36 (more precisely 1079.36...).
        # 1275.40 * 0.78 = 994.812
        # 994.812 * 1.085 = 1079.3711...
        # round to nearest cent: 1079.37. - 75 = 1004.37
        expected=1004.37,
        verifier=_num_eq(1004.37, tol=0.05),
    ),
    Puzzle(
        id="tax_then_discount",
        family="discount-stacking",
        question=(
            "A clothing item is priced at $137. Two discounts are applied IN "
            "ORDER: first 18% off the sticker price, then a flat $12 off the "
            "already-discounted price. A 6.25% sales tax is then applied to "
            "the ORIGINAL sticker price (not the discounted price). The "
            "customer pays (discounted price + tax). The store also offers "
            "free parking validation (irrelevant). What is the total in "
            "dollars (round to the nearest cent)?"
        ),
        # 137 * 0.82 = 112.34. - 12 = 100.34. Tax = 137 * 0.0625 = 8.5625 -> 8.56.
        # Total = 100.34 + 8.56 = 108.90.
        # If tax is computed exactly (not rounded mid): 100.34 + 8.5625 = 108.9025 -> 108.90.
        expected=108.90,
        verifier=_num_eq(108.90, tol=0.05),
    ),
    Puzzle(
        id="bakery_discount",
        family="discount-stacking",
        question=(
            "A bakery sells loaves at $5 each, with a 'buy 4 get 1 free' "
            "promotion (i.e., for every 5 loaves added to the cart, only 4 "
            "are paid for). The customer adds 14 loaves to the cart. A 12% "
            "loyalty discount is then applied to the subtotal (after the "
            "B4G1F deduction). A 7% sales tax is applied to the discounted "
            "amount. The bakery is open until 8 PM (irrelevant). What does "
            "the customer pay in dollars (round to the nearest cent)?"
        ),
        # 14 loaves: floor(14/5) = 2 free, pay for 12. 12 * 5 = 60.
        # 60 * 0.88 = 52.80. 52.80 * 1.07 = 56.496 -> 56.50.
        expected=56.50,
        verifier=_num_eq(56.50, tol=0.05),
    ),
    # ---------- elapsed-time ----------
    Puzzle(
        id="fence_posts",
        family="elapsed-time",
        question=(
            "A CIRCULAR fence encloses a garden. Posts are placed at exactly "
            "even intervals of 4 meters along the fence. The total perimeter "
            "of the fence is 40 meters. The garden contains 12 rose bushes "
            "(irrelevant). How many posts are needed? Answer with a single "
            "integer."
        ),
        # Loop: 40/4 = 10. Linear-fence trap = 11.
        expected=10,
        verifier=_num_eq(10),
    ),
    Puzzle(
        id="overnight_flight",
        family="elapsed-time",
        question=(
            "A flight departs Sydney at 9:50 PM local time on November 14. "
            "Sydney is UTC+10 (ignore daylight saving). The flight arrives "
            "in Los Angeles at 6:15 PM LOCAL time on November 14 (yes, the "
            "SAME calendar date — the flight crossed the International Date "
            "Line eastbound). Los Angeles is UTC-8 in November. The aircraft "
            "is a Boeing 787-9 with 296 seats (irrelevant). How many minutes "
            "was the aircraft in the air? Answer with a single integer."
        ),
        # Depart 21:50 UTC+10 -> 11:50 UTC Nov 14.
        # Arrive 18:15 UTC-8 -> 02:15 UTC Nov 15.
        # Delta = 14h 25min = 865 min.
        expected=865,
        verifier=_num_eq(865),
    ),
    # ---------- relative-speed ----------
    Puzzle(
        id="approaching_trains",
        family="relative-speed",
        question=(
            "Stations X and Y are 360 km apart on a straight track. Train A "
            "leaves Station X at 10:00 AM travelling east at 60 km/h "
            "(towards Station Y). Train B leaves Station Y at 11:00 AM "
            "travelling west at 90 km/h (towards Station X). The trains use "
            "parallel tracks. Each train has 8 carriages (irrelevant). At "
            "what time do they pass each other? Answer using 24-hour HH:MM "
            "format on a single line."
        ),
        # At 11:00 A has covered 60 km. 300 km remain. Closing 150 km/h.
        # 300/150 = 2h. Meet at 13:00. Trap: 360/150 from 10am = 12:24.
        expected="13:00",
        verifier=_time_eq(13, 0),
    ),
    Puzzle(
        id="round_trip_avg",
        family="relative-speed",
        question=(
            "A delivery van drives from town A to town B at a constant "
            "40 km/h, then immediately returns from B to A along the same "
            "route at a constant 24 km/h. The driver brags to a friend that "
            "the round-trip average speed must be (40 + 24) / 2 = 32 km/h. "
            "The van uses 30 liters of fuel total (irrelevant). What is the "
            "ACTUAL round-trip average speed in km/h? Answer with a single "
            "number; round to one decimal place if not exact."
        ),
        # 2*40*24/64 = 30. Trap: 32.
        expected=30.0,
        verifier=_num_eq(30.0),
    ),
    Puzzle(
        id="meeting_walk",
        family="relative-speed",
        question=(
            "Alice walks from home to school along a straight path at "
            "5 km/h, leaving at 8:00 AM. Bob walks from school to home along "
            "the same path at 4 km/h, but he leaves at 8:30 AM. Home and "
            "school are 5.5 km apart. Both are wearing red jackets "
            "(irrelevant). At what time do Alice and Bob meet on the path? "
            "Answer using 24-hour HH:MM format on a single line."
        ),
        # By 8:30 Alice walked 2.5 km. 3 km remain. Closing 9 km/h. 1/3 h = 20 min.
        # Meet at 8:50.
        expected="08:50",
        verifier=_time_eq(8, 50),
    ),
    # ---------- fill-drain ----------
    Puzzle(
        id="leaky_tank",
        family="fill-drain",
        question=(
            "A water tank has capacity 600 liters. An inlet pipe fills at "
            "30 L/min. A drain valve is open at the start, removing water "
            "at 10 L/min. The tank starts EMPTY. After exactly 12 minutes "
            "of filling, a worker notices and CLOSES the drain (the inlet "
            "keeps running at 30 L/min). The tank is made of stainless "
            "steel (irrelevant). How many TOTAL minutes (counted from "
            "t = 0) does it take to completely fill the tank? Answer with a "
            "single integer."
        ),
        # First 12min: net 20 L/min -> 240 L. Remaining 360 L. At 30 L/min: 12 min.
        # Total 24 min.
        expected=24,
        verifier=_num_eq(24),
    ),
    Puzzle(
        id="two_pipes_one_drain",
        family="fill-drain",
        question=(
            "Pipe X alone fills a pool in 6 hours. Pipe Y alone fills it in "
            "4 hours. A drain Z alone empties a full pool in 12 hours. All "
            "three are opened simultaneously on an EMPTY pool. After exactly "
            "1 hour, the drain Z is closed; pipes X and Y continue. The pool "
            "is rectangular, 10 m by 4 m by 1.5 m (irrelevant). From the "
            "moment the drain closes, how many MORE hours are needed to "
            "finish filling the pool? Answer with a single number; round to "
            "two decimal places if not exact."
        ),
        # First hour: rate 1/3 per hour -> 1/3 full.
        # After: rate 5/12 per hour. Remaining 2/3.
        # Time = (2/3)/(5/12) = 8/5 = 1.6 hours.
        expected=1.60,
        verifier=_num_eq(1.60, tol=0.02),
    ),
]


# Fix the chimney_growth puzzle with a clean weekend-erosion version that
# actually has a finite answer. (Earlier draft cycled at 0 — that was a bug
# in the puzzle, not the model.)
def _replace(pid: str, new: Puzzle) -> None:
    for i, p in enumerate(PUZZLES):
        if p.id == pid:
            PUZZLES[i] = new
            return
    raise KeyError(pid)


_replace(
    "chimney_growth",
    Puzzle(
        id="chimney_growth",
        family="regime-change",
        question=(
            "A team builds a chimney. Each day they add 7 meters of new "
            "height during the day. Each night, wind erodes 5 meters off "
            "the top — UNLESS that night is foggy, in which case 0 meters "
            "are eroded. Every third night is foggy (nights 3, 6, 9, ...). "
            "The target height is 30 meters; once the chimney reaches 30 m "
            "during a daytime build, a cap is installed and no further "
            "erosion occurs. The chimney starts at 0 m. The team consists "
            "of 5 masons (irrelevant). On which day does the chimney first "
            "reach 30 meters? Answer with a single integer."
        ),
        # State after night N:
        # Day 1: 0 -> 7. Night 1: -5 -> 2.
        # Day 2: 2 -> 9. Night 2: -5 -> 4.
        # Day 3: 4 -> 11. Night 3: foggy, 0 -> 11.
        # Day 4: 11 -> 18. Night 4: -5 -> 13.
        # Day 5: 13 -> 20. Night 5: -5 -> 15.
        # Day 6: 15 -> 22. Night 6: foggy, 0 -> 22.
        # Day 7: 22 -> 29. Night 7: -5 -> 24.
        # Day 8: 24 -> 31. REACH 30 during day 8.
        # Answer: 8. Trap: 30 / (7-5)*something.
        expected=8,
        verifier=_num_eq(8),
    ),
)


def verify(puzzle_id: str, answer: str) -> bool:
    for p in PUZZLES:
        if p.id == puzzle_id:
            return p.verifier(answer or "")
    raise KeyError(f"Unknown puzzle id: {puzzle_id}")


def get(puzzle_id: str) -> Puzzle:
    for p in PUZZLES:
        if p.id == puzzle_id:
            return p
    raise KeyError(f"Unknown puzzle id: {puzzle_id}")
