"""Fit ticks_per_metre from (commanded tick delta, measured displacement) pairs, and infer
yield_sign from (tick delta, force before, force after) trials.

Both are read with the arm, not inferred from a force reading. An earlier version of this
measurement planned to get the displacement from the camera (`dx = dF / k`, using
`lib.stiffness`), which needs no ruler on the arm. That turned out not to hold up: the tether's
own hysteresis (`lib.stiffness.compare_directions`) sets a floor on how well any force-inferred
displacement can be known, and on this tether that floor (6 mm, `data/tether_stiffness.csv`) is
larger than either builder assumed. It dominates error at every tick delta small enough to keep
the force under the therapy cap: even a 60 mm move, most of the tether's usable travel, is only
good to 10%. A ruler reading is good to about its own precision (roughly 1 mm) regardless of how
far the arm moves, so it wins at any displacement actually reachable here. See
docs/admittance-design.md and scripts/measure_ticks_per_metre.py for the full argument.

`fit_ticks_per_metre` is forced through the origin: zero commanded ticks is zero displacement by
construction (it is the baseline the ruler reading started from), so there is no intercept to
estimate, only a magnitude.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

#: One trial: ticks commanded relative to the baseline position, and the ruler's measured
#: displacement relative to the baseline reading, in millimetres. Both signed, in whatever
#: convention the two reference marks were read in.
TickPoint = tuple[int, float]

#: One trial for sign inference: the tick delta commanded, and the force read just before and
#: just after, in whatever unit was read (grams-force off the PESOLA is the expected one).
ForcePoint = tuple[int, float, float]

#: A force change smaller than this is reading noise, not a response to the move - a trial this
#: small is excluded from the sign vote rather than letting noise cast one.
NOISE_FLOOR_GRAMS = 5.0


@dataclass(frozen=True)
class TicksPerMetreFit:
    ticks_per_metre: float
    n_points: int
    #: Per-trial implied ticks_per_metre (ticks_i / m_i, signed), for eyeballing consistency
    #: across trials before trusting the combined fit - a sign flip or an outlier shows up here.
    per_point_ticks_per_metre: tuple[float, ...]
    #: Largest fractional deviation of any one trial's magnitude from the combined fit.
    max_relative_deviation: float


def fit_ticks_per_metre(points: Sequence[TickPoint]) -> TicksPerMetreFit:
    """Through-origin least squares on (ticks, displacement in mm) pairs.

    Requires every trial to agree on which sign of tick delta goes with which sign of measured
    displacement - a real tether has one consistent relationship between the two (commanding the
    same direction again should move the mark the same way again), so a disagreement means the
    setup is suspect (wrong motor, a slipping mark, backlash swamping a too-small delta), not
    something to average through.
    """
    if len(points) < 2:
        raise ValueError(f"need at least 2 points, got {len(points)}")
    for ticks, mm in points:
        if mm == 0:
            raise ValueError(
                f"a trial with ticks={ticks} measured zero displacement; did the ruler reading "
                "actually change, or was the delta too small to see?"
            )

    metres = [mm / 1000.0 for _, mm in points]
    ticks_list = [t for t, _ in points]
    denominator = sum(m * m for m in metres)
    slope = sum(t * m for t, m in zip(ticks_list, metres)) / denominator
    if slope == 0:
        raise ValueError("fitted slope is exactly zero; something is wrong with this data")

    per_point = tuple(t / m for t, m in zip(ticks_list, metres))
    wrong_sign = [i for i, p in enumerate(per_point) if (p > 0) != (slope > 0)]
    if wrong_sign:
        raise ValueError(
            f"trial(s) at index {wrong_sign} moved the opposite way from the rest: a commanded "
            "tick delta in one direction should consistently stretch, or consistently slacken, "
            "the tether - not sometimes either. Check the motor id and the reference marks "
            "before trusting a fit through this data."
        )

    max_dev = max(abs(abs(p) - abs(slope)) / abs(slope) for p in per_point)
    return TicksPerMetreFit(
        ticks_per_metre=abs(slope),
        n_points=len(points),
        per_point_ticks_per_metre=per_point,
        max_relative_deviation=max_dev,
    )


def infer_yield_sign(trials: Sequence[ForcePoint]) -> int:
    """+1 or -1: the sign of the tick delta that reduces the measured force.

    Raises if no trial moved the force enough to tell, or if trials disagree about which
    direction that is - either means this should not be trusted yet, not averaged past.
    """
    votes: set[int] = set()
    for delta_ticks, before, after in trials:
        if delta_ticks == 0:
            continue
        change = after - before
        if abs(change) < NOISE_FLOOR_GRAMS:
            continue
        reduced = change < 0
        votes.add(1 if (delta_ticks > 0) == reduced else -1)

    if not votes:
        raise ValueError(
            "no trial changed the force by more than the noise floor "
            f"({NOISE_FLOOR_GRAMS:g} gf); use a bigger tick delta"
        )
    if len(votes) > 1:
        raise ValueError(
            "trials disagree about which tick direction reduces force - check that the PESOLA "
            "is anchored firmly along the tether axis and isn't slipping"
        )
    return votes.pop()
