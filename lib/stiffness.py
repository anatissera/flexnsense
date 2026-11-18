"""Fit a spring constant from (force, displacement) pairs read off the PESOLA and a ruler.

Device-agnostic on purpose: this is least squares on a line, and it is used both for the tether
as a whole (spring plus cord plus attachments, which is the number the controller needs) and for
anything else elastic that gets measured the same way.

Units, per docs/contributing.md: the measurement is entered in the units a human reads, grams-force off the
PESOLA and millimetres off a ruler, and everything this module *returns* is SI, so `k` comes out
in N/m ready for the control math. The conversion happens here and nowhere else.

The fit is `F = k * (x - x0)`:

  - `k` is the stiffness, the only number the admittance loop needs (it converts a force change
    into the displacement that caused it, which is how `ticks_per_metre` gets measured without a
    ruler on the arm).
  - `x0` is the reading at zero force, i.e. where the free length of the span sits on whatever
    pair of reference marks was measured between. It is a nuisance parameter: it absorbs the
    arbitrary choice of marks, which is exactly why the raw ruler readings can be logged without
    first subtracting anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lib.units import grams_force_to_newtons, newtons_to_grams_force

#: One (grams-force, millimetres) measurement, in the units they were read in.
Point = tuple[float, float]


@dataclass(frozen=True)
class StiffnessFit:
    """The line through one sweep. `k_n_per_m` is the result; the rest says whether to trust it."""

    k_n_per_m: float
    #: Ruler reading extrapolated to zero force, in millimetres, in the same frame as the input.
    x0_mm: float
    #: Coefficient of determination. High on its own proves nothing (see force-calibration-result.md);
    #: read it together with `residual_max_mm`.
    r2: float
    n_points: int
    #: Worst single-point distance from the fitted line, in millimetres. This is the honest error
    #: bar: a ruler read to the nearest millimetre cannot do better than ~1 mm here.
    residual_max_mm: float

    @property
    def k_gf_per_mm(self) -> float:
        """The same stiffness in the units it was measured in, for sanity-checking by eye."""
        return newtons_to_grams_force(self.k_n_per_m) / 1000.0

    def displacement_m(self, force_delta_n: float) -> float:
        """How far the span stretched, for a measured change in force. The inverse of the spring law.

        This is what makes the `ticks_per_metre` measurement possible with no ruler on the arm:
        the camera reports dF, this reports dx, and the commanded tick delta divides into it.
        """
        return force_delta_n / self.k_n_per_m


def fit_stiffness(points: Sequence[Point]) -> StiffnessFit:
    """Least-squares `F = k * (x - x0)` through (grams-force, millimetre) pairs."""
    if len(points) < 2:
        raise ValueError(f"need at least 2 points to fit a line, got {len(points)}")
    grams = np.array([p[0] for p in points], dtype=np.float64)
    mm = np.array([p[1] for p in points], dtype=np.float64)
    if np.ptp(mm) == 0:
        raise ValueError("all displacement readings are identical; nothing to fit")

    newtons = grams_force_to_newtons(grams)
    metres = mm / 1000.0
    slope, intercept = np.polyfit(metres, newtons, 1)
    if slope <= 0:
        raise ValueError(
            f"fitted stiffness is {slope:.1f} N/m, not positive: more force must mean more "
            "stretch, so the two reference marks were probably read in opposite senses"
        )

    predicted = slope * metres + intercept
    ss_res = float(np.sum((newtons - predicted) ** 2))
    ss_tot = float(np.sum((newtons - newtons.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    residual_mm = np.abs(newtons - predicted) / slope * 1000.0

    return StiffnessFit(
        k_n_per_m=float(slope),
        x0_mm=float(-intercept / slope) * 1000.0,
        r2=float(r2),
        n_points=len(points),
        residual_max_mm=float(residual_mm.max()),
    )


@dataclass(frozen=True)
class HysteresisPoint:
    """One force level measured both going up and coming back down."""

    grams: float
    up_mm: float
    down_mm: float

    @property
    def gap_mm(self) -> float:
        """Signed: positive means the spring sat longer on the way down, the usual direction."""
        return self.down_mm - self.up_mm


def compare_directions(up: Sequence[Point], down: Sequence[Point]) -> list[HysteresisPoint]:
    """Pair the two sweeps by force level. Only levels measured in both directions are returned.

    The gap is the spring's own hysteresis, and it bounds how well any displacement inferred from
    a force reading can do: an inferred `dx` is only meaningful to within this.
    """
    down_by_grams = {g: mm for g, mm in down}
    return [
        HysteresisPoint(grams=g, up_mm=mm, down_mm=down_by_grams[g])
        for g, mm in up
        if g in down_by_grams
    ]
