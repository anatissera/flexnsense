"""The one place the gram-force <-> newton conversion constant lives.

Convention for this repo (see docs/contributing.md "Units"): the PESOLA is graduated in grams-force, and that is
what a human reads off it, what the gauge reader calibrates against, and what the clinician-facing
numbers use, because that is the unit hand-therapy literature and dynamometers already use. Everything
past that boundary, control math, thresholds, logged values consumed by the controller, any force or
torque equation, is in newtons (SI), so it composes with angles in radians, time in seconds and the
rest of the control stack without a second conversion constant hiding somewhere else.

Do not hardcode 9.80665 (or 9.81, or 1/1000) anywhere else in the repo: import from here.
"""
from __future__ import annotations

STANDARD_GRAVITY_M_S2 = 9.80665  # exact, by definition (BIPM); the same value dynamometers assume


def grams_force_to_newtons(grams_force: float) -> float:
    """1 gf is the weight of 1 g under standard gravity: N = kg * m/s^2."""
    return grams_force / 1000.0 * STANDARD_GRAVITY_M_S2


def newtons_to_grams_force(newtons: float) -> float:
    return newtons * 1000.0 / STANDARD_GRAVITY_M_S2
