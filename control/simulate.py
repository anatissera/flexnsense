"""Closed-loop simulation: a scripted finger pulls against a spring, the controller responds.

For choosing constants without a human or a robot in the loop, and for re-checking them after
any change to the control law. It models the arm as able to move exactly where it is
commanded, so it answers questions about the *control law* (will it keep up, will it
oscillate, where does the force peak), not about the arm's dynamics.

    python -m control.simulate                      # the default sweep
    python -m control.simulate --damping 10 --pull 20

What it already established (see docs/admittance-design.md):
  - overshoot above the cap settles at B x pull_speed, so B is bounded by the
    F_threshold-to-F_max gap and belongs near 10-20 for this tether, not 100;
  - a 50 mm/s yank reaches the ceiling at any B, because 33 ms of travel against a 56.2 N/m
    tether is already ~0.10 N. That case is handled by releasing, not by tracking.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from control.admittance import AdmittanceParams, step_from_velocity, yield_velocity
from control.safety import Action, SafetyGate, SafetyLimits

# Bench-measured tether stiffness (spring + cord + attachments), scripts/measure_tether_stiffness.py
# against data/tether_stiffness.csv: 56.2 N/m. Stiffer than the PESOLA-alone datasheet shortcut
# (600 gf / ~110 mm = 53.5 N/m) because the cord adds its own give in series with the spring.
SPRING_N_PER_M = 56.2


@dataclass
class SimResult:
    peak_force_n: float
    arm_travel_m: float
    released_at_s: float | None
    release_reason: str
    cycles: int

    @property
    def released(self) -> bool:
        return self.released_at_s is not None

    def describe(self) -> str:
        end = (
            f"released at {self.released_at_s:.1f}s ({self.release_reason})"
            if self.released
            else f"ran {self.cycles} cycles clean"
        )
        return (
            f"peak {self.peak_force_n:.2f} N, arm moved {self.arm_travel_m * 1000:.0f} mm, {end}"
        )


def simulate(
    limits: SafetyLimits,
    params: AdmittanceParams,
    pull_speed_m_s: float,
    spring_n_per_m: float = SPRING_N_PER_M,
    fps: float = 30.0,
    duration_s: float = 30.0,
) -> SimResult:
    """Pull the finger away at a constant speed and let the controller respond each frame."""
    dt = 1.0 / fps
    gate = SafetyGate(limits)
    finger_m = arm_m = 0.0
    peak_n = 0.0

    for i in range(int(duration_s * fps)):
        finger_m += pull_speed_m_s * dt
        force_n = spring_n_per_m * max(0.0, finger_m - arm_m)
        peak_n = max(peak_n, force_n)

        decision = gate.check(force_n, age_s=0.0, travel_m=arm_m)
        if decision.action is Action.RELEASE:
            return SimResult(peak_n, arm_m, i * dt, decision.reason, i)
        if decision.action is Action.YIELD:
            v = yield_velocity(force_n, limits.f_threshold_n, params)
            arm_m += step_from_velocity(v, dt, params)

    return SimResult(peak_n, arm_m, None, "", int(duration_s * fps))


def _default_limits() -> SafetyLimits:
    return SafetyLimits(f_threshold_n=1.96, f_max_n=3.0, max_travel_m=0.15)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--damping", type=float, help="B, in N.s/m; omit to sweep")
    p.add_argument("--pull", type=float, help="pull speed in mm/s; omit to sweep")
    p.add_argument("--v-max", type=float, default=30.0, help="speed ceiling in mm/s")
    p.add_argument("--spring", type=float, default=SPRING_N_PER_M, help="spring rate in N/m")
    args = p.parse_args(argv)

    limits = _default_limits()
    dampings = [args.damping] if args.damping else [5, 10, 20, 50, 100]
    pulls = [args.pull] if args.pull else [4, 20, 50]

    print(f"cap {limits.f_threshold_n} N, ceiling {limits.f_max_n} N, "
          f"spring {args.spring} N/m, headroom {limits.f_max_n - limits.f_threshold_n:.2f} N\n")

    for b in dampings:
        for pull_mm_s in pulls:
            params = AdmittanceParams(
                damping_n_s_per_m=b, v_max_m_s=args.v_max / 1000.0, deadband_n=0.05
            )
            result = simulate(limits, params, pull_mm_s / 1000.0, spring_n_per_m=args.spring)
            predicted = b * (pull_mm_s / 1000.0)
            print(f"  B={b:<5g} pull={pull_mm_s:>3g} mm/s  (needs {predicted:4.2f} N overshoot)  "
                  f"{result.describe()}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
