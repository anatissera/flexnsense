"""The admittance law: how far to move this cycle, given how far over the cap the force is.

Pure, like control/safety.py, and for the same reason: every clamp here is a thing that keeps
a motor from doing something abrupt next to a finger, and each one should be readable and
testable on its own. Nothing in this module talks to a camera or a robot.

Clinically this is assist-as-needed: hold position until the patient's pull reaches the
prescribed cap, then give way just enough to stop the force rising. Design and the reasoning
behind the constants: docs/admittance-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdmittanceParams:
    """Units: newtons, metres, seconds (AGENTS.md).

    `damping_n_s_per_m` is the virtual damping B in `v = (F - F_threshold) / B`: larger means
    the arm gives way more slowly for the same overshoot. Start large (over-damped) and lower
    it on the bench only as far as stays oscillation-free.
    """

    damping_n_s_per_m: float          # B
    v_max_m_s: float                  # speed ceiling, regardless of how far over the cap
    deadband_n: float = 0.0           # overshoot below this is treated as zero
    max_step_m: float = 0.002         # per-cycle displacement ceiling, the arithmetic backstop

    def __post_init__(self) -> None:
        if self.damping_n_s_per_m <= 0:
            raise ValueError(f"damping must be positive, got {self.damping_n_s_per_m}")
        if self.v_max_m_s <= 0:
            raise ValueError(f"v_max_m_s must be positive, got {self.v_max_m_s}")
        if self.deadband_n < 0:
            raise ValueError(f"deadband cannot be negative, got {self.deadband_n}")
        if self.max_step_m <= 0:
            raise ValueError(f"max_step_m must be positive, got {self.max_step_m}")


def yield_velocity(force_n: float, f_threshold_n: float, params: AdmittanceParams) -> float:
    """Speed to give way at, in m/s along the tether axis. Never negative: this law only ever
    reduces tension, it does not pull the finger back."""
    overshoot = force_n - f_threshold_n
    if overshoot <= params.deadband_n:
        return 0.0
    return min(overshoot / params.damping_n_s_per_m, params.v_max_m_s)


def step_from_velocity(velocity_m_s: float, dt_s: float, params: AdmittanceParams) -> float:
    """Displacement for one cycle, in metres.

    `dt_s` must come from real frame timestamps, not a nominal frame rate: the camera's
    interval is not exactly constant and stutters under load, and a hardcoded interval turns
    silently into a speed error whenever a frame is late (docs/admittance-design.md).

    A non-positive or absurdly long `dt_s` yields no motion rather than a guess: a stalled or
    time-travelling clock is not a reason to move a finger.
    """
    if dt_s <= 0 or dt_s > 1.0:
        return 0.0
    return min(velocity_m_s * dt_s, params.max_step_m)
