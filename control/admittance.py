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
    """Units: newtons, metres, seconds (docs/contributing.md).

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


def recover_velocity(force_n: float, f_threshold_n: float, params: AdmittanceParams,
                     gain: float = 1.0, floor_n: float = 0.0) -> float:
    """Speed to take slack back up at, in m/s along the tether axis, when the finger eases off.

    The mirror of `yield_velocity`, for the return stroke: the same law, driven by how far
    *under* the threshold the force is instead of over it. Yielding alone leaves the arm
    parked wherever the furthest flex pushed it, so as soon as the finger relaxes the tension
    collapses and the exercise's resistance is gone until the next session re-rigs. Following
    the finger back keeps the tether near the setpoint through the whole rep, out and back,
    which is what "the arm accompanies the finger" means on the return half.

    Never negative, same clamps as yielding (`deadband_n` so resting noise does not twitch
    the arm, `v_max_m_s` as the hard speed ceiling). `gain` scales how eagerly it chases the
    slack relative to how eagerly yielding gives way; 1.0 treats both directions alike.

    `floor_n` stops the chase once `force_n` is already at or below it, rather than letting
    the law keep commanding speed toward zero force. Measured on the bench: a finger that
    relaxes faster than `v_max_m_s` can follow drops the tether to true slack well before the
    arm's single-axis travel bookkeeping (frozen at the pull's starting pose, see
    `lib.kinematics.Chain.ticks_per_metre_along`) catches up and tells `ControlLoop` to stop.
    Recovery kept chasing the vanishing force all the way toward zero, past the tether's
    normal resting tension (`session.reps.RepMachine.pretension_n`), slack enough that the
    camera lost the PESOLA's reference markers and a *different* watchdog
    (`session.reps.RepMachine._deadman`, 300 ms with no usable reading) aborted the
    repetition. The right fix is a properly implemented `Law.PULL`/`Target.REP_START` motion
    (`session/modes.py`, not yet built: see docs/session-orchestration.md); this floor is the
    bounded stand-in that keeps the same failure from recurring in the meantime.

    What this function still does not know: whether moving this way is *allowed* at all. It
    computes a speed; `ControlLoop` owns the rule that recovery never retreats behind the
    session's starting position, because "how fast" is a law and "no further than" is a
    boundary, and keeping them separate is what keeps both testable.
    """
    # floor_n <= 0 means disabled, not "stop at literally zero force": 0.0 is the default and
    # has to keep recovering at true slack exactly as it did before this parameter existed
    # (a real force reading of 0.0 N is itself a legitimate case that must not get special-
    # cased into stopping).
    if floor_n > 0.0 and force_n <= floor_n:
        return 0.0
    slack = f_threshold_n - force_n
    if slack <= params.deadband_n:
        return 0.0
    return min(gain * slack / params.damping_n_s_per_m, params.v_max_m_s)


def step_from_velocity(velocity_m_s: float, dt_s: float, params: AdmittanceParams,
                       max_step_m: float | None = None) -> float:
    """Displacement for one cycle, in metres.

    `dt_s` must come from real frame timestamps, not a nominal frame rate: the camera's
    interval is not exactly constant and stutters under load, and a hardcoded interval turns
    silently into a speed error whenever a frame is late (docs/admittance-design.md).

    A non-positive or absurdly long `dt_s` yields no motion rather than a guess: a stalled or
    time-travelling clock is not a reason to move a finger.

    `max_step_m`, when given, overrides `params.max_step_m` for this call only. Exists for
    `ControlLoop`'s recovery step, which is a different motion with different reasons to be
    bounded (it can never retreat past the session's start) and no reason to share the
    *same number* as the yield step it is not. Sharing one bound was a real bug on the bench:
    `--max-step-mm` caps the yield step's smoothness against a flexing finger, so raising it
    to make a slow return faster also loosened that unrelated bound, which nobody wanted. At
    the camera's real, loaded frame rate (well under its nominal 30 fps - see
    `scripts/session/run_session.py`'s main loop), `velocity_m_s * dt_s` routinely exceeds a few
    millimetres, so whichever cap is in force wins outright: dozens of consecutive recovery
    cycles were seen pinned at exactly `params.max_step_m`, meaning the requested recovery or
    retrace speed had no effect at all until this split.
    """
    if dt_s <= 0 or dt_s > 1.0:
        return 0.0
    cap = params.max_step_m if max_step_m is None else max_step_m
    return min(velocity_m_s * dt_s, cap)
