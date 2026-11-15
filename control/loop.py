"""The control loop: camera frame in, arm position command out.

This is the only module in `control/` that touches hardware. The decisions it acts on come
from `control.safety` and `control.admittance`, which are pure and separately tested; what is
here is the wiring, the clock, and the conversion between metres and servo ticks.

Shape of one cycle:

    frame, ts  <- camera
    result     <- gauge_reader pipeline       (newtons, or None if the marker was not seen)
    decision   <- SafetyGate.check(...)       (HOLD / YIELD / RELEASE)
    step       <- admittance law, if YIELD
    set_position(...)                          (only ever on YIELD)

Design and the reasoning behind every limit: docs/admittance-design.md.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from control.admittance import AdmittanceParams, step_from_velocity, yield_velocity
from control.safety import Action, Decision, SafetyGate, SafetyLimits


@dataclass(frozen=True)
class AxisCalibration:
    """How a displacement along the tether maps onto one servo's position register.

    `ticks_per_metre` has to be measured on the bench: command a known tick delta, measure how
    far the attachment point actually moved along the pull axis. It depends on the arm's pose,
    so it is only valid for the pose the session runs in (the same reason the force
    calibration was pose-dependent, docs/force-calibration.md).

    `sign` is +1 or -1: which way the tick count has to go to *reduce* tension. Getting this
    backwards means yielding tightens the spring, so `ControlLoop` checks it against the
    measured force rather than trusting it (see `_check_yield_direction`).
    """

    motor_id: int
    ticks_per_metre: float
    sign: int = 1

    def __post_init__(self) -> None:
        if self.ticks_per_metre <= 0:
            raise ValueError(f"ticks_per_metre must be positive, got {self.ticks_per_metre}")
        if self.sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1, got {self.sign}")

    def ticks_for(self, step_m: float) -> int:
        return int(round(self.sign * step_m * self.ticks_per_metre))


@dataclass
class CycleRecord:
    """One cycle, for the session log and the force-displacement curve (roadmap item 6)."""

    t: float
    force_n: float | None
    action: str
    reason: str
    step_m: float
    travel_m: float
    position_ticks: int | None


@dataclass
class LoopState:
    start_ticks: int | None = None
    last_ticks: int | None = None
    last_t: float | None = None
    travel_m: float = 0.0
    records: list[CycleRecord] = field(default_factory=list)


class ControlLoop:
    """Drives one motor of arm B from the gauge reader's force readings.

    Not async, unlike `lib.station` - the caller owns the event loop and awaits the two I/O
    calls. That keeps this class testable with plain fakes (tests/control/test_loop.py).
    """

    def __init__(
        self,
        limits: SafetyLimits,
        params: AdmittanceParams,
        axis: AxisCalibration,
        gate: SafetyGate | None = None,
    ):
        self.limits = limits
        self.params = params
        self.axis = axis
        self.gate = gate if gate is not None else SafetyGate(limits)
        self.state = LoopState()
        self._yield_started_at: tuple[float, float] | None = None  # (t, force) at yield start

    # ------------------------------------------------------------------ one cycle, pure part
    def decide(
        self, force_n: float | None, now: float, frame_t: float, position_ticks: int | None
    ) -> tuple[Decision, float]:
        """Decide this cycle without performing any I/O. Returns (decision, step in metres).

        Separated from `step()` so the whole decision path, including the travel bookkeeping
        and the yield-direction check, is testable without a robot.
        """
        if position_ticks is not None:
            if self.state.start_ticks is None:
                self.state.start_ticks = position_ticks
            self.state.travel_m = (
                (position_ticks - self.state.start_ticks) / self.axis.ticks_per_metre
            )
            self.state.last_ticks = position_ticks

        decision = self.gate.check(force_n, age_s=now - frame_t, travel_m=self.state.travel_m)

        if decision.action is not Action.YIELD or force_n is None:
            self._yield_started_at = None
            return decision, 0.0

        if not self._check_yield_direction(force_n, now):
            self.gate.latched = Decision(
                Action.RELEASE,
                "yielding did not reduce the force; tether direction may be wrong",
                force_n,
            )
            return self.gate.latched, 0.0

        dt = 0.0 if self.state.last_t is None else now - self.state.last_t
        velocity = yield_velocity(force_n, self.limits.f_threshold_n, self.params)
        step_m = step_from_velocity(velocity, dt, self.params)
        return decision, step_m

    def _check_yield_direction(self, force_n: float, now: float, window_s: float = 0.5) -> bool:
        """While yielding, the force must actually come down.

        If the arm has been giving way for `window_s` and the force is no lower than when it
        started, the assumed pull direction is wrong and moving further makes it worse: the
        positive-feedback failure in docs/admittance-control.md. One check catches a flipped
        sign, a hand that moved, and a detached hook alike.
        """
        if self._yield_started_at is None:
            self._yield_started_at = (now, force_n)
            return True
        started_t, started_force = self._yield_started_at
        if now - started_t < window_s:
            return True
        return force_n < started_force

    # ------------------------------------------------------------------ one cycle, with I/O
    async def step(self, client, bus_serial: str, force_n: float | None, frame_t: float,
                   position_ticks: int | None, now: float | None = None) -> Decision:
        """Run one cycle and command the arm. Returns the decision taken."""
        from lib.station import set_position

        now = time.monotonic() if now is None else now
        decision, step_m = self.decide(force_n, now, frame_t, position_ticks)

        commanded = position_ticks
        if decision.action is Action.YIELD and step_m > 0 and position_ticks is not None:
            commanded = position_ticks + self.axis.ticks_for(step_m)
            await set_position(client, bus_serial, self.axis.motor_id, commanded)

        self.state.last_t = now
        self.state.records.append(
            CycleRecord(now, force_n, decision.action.value, decision.reason, step_m,
                        self.state.travel_m, commanded)
        )
        return decision

    async def release(self, client, bus_serial: str, back_off_m: float = 0.01) -> None:
        """Unload the spring and stop. Safe to call more than once.

        Moves a bounded amount in the slack direction rather than holding position: a stuck
        arm holding tension against a finger, with no force reading to see it, is the worst
        state this machine has (docs/admittance-design.md).
        """
        from lib.station import set_position

        if self.state.last_ticks is None:
            return
        target = self.state.last_ticks + self.axis.ticks_for(back_off_m)
        await set_position(client, bus_serial, self.axis.motor_id, target)
        self.state.last_ticks = target
