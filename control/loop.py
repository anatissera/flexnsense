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

from control.admittance import (
    AdmittanceParams,
    recover_velocity,
    step_from_velocity,
    yield_velocity,
)
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


@dataclass(frozen=True)
class CoordinatedAxis:
    """A straight pull along the tether, using several joints at once.

    The reason this exists rather than `AxisCalibration`: a revolute joint moves the gripper
    along an **arc**. Over a few millimetres that is indistinguishable from a line, so a short
    bench test would not catch it, but one full-range repetition needs 72 to 88 mm of travel
    (`config/calibration/bench.md`). Over that distance a single joint leaves the tether's axis
    by more than 20 mm, measured from NormaCore's URDF. The tether does not bend, so that
    deviation becomes side-load on the finger or slack in the line, and neither is the exercise
    the controller believes it is running.

    `motors` must be able to span three dimensions, which takes **three independent joints**:
    two give a rank-2 jacobian whose reachable directions are a plane, so a pull that happens
    to lie in it works while the return stroke misses by millimetres. `scripts/which_motor.py`
    and `lib.kinematics.Chain.jacobian` are how that is checked for a given pose.

    `direction` is the tether's axis in the arm's base frame, pointing the way that **increases**
    tension. Yielding moves the opposite way, so `ControlLoop` can keep checking the result
    against the measured force exactly as it does for a single axis.
    """

    motors: list[int]
    direction: tuple[float, float, float]
    #: Largest tick change any one joint may be given in a single cycle. The per-cycle step is
    #: already bounded in metres by `AdmittanceParams.max_step_m`; this is the second bound, in
    #: the joint space, so a badly conditioned pose cannot turn a small tip motion into a large
    #: joint motion (see `lib.kinematics.Chain.ik_step` on damping).
    max_ticks_per_cycle: int = 60
    #: Raw ST3215 units written to RAM_GOAL_SPEED and RAM_ACC before each cycle's position
    #: command, scaled per joint by `lib.station.scaled_speeds` so the three arrive together.
    #: Tuned on the bench with `scripts/jog_coordinated.py`, which is also how new values
    #: should be found if the arm's load changes: 700/50 tracked a straight line with no
    #: visible wobble at a pace that kept up with a hand-paced pull; well under that and a
    #: step looks sluggish, well over it and gear noise and overshoot both show up again.
    speed: int = 700
    accel: int = 50

    def __post_init__(self) -> None:
        if len(self.motors) < 3:
            raise ValueError(
                f"a straight pull needs at least three independent joints, got {self.motors}. "
                "Two span only a plane, which fails on the return stroke"
            )
        if len(set(self.motors)) != len(self.motors):
            raise ValueError(f"repeated motor in {self.motors}")
        if self.max_ticks_per_cycle <= 0:
            raise ValueError(f"max_ticks_per_cycle must be positive, got {self.max_ticks_per_cycle}")
        if self.speed <= 0:
            raise ValueError(f"speed must be positive, got {self.speed}")
        if self.accel <= 0:
            raise ValueError(f"accel must be positive, got {self.accel}")
        norm = sum(c * c for c in self.direction) ** 0.5
        if norm < 1e-9:
            raise ValueError("direction must not be the zero vector")

    @classmethod
    def for_pose(cls, pose_ticks: dict[int, float], direction, chain=None,
                 distance_m: float = 0.090, travel_direction=None,
                 tick_ranges: dict[int, tuple[float, float]] | None = None,
                 **kw) -> "CoordinatedAxis":
        """Build one by choosing the joints that suit *this* pose, and check the pull is real.

        The alternative, naming joints once in a config file, is wrong because the right set
        depends on the pose: `lib.kinematics.Chain.choose_pull_motors` picks a different three
        for a folded arm than for an extended one. This also refuses outright when the pull is
        not followable from here, which is a real condition rather than a theoretical one: a
        pose can start well conditioned and drive itself into a singularity partway through a
        90 mm pull, tracking the first few millimetres perfectly and then wandering.

        `travel_direction` is the way the gripper will actually be moved, and defaults to
        `direction`. They are not the same thing for a yielding controller: `direction` is the
        sense that *increases* tension, and `ControlLoop` only ever commands the opposite, so
        it passes `-direction`. The distinction is not cosmetic, because feasibility is not
        symmetric. Measured on this arm at the pose of 2026-11-15, joints [1, 2, 4] hold 54 mm
        per radian for a full 90 mm of yielding, while the same joints travelling the other way
        from the same pose collapse to 1 mm per radian after about 30 mm. Checking the stroke
        that never happens would have refused a pull that is entirely sound.

        `tick_ranges` is passed straight through to `can_pull_straight`. Leaving it `None`
        checks straightness and conditioning only, the same as before this parameter existed;
        it does not check whether any joint runs past its real travel partway along the path,
        which is a failure that looks identical to a perfectly good pull right up until a
        joint pins against its stop and the other two quietly bend the line to compensate.
        """
        if chain is None:
            from lib.kinematics import load_chain

            chain = load_chain()
        if travel_direction is None:
            travel_direction = direction
        motors = chain.choose_pull_motors(pose_ticks, direction)
        ok, why = chain.can_pull_straight(motors, pose_ticks, travel_direction, distance_m,
                                          tick_ranges=tick_ranges)
        if not ok:
            raise ValueError(
                f"cannot pull straight from this pose with joints {motors}: {why}"
            )
        d = tuple(float(v) for v in direction)
        return cls(motors=motors, direction=d, **kw)

    def unit_direction(self):
        import numpy as np

        d = np.array(self.direction, dtype=float)
        return d / np.linalg.norm(d)

    def ticks_for(self, step_m: float, pose_ticks: dict[int, float], chain,
                  residual: dict[int, float] | None = None) -> dict[int, int]:
        """Tick changes that move the gripper `step_m` along the tether, sharing the work.

        `step_m` is positive to give way, matching `AxisCalibration.ticks_for`, so the motion
        is along the direction that *reduces* tension.

        `residual` carries the fraction of a tick each joint was owed but could not be given,
        and is **updated in place** for the next call. Pass one for any repeated pull; without
        it the straightness quietly collapses at exactly the step sizes this controller spends
        most of its time at.

        Why it matters this much: a straight line is a *ratio* between the joints, and the
        servo takes whole ticks. One 1 mm step here wants joints [1, 2, 4] to move
        (-1.23, +9.98, -6.24) ticks, which rounds to (-1, +10, -6) and is still the right
        shape. The admittance law, holding just above the cap, asks for far smaller steps: at
        0.2 mm the same ratio is (-0.25, +2.0, -1.25), and rounding that independently each
        cycle throws away a quarter of the motion in a direction that does not average out.
        Simulated over a 34 mm pull, that is 8.12 mm off the tether's line against 0.59 mm for
        the same pull taken in 1 mm steps. The tether does not bend, so that is side-load on a
        finger. Accumulating the remainder brings it back to 0.3 mm.
        """
        delta = self.unit_direction() * (-step_m)
        raw = chain.ik_step(self.motors, pose_ticks, delta)
        wanted = ({m: v + residual.get(m, 0.0) for m, v in raw.items()}
                  if residual is not None else raw)

        out = {m: int(round(v)) for m, v in wanted.items()}
        biggest = max((abs(v) for v in out.values()), default=0)
        if biggest > self.max_ticks_per_cycle:
            scale = self.max_ticks_per_cycle / biggest
            out = {m: int(round(v * scale)) for m, v in out.items()}
            if residual is not None:
                # The clamp is a safety limit, not a rounding error. Carrying the difference
                # forward would hand it back as a burst on a later cycle, which is the one
                # thing the per-cycle bound exists to prevent.
                residual.clear()
            return out

        if residual is not None:
            residual.update({m: wanted[m] - out[m] for m in out})
        return out


@dataclass(frozen=True)
class PullPlan:
    """A device-driven move toward a position target: `session.modes.Law.PULL`.

    The arm is the source of motion here, which inverts almost every assumption the yield law
    rests on. `yield_velocity` answers "the force is too high, how fast do I get out of the
    way"; there is no such question when the arm is the one moving a relaxed finger, because
    under the cap the force says nothing about how fast to go. So speed is a *setting* here,
    not a computed response, and the thing that says when to stop is the position target.

    That is also why this mode can be quicker and smoother than yielding. A yielding arm is
    chasing a signal and its pace is bounded by how promptly it can react to one; a pull is
    executing a path that was decided before it started, so it is bounded only by what the
    joints can do without wobbling (`scripts/jog_coordinated.py` is where that was found).

    Two rules from `session.modes.PhasePlan.device_driven` are *not* optional and are enforced
    by `ControlLoop`, not here: `control.safety.Decision.may_move` must not gate this motion
    (it is true only on YIELD, which is exactly backwards for a phase that must advance while
    the force is *under* the cap), and `_check_yield_direction` does not apply (it assumes the
    force should be falling, and while the arm pulls a rising force is the expected thing).
    """

    #: Where `travel_m * axis.sign` should end up. Negative targets raise tension (the arm
    #: stretches the finger toward extension); positive ones pay line out. Expressed in the
    #: same signed "how far has this arm given way" quantity the yield and recovery steps use,
    #: so one bookkeeping and one travel limit cover every motion this loop can command.
    target_yielded_m: float
    #: How fast to travel, in m/s along the tether. A setting, for the reason above.
    speed_m_s: float
    #: Per-cycle displacement ceiling, overriding `AdmittanceParams.max_step_m` when given.
    #: Separate for the same reason recovery's is: `max_step_m` bounds the yield step's
    #: smoothness against a flexing finger, and a pull is not that motion.
    max_step_m: float | None = None
    #: The force at which this pull stops raising tension, overriding the session cap.
    #:
    #: Exists for `session.modes.Phase.ARMED`, which is the one pull bounded by a force rather
    #: than by a position: it takes the slack up to the tether's resting tension and must stop
    #: *there*, well below the cap, because its whole job is to leave the finger pre-tensioned
    #: and not already working. `Target.NONE` is honest about it having no position target, so
    #: without this the only thing left to stop it would be the cap, and arming would hand the
    #: patient a finger already at therapeutic force while still being cued.
    #:
    #: `None` means the session cap (strictly, `SafetyLimits.yield_from_n`), which is right for
    #: every position-targeted pull: there the position is what ends the stroke and the cap is
    #: the early stop for a finger stiffer than the prescription assumed.
    stop_above_n: float | None = None

    def __post_init__(self) -> None:
        if self.speed_m_s <= 0:
            raise ValueError(f"speed_m_s must be positive, got {self.speed_m_s}")
        if self.max_step_m is not None and self.max_step_m <= 0:
            raise ValueError(f"max_step_m must be positive or None, got {self.max_step_m}")
        if self.stop_above_n is not None and self.stop_above_n <= 0:
            raise ValueError(f"stop_above_n must be positive or None, got {self.stop_above_n}")


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
        coordinated: "CoordinatedAxis | None" = None,
        yield_check_window_s: float | None = 0.5,
        recover_gain: float | None = None,
        recover_floor_n: float = 0.0,
        recover_retrace_m_s: float = 0.0,
    ):
        """`axis` names the joint whose travel is tracked, and drives the arm on its own when
        `coordinated` is None.

        Pass `coordinated` to pull along a straight line with several joints instead. `axis`
        stays required either way: the travel limit and the yield-direction check both measure
        progress against one joint's register, and keeping that unchanged means the safety
        path is identical whether one motor moves or three.

        `yield_check_window_s` is `_check_yield_direction`'s grace period: how long the arm may
        yield without the measured force actually coming down before this concludes the pull
        direction is wrong and releases. Its assumption, "giving way reduces tension", holds
        when the arm's yield speed matches or exceeds how fast the finger is adding tension; it
        does not hold if a person flexes faster than the arm's `v_max_m_s` can give way, in
        which case net tension keeps rising even with the direction and the motion both
        correct, and this check releases for the wrong reason. Passing `None` disables it
        entirely: on hardware too weak to injure anyone even while moving in a genuinely wrong
        direction (confirmed for this bench, a 3D-printed ElRobot with no force capable of
        harming a person), that failure mode is a nuisance rather than a hazard, so the
        decision of whether to carry it is left to the caller rather than hardcoded. The
        ceiling (`SafetyLimits.f_max_n`) and the camera dropout check are unrelated and keep
        running regardless of this setting.

        `recover_gain`, when not `None`, lets the arm take slack back up on the return
        stroke: while the gate says HOLD with a live reading and the arm has previously
        yielded ground, it moves back toward where the session started at
        `control.admittance.recover_velocity`'s speed (this value is that function's `gain`).
        `None`, the default, is the original behaviour: the arm only ever gives way, and the
        finger coming back simply slackens the tether. The recovery is bounded in three ways
        that are not negotiable with the gain: it never retreats past the session's starting
        position, it is speed- and step-clamped by the same `AdmittanceParams` as yielding,
        and it only runs on a real reading under HOLD - a dropout or a release stops it like
        it stops everything else.

        `recover_floor_n` is the other bound recovery needs and does not get from the tick
        boundary alone: a force floor below which it stops chasing, rather than commanding
        speed toward zero force. The single-axis travel bookkeeping used for the tick boundary
        is computed once from the pull's starting pose (`lib.kinematics.Chain
        .ticks_per_metre_along`), and over a long coordinated pull it is no longer exactly
        right by the time the arm is retreating - which on the bench meant recovery outran
        that boundary and drove the tether to true slack (losing the camera's reference
        markers, and triggering `session.reps.RepMachine`'s own 300 ms deadman) before the
        tick-based stop caught up. Defaults to 0.0, the force never stops it, matching the
        original behaviour; pass `session.reps.RepMachine.pretension_n` to stop recovery at
        the tether's normal resting tension instead of at true slack.

        `recover_floor_n` creates its own gap, which `recover_retrace_m_s` closes: once force
        reaches the floor, `recover_velocity` has nothing left to say (it has already reached
        the value it was told to protect), so without `recover_retrace_m_s` the arm simply
        stops wherever it is - which on the bench, whenever a finger relaxed faster than
        `v_max_m_s` could follow, left it tens of millimetres from the session's start with
        no force signal left to carry it the rest of the way. `recover_retrace_m_s > 0`
        covers exactly that gap: once the force-driven velocity is zero, it finishes the
        remaining distance at this fixed, slow speed instead, bounded the same way as every
        other recovery step (never past the start, step- and speed-clamped). `0.0`, the
        default, disables it and keeps the arm stopping at the floor, as before this
        parameter existed.
        """
        self.limits = limits
        self.params = params
        self.axis = axis
        self.coordinated = coordinated
        self.gate = gate if gate is not None else SafetyGate(limits)
        self.yield_check_window_s = yield_check_window_s
        self.recover_gain = recover_gain
        self.recover_floor_n = recover_floor_n
        self.recover_retrace_m_s = recover_retrace_m_s
        self.state = LoopState()
        self._yield_started_at: tuple[float, float] | None = None  # (t, force) at yield start
        self._chain = None   # lazily loaded URDF, only when coordinated motion is used
        #: Sub-tick motion owed to each joint, carried between cycles. See
        #: `CoordinatedAxis.ticks_for`: without it a pull taken in the small steps the
        #: admittance law produces wanders off the tether's line.
        self._tick_residual: dict[int, float] = {}
        #: RAM_ACC rarely needs to change mid-session, unlike speed, which is recomputed every
        #: cycle from that cycle's deltas. Written once, lazily, rather than every cycle at
        #: 30 fps for a value that is not moving.
        self._accel_set = False

    # ------------------------------------------------------------------ one cycle, pure part
    def decide(
        self, force_n: float | None, now: float, frame_t: float, position_ticks: int | None,
        pull: "PullPlan | None" = None,
    ) -> tuple[Decision, float]:
        """Decide this cycle without performing any I/O. Returns (decision, step in metres).

        Separated from `step()` so the whole decision path, including the travel bookkeeping
        and the yield-direction check, is testable without a robot.

        `pull`, when given, replaces the yield law for this cycle with `Law.PULL`: the arm is
        the one moving, toward `PullPlan.target_yielded_m`. The gate still runs and a latched
        RELEASE still stops everything, but HOLD no longer means "stay put", because in a
        device-driven phase it is the position target and the cap that end the motion, not the
        gate's permission to give way.
        """
        if position_ticks is not None:
            if self.state.start_ticks is None:
                self.state.start_ticks = position_ticks
            self.state.travel_m = (
                (position_ticks - self.state.start_ticks) / self.axis.ticks_per_metre
            )
            self.state.last_ticks = position_ticks

        decision = self.gate.check(force_n, age_s=now - frame_t, travel_m=self.state.travel_m)

        if pull is not None:
            # The yield-direction check is deliberately skipped (see `PullPlan`), so clear its
            # window: a pull that ran for a while must not leave a stale start time behind for
            # whatever yields next to measure itself against.
            self._yield_started_at = None
            if decision.action is Action.RELEASE or force_n is None:
                # No reading, no pull. A device-driven phase that keeps moving blind is moving
                # a finger with no idea how hard, which is the one thing it must never do;
                # this is the loop's half of the same rule `session.reps.RepMachine._deadman`
                # enforces at the phase level.
                return decision, 0.0
            return decision, self._pull_step(force_n, now, pull)

        if decision.action is not Action.YIELD or force_n is None:
            self._yield_started_at = None
            if (self.recover_gain is not None and decision.action is Action.HOLD
                    and force_n is not None):
                return decision, self._recover_step(force_n, now)
            return decision, 0.0

        if self.yield_check_window_s is not None and not self._check_yield_direction(force_n, now):
            self.gate.latched = Decision(
                Action.RELEASE,
                "yielding did not reduce the force; tether direction may be wrong",
                force_n,
            )
            return self.gate.latched, 0.0

        dt = 0.0 if self.state.last_t is None else now - self.state.last_t
        # The same point the gate started yielding from (see SafetyLimits.anticipation_n), not
        # always the cap: feeding the law the plain cap here would make "overshoot" read zero
        # right as YIELD begins whenever anticipation_n > 0, so the arm would be commanded to
        # move at zero velocity at the exact moment the gate says to move, defeating the point.
        velocity = yield_velocity(force_n, self.limits.yield_from_n, self.params)
        step_m = step_from_velocity(velocity, dt, self.params)
        return decision, step_m

    def _pull_step(self, force_n: float, now: float, plan: "PullPlan") -> float:
        """One cycle of `Law.PULL`: toward the target, at the plan's speed, or 0.0.

        Sign convention is `decide()`'s throughout: positive pays line out and lowers tension,
        negative takes line up and raises it. So the step's direction falls out of which side
        of the target the arm is on, and arriving produces a zero step without needing a
        tolerance, because the move is clamped to what is actually remaining.

        **The cap stops only the tension-raising half of the stroke.** That asymmetry is the
        whole correctness of running a pull out and back, and getting it wrong is not subtle:
        an out-stroke ends either at its target or at the cap, and when it ends at the cap the
        return stroke begins with the force sitting *at* the cap. A symmetric "stop whenever
        force >= cap" would therefore refuse to move at the exact moment the arm is needed to
        let the finger go, leaving it stretched at the cap until a watchdog fired. A step that
        reduces tension cannot make an over-tense finger worse, so it is never blocked here;
        `SafetyLimits.f_max_n` (the hard ceiling) and the travel limit still apply through the
        gate and are what stop this motion in a genuine fault.

        **The stop is the cap itself, not `limits.yield_from_n`**, and that is deliberate
        rather than an oversight. The anticipation margin exists so *yielding* starts before
        the cap, because the yield law's own velocity is zero exactly at the threshold and the
        arm needs to already be moving by the time the reading arrives there. A pull has no
        such problem: it travels at a set speed and has it from the first cycle.

        Stopping early here was tried and is a deadlock, found by
        `tests/control/test_cpm_closed_loop.py` and invisible to every unit test either side
        of it. `session.reps.RepMachine` decides a stretch has ended at the force reaching
        `Block.cap_n`. If the motion stops at `yield_from_n` instead, there is a band exactly
        `anticipation_n` wide where the arm has stopped and the machine is still waiting for a
        cap that can no longer be reached, because the only thing that was raising the force
        has halted. On the bench that is a stretch that completes correctly and is then
        abandoned on a phase timeout. One threshold, shared by the motion and the phase that
        ends on it, is what makes the two agree.

        The hard ceiling (`SafetyLimits.f_max_n`) is unaffected and still latches RELEASE
        through the gate, so the cap is where this *stops* and not where safety begins.
        """
        remaining = plan.target_yielded_m - self.state.travel_m * self.axis.sign
        if remaining == 0.0:
            return 0.0
        dt = 0.0 if self.state.last_t is None else now - self.state.last_t
        advance = step_from_velocity(plan.speed_m_s, dt, self.params,
                                     max_step_m=plan.max_step_m)
        step = min(advance, abs(remaining)) * (1.0 if remaining > 0 else -1.0)
        ceiling = (self.limits.f_threshold_n if plan.stop_above_n is None
                   else plan.stop_above_n)
        if step < 0 and force_n >= ceiling:
            return 0.0
        return step

    def _recover_step(self, force_n: float, now: float) -> float:
        """Negative step (take slack up) for one HOLD cycle, or 0.0 when there is nothing to
        recover.

        The sign convention is `decide()`'s: positive steps give way, so recovery is negative.
        Two independent bounds apply, and both have to hold rather than either one alone: the
        magnitude is clamped to how far the arm has actually yielded this session (`travel`
        converted into the yield sense by `axis.sign`, never commanding past the start), and
        `recover_velocity` itself stops once `force_n` reaches `self.recover_floor_n` (never
        chasing the force below the tether's normal resting tension). The tick boundary alone
        is not enough on a long coordinated pull where the single-axis travel bookkeeping can
        drift from the real geometry (see `recover_velocity`'s docstring for the bench
        failure this closes); the force floor is what stops it even when that bookkeeping is
        briefly wrong.

        Once force reaches the floor, `self.recover_retrace_m_s`, not `recover_velocity`,
        decides whether to keep going: without it, the arm simply stops wherever it is, which
        on the bench meant it never got back to the session's start at all. If a finger
        relaxes faster than `v_max_m_s` can follow (routine, not a fault: muscle relaxation
        has no reason to match this arm's speed limit), the tether goes slack and `force_n`
        hits the floor while `yielded_m` is still large - the force signal that drives
        `recover_velocity` has nothing left to say at that point, having already reached the
        value it exists to protect, but the arm is nowhere near home and nothing else was
        telling it to keep moving. This is the bounded, position-only stand-in for the
        `Law.PULL`/`Target.REP_START` motion `session/modes.py` specifies and `control/`
        never implemented (see `scripts/run_session.py`'s module docstring): once there is no
        force reason left to retreat, retrace the remaining `yielded_m` at a slow, fixed
        speed instead of stopping short of it. `0.0` (the default) disables this and keeps
        the original behaviour: the arm stops wherever `recover_velocity` leaves it.
        """
        yielded_m = self.state.travel_m * self.axis.sign
        if yielded_m <= 0:
            return 0.0
        dt = 0.0 if self.state.last_t is None else now - self.state.last_t
        velocity = recover_velocity(force_n, self.limits.yield_from_n, self.params,
                                    self.recover_gain, self.recover_floor_n)
        if velocity <= 0.0 and self.recover_retrace_m_s > 0.0:
            velocity = self.recover_retrace_m_s
        return -min(step_from_velocity(velocity, dt, self.params), yielded_m)

    def _check_yield_direction(self, force_n: float, now: float) -> bool:
        """While yielding, the force must actually come down.

        If the arm has been giving way for `self.yield_check_window_s` and the force is no
        lower than when it started, the assumed pull direction is wrong and moving further
        makes it worse: the positive-feedback failure in docs/admittance-control.md. One check
        catches a flipped sign, a hand that moved, and a detached hook alike.

        It also fires on a true positive that is not that failure: a person flexing faster
        than `AdmittanceParams.v_max_m_s` can give way, where tension keeps rising despite
        correct direction and real motion simply because the finger is closing the gap faster
        than the arm is opening it. `step()` only calls this when `self.yield_check_window_s`
        is not `None`, which is the caller's way of deciding whether that false positive is
        worth tolerating on this hardware.
        """
        if self._yield_started_at is None:
            self._yield_started_at = (now, force_n)
            return True
        started_t, started_force = self._yield_started_at
        if now - started_t < self.yield_check_window_s:
            return True
        return force_n < started_force

    # ------------------------------------------------------------------ one cycle, with I/O
    async def step(self, client, bus_serial: str, force_n: float | None, frame_t: float,
                   position_ticks: int | None, now: float | None = None,
                   pose_ticks: dict[int, float] | None = None,
                   pull: "PullPlan | None" = None) -> Decision:
        """Run one cycle and command the arm. Returns the decision taken.

        `pose_ticks` is every joint's current reading, and is required when this loop was built
        with a `CoordinatedAxis`: a straight pull is solved from the whole pose, not from one
        joint's position. It is ignored for a single-axis loop.

        `pull` runs this cycle under `Law.PULL` instead of the yield law: see `PullPlan` and
        `decide()`. A caller passes it on the cycles a device-driven phase is in progress and
        omits it otherwise, so a mode whose beats alternate drivers alternates between the two
        laws with no state kept here about which is in force.
        """
        from lib.station import set_position

        now = time.monotonic() if now is None else now
        decision, step_m = self.decide(force_n, now, frame_t, position_ticks, pull=pull)

        commanded = position_ticks
        # Three motions and only three: giving way on YIELD (positive step), taking bounded
        # slack back up on HOLD when recovery is enabled (negative step, `_recover_step`), and
        # a device-driven pull in either direction (`_pull_step`). RELEASE never commands
        # anything from here, and a HOLD with neither recovery nor a pull is a zero step, so
        # the behaviour without either is untouched.
        if pull is not None:
            # Deliberately not gated on the action: HOLD is the normal state while the arm
            # pulls a relaxed finger, and `Decision.may_move` is true only on YIELD, so
            # requiring either would mean a pull that sits still until the force is already at
            # the cap and only then starts moving. `decide()` has already returned 0.0 for a
            # RELEASE or a missing reading, which is what actually stops this motion.
            wants_motion = step_m != 0
        else:
            wants_motion = ((decision.action is Action.YIELD and step_m > 0)
                            or (decision.action is Action.HOLD and step_m < 0))
        if wants_motion and position_ticks is not None:
            if self.coordinated is not None:
                commanded = await self._command_coordinated(
                    client, bus_serial, step_m, pose_ticks, position_ticks
                )
            else:
                commanded = position_ticks + self.axis.ticks_for(step_m)
                await set_position(client, bus_serial, self.axis.motor_id, commanded)

        self.state.last_t = now
        self.state.records.append(
            CycleRecord(now, force_n, decision.action.value, decision.reason, step_m,
                        self.state.travel_m, commanded)
        )
        return decision

    async def _command_coordinated(self, client, bus_serial: str, step_m: float,
                                   pose_ticks: dict[int, float] | None,
                                   position_ticks: int) -> int:
        """Solve the straight-line step and command every joint it uses.

        Returns the tracked motor's new tick value, so travel bookkeeping and the yield
        direction check keep working unchanged: the loop still measures its own progress
        against one joint even though several are moving.

        Commands every joint in one `sync_write` per register, speed scaled per joint by
        `lib.station.scaled_speeds`, exactly as `scripts/jog_coordinated.py` does and for the
        identical reason: three separate position commands, even dispatched together, can
        land a few milliseconds apart and arrive at different times if sent at the same raw
        speed regardless of how far each has to travel, and either one bends the tip off the
        tether's line mid-cycle rather than only between cycles.
        """
        from lib.station import RAM_ACC, RAM_GOAL_POSITION, RAM_GOAL_SPEED, scaled_speeds, sync_write

        if not pose_ticks:
            # No pose, no inverse kinematics. Commanding the tracked joint alone would pull the
            # finger along an arc instead of the tether's line, so refuse to move at all.
            return position_ticks

        if self._chain is None:
            from lib.kinematics import load_chain

            self._chain = load_chain()

        if not self._accel_set:
            await sync_write(client, bus_serial, RAM_ACC, {
                m: self.coordinated.accel.to_bytes(1, byteorder="little")
                for m in self.coordinated.motors
            })
            self._accel_set = True

        deltas = self.coordinated.ticks_for(step_m, pose_ticks, self._chain,
                                            residual=self._tick_residual)
        speeds = scaled_speeds(deltas, self.coordinated.speed)
        await sync_write(client, bus_serial, RAM_GOAL_SPEED, {
            m: speeds[m].to_bytes(2, byteorder="little") for m in self.coordinated.motors
        })
        await sync_write(client, bus_serial, RAM_GOAL_POSITION, {
            m: int(round(pose_ticks[m] + deltas.get(m, 0.0))).to_bytes(2, byteorder="little")
            for m in self.coordinated.motors
        })
        return position_ticks + deltas.get(self.axis.motor_id, 0)

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
