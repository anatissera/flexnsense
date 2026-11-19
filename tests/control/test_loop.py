import asyncio

import pytest

from arm1.control.admittance import AdmittanceParams
from arm1.control.loop import AxisCalibration, ControlLoop, PullPlan
from arm1.control.safety import Action, SafetyLimits

THRESHOLD = 1.96
TICKS_PER_M = 100_000.0   # 0.01 mm per tick, the order of magnitude of a 4096-tick servo arc


def make_loop(**kw) -> ControlLoop:
    limits = SafetyLimits(
        f_threshold_n=THRESHOLD, f_max_n=3.0, max_age_s=0.1, max_dropouts=2, max_travel_m=0.05
    )
    params = AdmittanceParams(
        damping_n_s_per_m=kw.pop("damping", 100.0),
        v_max_m_s=kw.pop("v_max", 0.01),
        deadband_n=kw.pop("deadband", 0.0),
        max_step_m=kw.pop("max_step", 0.002),
    )
    axis = AxisCalibration(motor_id=4, ticks_per_metre=TICKS_PER_M, sign=kw.pop("sign", 1))
    return ControlLoop(limits, params, axis, **kw)


class FakeClient:
    """Records set_position calls instead of talking to Station."""

    def __init__(self):
        self.commands: list[tuple[str, int, int]] = []


@pytest.fixture
def patched_set_position(monkeypatch):
    """Intercept arm1.lib.station.set_position, which loop.step imports lazily."""
    calls: list[tuple[int, int]] = []

    async def fake(client, bus_serial, motor_id, position):
        calls.append((motor_id, position))

    import arm1.lib.station
    monkeypatch.setattr(arm1.lib.station, "set_position", fake)
    return calls


# --------------------------------------------------------------------------- AxisCalibration

def test_ticks_for_respects_sign():
    pos = AxisCalibration(motor_id=1, ticks_per_metre=1000.0, sign=1)
    neg = AxisCalibration(motor_id=1, ticks_per_metre=1000.0, sign=-1)
    assert pos.ticks_for(0.01) == 10
    assert neg.ticks_for(0.01) == -10


def test_invalid_axis_rejected():
    with pytest.raises(ValueError, match="ticks_per_metre"):
        AxisCalibration(motor_id=1, ticks_per_metre=0.0)
    with pytest.raises(ValueError, match="sign"):
        AxisCalibration(motor_id=1, ticks_per_metre=100.0, sign=0)


# --------------------------------------------------------------------------- decide()

def test_under_cap_holds_and_does_not_step():
    loop = make_loop()
    decision, step = loop.decide(force_n=1.0, now=1.0, frame_t=1.0, position_ticks=2000)
    assert decision.action is Action.HOLD
    assert step == 0.0


def test_first_yield_cycle_has_no_dt_so_no_step():
    """dt comes from the gap to the previous cycle; the first cycle has no previous."""
    loop = make_loop()
    decision, step = loop.decide(force_n=2.5, now=1.0, frame_t=1.0, position_ticks=2000)
    assert decision.action is Action.YIELD
    assert step == 0.0


def test_second_yield_cycle_steps():
    loop = make_loop()
    loop.decide(force_n=2.5, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.state.last_t = 1.0
    decision, step = loop.decide(force_n=2.5, now=1.033, frame_t=1.033, position_ticks=2000)
    assert decision.action is Action.YIELD
    assert step > 0.0


def test_travel_is_measured_from_the_first_position_seen():
    loop = make_loop()
    loop.decide(force_n=1.0, now=1.0, frame_t=1.0, position_ticks=2000)
    assert loop.state.travel_m == 0.0
    loop.decide(force_n=1.0, now=1.1, frame_t=1.1, position_ticks=2000 + int(0.01 * TICKS_PER_M))
    assert loop.state.travel_m == pytest.approx(0.01)


def test_travel_past_limit_releases():
    loop = make_loop()
    loop.decide(force_n=1.0, now=1.0, frame_t=1.0, position_ticks=0)
    decision, step = loop.decide(
        force_n=1.0, now=1.1, frame_t=1.1, position_ticks=int(0.06 * TICKS_PER_M)
    )
    assert decision.action is Action.RELEASE
    assert step == 0.0


def test_dropout_during_yield_stops_stepping():
    loop = make_loop()
    loop.decide(force_n=2.5, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.state.last_t = 1.0
    decision, step = loop.decide(force_n=None, now=1.033, frame_t=1.033, position_ticks=2000)
    assert decision.action is Action.HOLD
    assert step == 0.0


def test_stale_frame_releases_without_stepping():
    loop = make_loop()
    decision, step = loop.decide(force_n=2.5, now=5.0, frame_t=1.0, position_ticks=2000)
    assert decision.action is Action.RELEASE
    assert step == 0.0


# ------------------------------------------------------- monotonicity / direction guard

def test_yield_that_does_not_reduce_force_releases():
    """The positive-feedback failure: if giving way does not lower the force within the
    window, the assumed direction is wrong and moving further makes it worse."""
    loop = make_loop()
    t = 1.0
    loop.decide(force_n=2.5, now=t, frame_t=t, position_ticks=2000)
    loop.state.last_t = t
    # Same force, well past the 0.5 s window.
    decision, step = loop.decide(force_n=2.5, now=t + 0.6, frame_t=t + 0.6, position_ticks=2000)
    assert decision.action is Action.RELEASE
    assert "direction" in decision.reason
    assert step == 0.0


def test_yield_that_does_reduce_force_continues():
    loop = make_loop()
    t = 1.0
    loop.decide(force_n=2.5, now=t, frame_t=t, position_ticks=2000)
    loop.state.last_t = t
    decision, step = loop.decide(force_n=2.3, now=t + 0.6, frame_t=t + 0.6, position_ticks=2000)
    assert decision.action is Action.YIELD
    assert step > 0.0


def test_rising_force_inside_the_window_is_tolerated():
    """Briefly rising just after starting to yield is normal; only a stalled force past the
    window means the direction is wrong."""
    loop = make_loop()
    t = 1.0
    loop.decide(force_n=2.5, now=t, frame_t=t, position_ticks=2000)
    loop.state.last_t = t
    decision, _ = loop.decide(force_n=2.6, now=t + 0.1, frame_t=t + 0.1, position_ticks=2000)
    assert decision.action is Action.YIELD


def test_dropping_below_the_cap_resets_the_direction_check():
    """Coming back under the cap ends the yield episode, so the next one starts fresh rather
    than being compared against a force from minutes ago."""
    loop = make_loop()
    loop.decide(force_n=2.5, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.decide(force_n=1.0, now=1.1, frame_t=1.1, position_ticks=2000)   # back under
    assert loop._yield_started_at is None


# --------------------------------------------------------------------------- step() with I/O

def test_step_commands_the_arm_on_yield(patched_set_position):
    loop = make_loop()
    client = FakeClient()

    async def run():
        await loop.step(client, "BUS", force_n=2.5, frame_t=1.0, position_ticks=2000, now=1.0)
        await loop.step(client, "BUS", force_n=2.5, frame_t=1.033, position_ticks=2000, now=1.033)

    asyncio.run(run())
    assert len(patched_set_position) == 1          # the first cycle has no dt, so no command
    motor_id, position = patched_set_position[0]
    assert motor_id == 4
    assert position > 2000                          # sign=+1 moves the ticks up


def test_step_does_not_command_on_hold(patched_set_position):
    loop = make_loop()

    async def run():
        await loop.step(FakeClient(), "BUS", force_n=1.0, frame_t=1.0, position_ticks=2000, now=1.0)

    asyncio.run(run())
    assert patched_set_position == []


def test_step_does_not_command_on_release(patched_set_position):
    loop = make_loop()

    async def run():
        await loop.step(FakeClient(), "BUS", force_n=99.0, frame_t=1.0, position_ticks=2000, now=1.0)

    asyncio.run(run())
    assert patched_set_position == []


def test_negative_sign_commands_the_other_way(patched_set_position):
    loop = make_loop(sign=-1)

    async def run():
        await loop.step(FakeClient(), "BUS", force_n=2.5, frame_t=1.0, position_ticks=2000, now=1.0)
        await loop.step(FakeClient(), "BUS", force_n=2.5, frame_t=1.033, position_ticks=2000, now=1.033)

    asyncio.run(run())
    _, position = patched_set_position[0]
    assert position < 2000


def test_every_cycle_is_recorded(patched_set_position):
    loop = make_loop()

    async def run():
        await loop.step(FakeClient(), "BUS", force_n=1.0, frame_t=1.0, position_ticks=2000, now=1.0)
        await loop.step(FakeClient(), "BUS", force_n=None, frame_t=1.1, position_ticks=2000, now=1.1)

    asyncio.run(run())
    assert [r.action for r in loop.state.records] == ["hold", "hold"]
    assert loop.state.records[1].force_n is None


def test_release_backs_off_from_the_last_position(patched_set_position):
    loop = make_loop()

    async def run():
        await loop.step(FakeClient(), "BUS", force_n=1.0, frame_t=1.0, position_ticks=2000, now=1.0)
        await loop.release(FakeClient(), "BUS", back_off_m=0.01)

    asyncio.run(run())
    _, position = patched_set_position[-1]
    assert position == 2000 + int(0.01 * TICKS_PER_M)


def test_release_without_a_known_position_does_nothing(patched_set_position):
    loop = make_loop()
    asyncio.run(loop.release(FakeClient(), "BUS"))
    assert patched_set_position == []


# ------------------------------------------------------- the yield-direction check is optional
def test_the_check_can_be_disabled_entirely():
    """`yield_check_window_s=None` is the opt-out for hardware with no force capable of
    hurting anyone, where the check's false-positive mode (a person flexing faster than
    v_max, not an actually wrong direction) is a nuisance rather than a hazard. The hard
    ceiling is a separate mechanism and must still fire regardless."""
    loop = make_loop(yield_check_window_s=None)
    t = 1.0
    loop.decide(force_n=2.5, now=t, frame_t=t, position_ticks=2000)
    loop.state.last_t = t
    # Same, unreduced force, well past the normal 0.5 s window: with the check on this
    # releases (see test_yield_that_does_not_reduce_force_releases); with it off, it must not.
    decision, step = loop.decide(force_n=2.5, now=t + 0.6, frame_t=t + 0.6, position_ticks=2000)
    assert decision.action is Action.YIELD
    assert step >= 0.0


def test_the_ceiling_still_fires_with_the_check_disabled():
    loop = make_loop(yield_check_window_s=None)
    decision, step = loop.decide(force_n=3.5, now=1.0, frame_t=1.0, position_ticks=2000)
    assert decision.action is Action.RELEASE
    assert step == 0.0


def test_a_longer_window_tolerates_a_longer_rise_before_releasing():
    """The middle ground between the default 0.5 s and disabling it outright: a person who
    flexes briefly faster than the arm's v_max gets more grace before this concludes the
    direction is wrong, without removing the check altogether."""
    loop = make_loop(yield_check_window_s=2.0)
    t = 1.0
    loop.decide(force_n=2.5, now=t, frame_t=t, position_ticks=2000)
    loop.state.last_t = t
    # Past the default 0.5 s window but inside the extended 2.0 s one.
    decision, step = loop.decide(force_n=2.5, now=t + 1.0, frame_t=t + 1.0, position_ticks=2000)
    assert decision.action is Action.YIELD


def test_the_default_window_is_unchanged():
    """Backwards compatible: the pre-existing tests above construct `make_loop()` with no
    `yield_check_window_s` at all, and must keep behaving exactly as before this parameter
    existed."""
    loop = make_loop()
    assert loop.yield_check_window_s == 0.5


# ------------------------------------------------------- anticipation_n reaches the admittance law
def test_anticipation_gives_nonzero_velocity_exactly_at_the_cap():
    """The point of SafetyLimits.anticipation_n: without it, `yield_velocity`'s overshoot is
    zero right at the cap by construction, so the arm is commanded to start from a
    standstill at the worst possible moment. With it, the loop must feed the admittance law
    the earlier yield_from_n point, not the plain cap, so there is already overshoot - and
    therefore nonzero velocity - once the reading reaches the cap itself."""
    limits = SafetyLimits(f_threshold_n=THRESHOLD, f_max_n=3.0, anticipation_n=0.2)
    params = AdmittanceParams(damping_n_s_per_m=100.0, v_max_m_s=0.01)
    axis = AxisCalibration(motor_id=4, ticks_per_metre=TICKS_PER_M, sign=1)
    loop = ControlLoop(limits, params, axis)

    loop.decide(force_n=THRESHOLD, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.state.last_t = 1.0
    decision, step = loop.decide(force_n=THRESHOLD, now=1.033, frame_t=1.033, position_ticks=2000)
    assert decision.action is Action.YIELD
    assert step > 0.0


def test_with_no_anticipation_the_cap_itself_still_starts_at_zero_velocity():
    """The behaviour this must not change for the default case: right at the literal cap,
    with no anticipation, overshoot is genuinely zero and so is velocity."""
    loop = make_loop()
    loop.decide(force_n=THRESHOLD, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.state.last_t = 1.0
    decision, step = loop.decide(force_n=THRESHOLD, now=1.033, frame_t=1.033, position_ticks=2000)
    assert step == 0.0


# ------------------------------------------------------- recovery: following the finger back
def test_recovery_is_off_by_default():
    """The original contract, unchanged unless asked for: under the cap the arm holds, no
    matter how much ground it has yielded."""
    loop = make_loop()
    loop.decide(force_n=2.5, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.state.last_t = 1.0
    # Arm has visibly yielded (ticks moved up, sign=+1), force now well under the cap.
    decision, step = loop.decide(force_n=0.5, now=1.033, frame_t=1.033, position_ticks=3000)
    assert decision.action is Action.HOLD
    assert step == 0.0


def test_recovery_takes_slack_up_after_yielding(patched_set_position):
    loop = make_loop(recover_gain=1.0)
    client = FakeClient()

    async def run():
        # Yield some ground first (sign=+1 moves ticks up)...
        await loop.step(client, "BUS", force_n=2.5, frame_t=1.0, position_ticks=2000, now=1.0)
        await loop.step(client, "BUS", force_n=2.5, frame_t=1.033, position_ticks=2000, now=1.033)
        # ...then the finger eases off with the arm sitting 1000 ticks (10 mm) out.
        await loop.step(client, "BUS", force_n=0.5, frame_t=1.066, position_ticks=3000, now=1.066)

    asyncio.run(run())
    assert len(patched_set_position) == 2
    recover_motor, recover_target = patched_set_position[-1]
    assert recover_motor == 4
    assert recover_target < 3000, "recovery must move back toward the start, not further out"


def test_recovery_never_passes_the_starting_position():
    """The boundary that makes this safe with a finger attached: whatever the gain asks for,
    the arm never ends tighter than the operator rigged it."""
    loop = make_loop(recover_gain=1000.0, max_step=1.0)   # absurd gain and step on purpose
    loop.decide(force_n=1.0, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.state.last_t = 1.0
    # Only 5 ticks (0.05 mm) of yielded ground to take back.
    decision, step = loop.decide(force_n=0.0, now=1.033, frame_t=1.033, position_ticks=2005)
    assert decision.action is Action.HOLD
    assert step == pytest.approx(-5 / TICKS_PER_M)


def test_recovery_does_nothing_with_no_ground_yielded():
    loop = make_loop(recover_gain=1.0)
    loop.decide(force_n=1.0, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.state.last_t = 1.0
    decision, step = loop.decide(force_n=0.5, now=1.033, frame_t=1.033, position_ticks=2000)
    assert step == 0.0


def test_recovery_does_not_run_on_a_dropout():
    """No reading, no motion, recovery included: a camera dropout must never be the thing
    that tightens the tether."""
    loop = make_loop(recover_gain=1.0)
    loop.decide(force_n=1.0, now=1.0, frame_t=1.0, position_ticks=3000)
    loop.state.last_t = 1.0
    decision, step = loop.decide(force_n=None, now=1.033, frame_t=1.033, position_ticks=3000)
    assert step == 0.0


def test_recovery_respects_the_yield_sign():
    """sign=-1 means yielding moved the ticks DOWN, so recovery must ask for motion back UP,
    which in decide()'s convention is still a negative step handed to ticks_for."""
    loop = make_loop(recover_gain=1.0, sign=-1)
    loop.decide(force_n=1.0, now=1.0, frame_t=1.0, position_ticks=3000)
    loop.state.last_t = 1.0
    # Yielded ground with sign=-1: ticks went DOWN from 3000 to 2000.
    decision, step = loop.decide(force_n=0.5, now=1.033, frame_t=1.033, position_ticks=2000)
    assert step < 0
    assert loop.axis.ticks_for(step) > 0, "the commanded ticks must move back up toward start"


# ------------------------------------------------- recovery stops at a force floor, not zero
def test_recovery_stops_at_the_floor_before_exhausting_travel():
    """The bug this closes: on the bench, recovery kept commanding motion as long as there
    was tick-based travel left to retreat through, even once the measured force had already
    dropped to true slack. A floor stops the velocity law itself, so it holds even while
    there is still plenty of yielded ground the tick boundary alone would have allowed."""
    loop = make_loop(recover_gain=1.0, recover_floor_n=1.0)
    loop.decide(force_n=2.5, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.state.last_t = 1.0
    # Lots of yielded ground (position_ticks far from the 2000 start) but the force has
    # already dropped below the floor.
    decision, step = loop.decide(force_n=0.5, now=1.033, frame_t=1.033, position_ticks=10000)
    assert step == 0.0


def test_recovery_runs_normally_above_the_floor():
    loop = make_loop(recover_gain=1.0, recover_floor_n=1.0)
    loop.decide(force_n=2.5, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.state.last_t = 1.0
    decision, step = loop.decide(force_n=1.5, now=1.033, frame_t=1.033, position_ticks=3000)
    assert step < 0.0


def test_the_default_floor_is_zero_unbounded():
    """Backwards compatible: every recovery test above this one constructs make_loop() with
    no recover_floor_n at all, and must keep chasing force all the way to true slack."""
    loop = make_loop(recover_gain=1.0)
    assert loop.recover_floor_n == 0.0


# ------------------------------------------------- recovery keeps going past the force floor
def test_without_retrace_recovery_stops_dead_at_the_floor():
    """The bug this closes: once force reaches recover_floor_n, recover_velocity has nothing
    left to say (it already reached the value it exists to protect), so with retrace
    disabled (the default, 0.0) the arm simply stops - even with plenty of yielded ground
    still between it and the session's start. On the bench this meant a finger that relaxed
    faster than v_max_m_s could follow left the arm stranded tens of millimetres from home
    with nothing left to carry it the rest of the way."""
    loop = make_loop(recover_gain=1.0, recover_floor_n=1.0)
    loop.decide(force_n=2.5, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.state.last_t = 1.0
    # Force already at the floor, but 10 mm of yielded ground remains (position_ticks far
    # from the 2000 start).
    decision, step = loop.decide(force_n=1.0, now=1.033, frame_t=1.033, position_ticks=6000)
    assert step == 0.0


def test_retrace_keeps_recovering_once_the_floor_is_reached():
    loop = make_loop(recover_gain=1.0, recover_floor_n=1.0, recover_retrace_m_s=0.01)
    loop.decide(force_n=2.5, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.state.last_t = 1.0
    decision, step = loop.decide(force_n=1.0, now=1.033, frame_t=1.033, position_ticks=6000)
    assert step < 0.0


def test_retrace_is_still_bounded_by_the_actual_yielded_ground():
    """Retrace does not get its own exemption from the one rule that makes recovery safe to
    run with a finger attached: it may never be commanded past the session's starting
    position, same as the force-driven case."""
    loop = make_loop(recover_gain=1.0, recover_floor_n=1.0, recover_retrace_m_s=10.0,
                     max_step=1.0)   # an absurdly fast retrace and step cap, on purpose
    loop.decide(force_n=2.5, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.state.last_t = 1.0
    # Only 5 ticks (0.05 mm) of yielded ground left.
    decision, step = loop.decide(force_n=1.0, now=1.033, frame_t=1.033, position_ticks=2005)
    assert step == pytest.approx(-5 / TICKS_PER_M)


def test_retrace_does_not_fire_while_force_still_drives_recovery():
    """Above the floor, the force-driven velocity is already positive, so retrace's `<= 0.0`
    check must not override it with a different, fixed speed."""
    loop = make_loop(recover_gain=1.0, recover_floor_n=0.5, recover_retrace_m_s=0.01,
                     damping=10.0, v_max=0.001)
    loop.decide(force_n=2.5, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.state.last_t = 1.0
    decision, step = loop.decide(force_n=1.8, now=1.033, frame_t=1.033, position_ticks=6000)
    # Force-driven velocity is clamped to v_max_m_s=0.001, well under retrace's 0.01; the
    # step must reflect the slower, force-driven speed, not the faster retrace one.
    assert step == pytest.approx(-min(0.001 * 0.033, 0.002))


def test_retrace_does_not_run_on_a_dropout():
    """No reading, no motion, retrace included - the same rule recovery already followed."""
    loop = make_loop(recover_gain=1.0, recover_floor_n=1.0, recover_retrace_m_s=0.01)
    loop.decide(force_n=1.0, now=1.0, frame_t=1.0, position_ticks=12000)
    loop.state.last_t = 1.0
    decision, step = loop.decide(force_n=None, now=1.033, frame_t=1.033, position_ticks=12000)
    assert step == 0.0


def test_the_default_retrace_is_disabled():
    """Backwards compatible: every recovery test above this one constructs make_loop() with
    no recover_retrace_m_s at all, and must keep the original stop-at-the-floor behaviour."""
    loop = make_loop(recover_gain=1.0)
    assert loop.recover_retrace_m_s == 0.0


# ------------------------------------------------------------------------------- Law.PULL
# The arm is the source of motion in these: it moves a relaxed finger along a path decided
# before the stroke began, with the force acting as a ceiling rather than as the law. Nothing
# in `control/` implemented this before, so these are the tests that have to hold before it
# touches a finger.

def pull_loop(**kw):
    """A loop sitting at the rep's start: travel bookkeeping zeroed, a cycle of clock behind
    it so `dt` is real rather than the 0.0 of a first-ever cycle."""
    loop = make_loop(**kw)
    loop.decide(0.2, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.state.last_t = 1.0
    return loop


def test_a_pull_toward_a_negative_target_raises_tension():
    """The out-stroke: the arm stretches the finger toward extension, which is the
    tension-raising direction, so the step is negative in decide()'s convention."""
    loop = pull_loop()
    _, step = loop.decide(0.2, now=1.1, frame_t=1.1, position_ticks=2000,
                          pull=PullPlan(target_yielded_m=-0.02, speed_m_s=0.01))
    assert step < 0


def test_a_pull_toward_a_positive_target_pays_line_out():
    loop = pull_loop()
    _, step = loop.decide(0.2, now=1.1, frame_t=1.1, position_ticks=2000,
                          pull=PullPlan(target_yielded_m=0.02, speed_m_s=0.01))
    assert step > 0


def test_a_pull_travels_at_the_speed_it_was_given():
    """Not at a force-derived speed. Under the cap the force says nothing about how fast to
    move a relaxed finger, which is why speed is a setting here and not a response."""
    loop = pull_loop(max_step=1.0)
    _, step = loop.decide(0.2, now=1.5, frame_t=1.5, position_ticks=2000,
                          pull=PullPlan(target_yielded_m=-1.0, speed_m_s=0.02))
    assert step == pytest.approx(-0.02 * 0.5)   # speed x dt, unclamped


def test_a_pull_never_overshoots_its_target():
    """The clamp that makes 'back to exactly where the rep began' true rather than approximate."""
    loop = pull_loop(max_step=1.0)
    _, step = loop.decide(0.2, now=2.0, frame_t=2.0, position_ticks=2000,
                          pull=PullPlan(target_yielded_m=-0.003, speed_m_s=0.5))
    assert step == pytest.approx(-0.003)


def test_arriving_at_the_target_produces_no_motion():
    loop = pull_loop()
    _, step = loop.decide(0.2, now=1.1, frame_t=1.1, position_ticks=2000,
                          pull=PullPlan(target_yielded_m=0.0, speed_m_s=0.01))
    assert step == 0.0


def test_the_cap_stops_the_tension_raising_half_of_the_stroke():
    """Law.PULL advances while the force is under the cap and stops on reaching it."""
    loop = pull_loop()
    _, step = loop.decide(THRESHOLD + 0.1, now=1.1, frame_t=1.1, position_ticks=2000,
                          pull=PullPlan(target_yielded_m=-0.02, speed_m_s=0.01))
    assert step == 0.0


def test_the_cap_does_not_block_the_return_stroke():
    """The asymmetry that makes out-and-back work at all. An out-stroke that ended at the cap
    hands the return stroke a force already sitting at the cap; a symmetric stop would refuse
    to move exactly when the arm is needed to let the finger go, leaving it stretched there."""
    loop = pull_loop()
    # 20 mm out: travel_m is derived from the tick reading (start was 2000), never assigned.
    _, step = loop.decide(THRESHOLD + 0.1, now=1.1, frame_t=1.1, position_ticks=0,
                          pull=PullPlan(target_yielded_m=0.0, speed_m_s=0.01))
    assert step > 0, "a step that reduces tension cannot make an over-tense finger worse"


def test_a_pull_stops_at_the_cap_and_not_at_the_anticipation_margin():
    """The anticipation margin applies to yielding and deliberately not to pulling.

    Yielding starts early because the yield law's velocity is exactly zero at the threshold,
    so the arm needs to already be moving by the time a reading arrives there. A pull travels
    at a set speed and has it from the first cycle, so it has no such problem.

    Stopping early here was tried and deadlocks, which `test_cpm_closed_loop.py` demonstrates
    end to end: `arm1.session.reps.RepMachine` ends a stretch at the force reaching the cap, so a
    motion that stops `anticipation_n` below it leaves a band where the arm has halted and the
    machine waits for a cap that can no longer be reached, because the thing raising the force
    was the motion that just stopped. One threshold for both is what keeps them agreeing.
    """
    limits = SafetyLimits(f_threshold_n=THRESHOLD, f_max_n=3.0, max_age_s=0.1, max_dropouts=2,
                          max_travel_m=0.05, anticipation_n=0.2)
    params = AdmittanceParams(damping_n_s_per_m=100.0, v_max_m_s=0.01, max_step_m=0.002)
    axis = AxisCalibration(motor_id=4, ticks_per_metre=TICKS_PER_M, sign=1)
    loop = ControlLoop(limits, params, axis)
    loop.decide(0.2, now=1.0, frame_t=1.0, position_ticks=2000)
    loop.state.last_t = 1.0
    plan = PullPlan(target_yielded_m=-0.02, speed_m_s=0.01)
    # Inside the anticipation band, so a yielding arm would already be giving way here: a
    # pull keeps going, because the cap is what ends it.
    _, in_band = loop.decide(THRESHOLD - 0.1, now=1.1, frame_t=1.1, position_ticks=2000,
                             pull=plan)
    _, at_cap = loop.decide(THRESHOLD, now=1.2, frame_t=1.2, position_ticks=2000, pull=plan)
    assert in_band < 0, "the anticipation margin must not stop a pull short of the cap"
    assert at_cap == 0.0, "the cap itself must stop it"


def test_a_pull_does_not_move_without_a_reading():
    """The loop's half of the device-driven deadman. A pull that keeps going blind is moving a
    finger with no idea how hard, which is the one thing it must never do."""
    loop = pull_loop()
    _, step = loop.decide(None, now=1.1, frame_t=1.1, position_ticks=2000,
                          pull=PullPlan(target_yielded_m=-0.02, speed_m_s=0.01))
    assert step == 0.0


def test_a_pull_does_not_move_once_the_gate_has_released():
    loop = pull_loop()
    loop.decide(5.0, now=1.1, frame_t=1.1, position_ticks=2000)   # over the hard ceiling
    decision, step = loop.decide(0.2, now=1.2, frame_t=1.2, position_ticks=2000,
                                 pull=PullPlan(target_yielded_m=-0.02, speed_m_s=0.01))
    assert decision.action is Action.RELEASE
    assert step == 0.0


def test_a_pull_is_still_bounded_by_the_travel_limit():
    """The gate's own limit, unchanged: it is absolute, so it bounds a pull going the other
    way exactly as it bounds yielding."""
    loop = pull_loop()
    decision, step = loop.decide(0.2, now=1.1, frame_t=1.1, position_ticks=-8000,
                                 pull=PullPlan(target_yielded_m=-0.2, speed_m_s=0.01))
    assert decision.action is Action.RELEASE
    assert step == 0.0


def test_a_pull_respects_its_own_per_cycle_step_cap():
    """Separate from AdmittanceParams.max_step_m for the same reason recovery's is: that
    number bounds the yield step's smoothness against a flexing finger, and a pull is not
    that motion."""
    loop = pull_loop(max_step=0.0005)
    _, step = loop.decide(0.2, now=2.0, frame_t=2.0, position_ticks=2000,
                          pull=PullPlan(target_yielded_m=-0.05, speed_m_s=0.02,
                                        max_step_m=0.004))
    assert step == pytest.approx(-0.004)


def test_a_pull_falls_back_to_the_shared_step_cap_when_given_none():
    loop = pull_loop(max_step=0.0005)
    _, step = loop.decide(0.2, now=2.0, frame_t=2.0, position_ticks=2000,
                          pull=PullPlan(target_yielded_m=-0.05, speed_m_s=0.02))
    assert step == pytest.approx(-0.0005)


def test_a_pull_does_not_run_the_yield_direction_check():
    """That check assumes the arm is giving way and the force should therefore fall. While the
    arm pulls, a rising force is the expected thing, so running it would release on success."""
    loop = pull_loop(yield_check_window_s=0.1)
    plan = PullPlan(target_yielded_m=-0.05, speed_m_s=0.01)
    force = 0.2
    for i in range(10):
        t = 1.1 + i * 0.1
        force += 0.1          # exactly what the yield check releases on
        decision, step = loop.decide(force, now=t, frame_t=t, position_ticks=2000, pull=plan)
        loop.state.last_t = t
        assert decision.action is not Action.RELEASE
    assert step != 0.0


def test_a_pull_leaves_no_stale_yield_window_behind():
    """A pull skips the yield check, so it must also clear its window: a yield that starts
    after a pull has to measure itself from its own beginning, not from the pull's."""
    loop = pull_loop(yield_check_window_s=0.1)
    loop.decide(0.2, now=1.1, frame_t=1.1, position_ticks=2000,
                pull=PullPlan(target_yielded_m=-0.02, speed_m_s=0.01))
    assert loop._yield_started_at is None


def test_a_negative_sign_axis_pulls_the_same_way_in_target_terms():
    """target_yielded_m is in the signed 'how far has this arm given way' quantity, so a
    protocol does not change when the rig's tick direction does."""
    for sign in (1, -1):
        loop = pull_loop(sign=sign)
        _, step = loop.decide(0.2, now=1.1, frame_t=1.1, position_ticks=2000,
                              pull=PullPlan(target_yielded_m=-0.02, speed_m_s=0.01))
        assert step < 0, f"sign {sign:+d} should still take line up"


def test_a_pull_plan_rejects_a_nonsense_speed():
    with pytest.raises(ValueError, match="speed_m_s must be positive"):
        PullPlan(target_yielded_m=-0.02, speed_m_s=0.0)
    with pytest.raises(ValueError, match="max_step_m must be positive"):
        PullPlan(target_yielded_m=-0.02, speed_m_s=0.01, max_step_m=0.0)


# ---------------------------------------------------------- Law.PULL actually commands the arm
def test_a_pull_commands_the_arm_while_the_gate_says_hold(patched_set_position):
    """The trap PhasePlan.device_driven warns about: Decision.may_move is true only on YIELD,
    so gating a pull on it would leave the arm sitting still until the force is already at the
    cap and only then starting to move. HOLD is the normal state while the arm pulls."""
    loop = pull_loop(max_step=1.0)
    loop.state.last_t = 1.0
    decision = asyncio.run(loop.step(
        FakeClient(), "BUS", 0.2, 1.1, 2000, now=1.1,
        pull=PullPlan(target_yielded_m=-0.02, speed_m_s=0.01),
    ))
    assert decision.action is Action.HOLD
    assert patched_set_position, "a pull must move the arm under HOLD, not wait for YIELD"
    motor_id, commanded = patched_set_position[-1]
    assert motor_id == 4
    assert commanded < 2000, "the out-stroke takes line up"


def test_the_return_stroke_commands_the_arm_back_toward_the_start(patched_set_position):
    loop = pull_loop(max_step=1.0)
    loop.state.last_t = 1.0
    asyncio.run(loop.step(
        FakeClient(), "BUS", 0.2, 1.1, 0, now=1.1,   # 20 mm out from the rep's start
        pull=PullPlan(target_yielded_m=0.0, speed_m_s=0.01),
    ))
    assert patched_set_position
    assert patched_set_position[-1][1] > 0, "the return stroke pays line out"


def test_a_pull_commands_nothing_without_a_reading(patched_set_position):
    loop = pull_loop(max_step=1.0)
    loop.state.last_t = 1.0
    asyncio.run(loop.step(
        FakeClient(), "BUS", None, 1.1, 2000, now=1.1,
        pull=PullPlan(target_yielded_m=-0.02, speed_m_s=0.01),
    ))
    assert patched_set_position == []


def test_omitting_the_pull_leaves_the_yield_behaviour_untouched(patched_set_position):
    """The whole point of passing the plan per cycle: a mode whose beats alternate drivers
    alternates between the two laws, and the law that was already on the bench is unchanged."""
    loop = pull_loop(max_step=1.0)
    loop.state.last_t = 1.0
    decision = asyncio.run(loop.step(FakeClient(), "BUS", 0.2, 1.1, 2000, now=1.1))
    assert decision.action is Action.HOLD
    assert patched_set_position == [], "HOLD with no recovery and no pull is a zero step"


def test_arming_stops_at_its_own_force_instead_of_the_cap():
    """ARMED is the one pull bounded by a force rather than a position: it takes slack up to
    the tether's resting tension and must stop there, well under the cap, or the patient is
    handed a finger already at therapeutic force while still being cued."""
    loop = pull_loop(max_step=1.0)
    resting = THRESHOLD * 0.3
    plan = PullPlan(target_yielded_m=-0.05, speed_m_s=0.01, stop_above_n=resting)
    _, below = loop.decide(resting - 0.1, now=1.1, frame_t=1.1, position_ticks=2000, pull=plan)
    _, at = loop.decide(resting, now=1.2, frame_t=1.2, position_ticks=2000, pull=plan)
    assert below < 0, "under the resting tension, arming keeps taking slack up"
    assert at == 0.0, "at the resting tension, arming stops, nowhere near the cap"


def test_a_force_bounded_pull_still_cannot_pass_the_cap():
    """stop_above_n lowers the stop, and must not be a way to raise it past the session cap."""
    loop = pull_loop(max_step=1.0)
    plan = PullPlan(target_yielded_m=-0.05, speed_m_s=0.01, stop_above_n=THRESHOLD * 5)
    decision, step = loop.decide(4.0, now=1.1, frame_t=1.1, position_ticks=2000, pull=plan)
    assert decision.action is Action.RELEASE, "the hard ceiling is not negotiable from a plan"
    assert step == 0.0


def test_a_position_targeted_pull_defaults_to_the_session_cap():
    loop = pull_loop(max_step=1.0)
    plan = PullPlan(target_yielded_m=-0.05, speed_m_s=0.01)
    _, step = loop.decide(THRESHOLD + 0.1, now=1.1, frame_t=1.1, position_ticks=2000, pull=plan)
    assert step == 0.0


def test_a_nonsense_force_bound_is_rejected():
    with pytest.raises(ValueError, match="stop_above_n must be positive"):
        PullPlan(target_yielded_m=-0.02, speed_m_s=0.01, stop_above_n=0.0)
