import asyncio

import pytest

from control.admittance import AdmittanceParams
from control.loop import AxisCalibration, ControlLoop
from control.safety import Action, SafetyLimits

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
    return ControlLoop(limits, params, axis)


class FakeClient:
    """Records set_position calls instead of talking to Station."""

    def __init__(self):
        self.commands: list[tuple[str, int, int]] = []


@pytest.fixture
def patched_set_position(monkeypatch):
    """Intercept lib.station.set_position, which loop.step imports lazily."""
    calls: list[tuple[int, int]] = []

    async def fake(client, bus_serial, motor_id, position):
        calls.append((motor_id, position))

    import lib.station
    monkeypatch.setattr(lib.station, "set_position", fake)
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
