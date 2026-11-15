"""Pulling along a straight line with several joints, instead of arcing with one.

The reason this exists is measured, not theoretical: over the 72 to 88 mm one full-range
repetition needs, a single revolute joint leaves the tether's axis by more than 20 mm
(`tests/lib/test_kinematics.py`). The tether does not bend, so that is side-load on the finger
or slack in the line.
"""
import asyncio

import numpy as np
import pytest

from control.admittance import AdmittanceParams
from control.loop import AxisCalibration, ControlLoop, CoordinatedAxis
from control.safety import Action, SafetyLimits
from lib.units import grams_force_to_newtons

POSE = {1: 1976, 2: 1719, 3: 2526, 4: 2755, 5: 1907, 6: 1270, 7: 1282}
PULL_MOTORS = [2, 3, 4]
DIRECTION = (-0.09, -0.72, -0.68)


def axis() -> AxisCalibration:
    return AxisCalibration(motor_id=4, ticks_per_metre=4300.0, sign=-1)


def coordinated(**kw) -> CoordinatedAxis:
    base = dict(motors=PULL_MOTORS, direction=DIRECTION)
    base.update(kw)
    return CoordinatedAxis(**base)


class FakeStation:
    """Records set_position calls instead of moving an arm."""

    def __init__(self):
        self.commands: list[tuple[int, int]] = []

    async def set_position(self, client, bus_serial, motor_id, position):
        self.commands.append((motor_id, position))


@pytest.fixture
def fake_station(monkeypatch):
    import lib.station

    fake = FakeStation()
    monkeypatch.setattr(lib.station, "set_position", fake.set_position)
    return fake


# ---------------------------------------------------------------- the rank requirement
def test_two_joints_are_rejected():
    """Two revolute joints span a plane, so the return stroke misses. Caught at construction
    rather than discovered mid-session on a finger."""
    with pytest.raises(ValueError, match="three independent joints"):
        CoordinatedAxis(motors=[3, 4], direction=DIRECTION)


def test_a_repeated_motor_is_rejected():
    with pytest.raises(ValueError, match="repeated motor"):
        CoordinatedAxis(motors=[3, 4, 4], direction=DIRECTION)


def test_a_zero_direction_is_rejected():
    with pytest.raises(ValueError, match="zero vector"):
        CoordinatedAxis(motors=PULL_MOTORS, direction=(0.0, 0.0, 0.0))


def test_the_direction_is_normalised():
    a = CoordinatedAxis(motors=PULL_MOTORS, direction=(0.0, -10.0, 0.0))
    assert np.allclose(a.unit_direction(), [0.0, -1.0, 0.0])


# ---------------------------------------------------------------- solving a step
def test_yielding_moves_along_the_tether_not_against_it():
    """Positive `step_m` means give way, so the gripper must travel opposite to the direction
    that increases tension. Getting this backwards tightens the spring on a finger."""
    from lib.kinematics import load_chain

    chain = load_chain()
    a = coordinated()
    deltas = a.ticks_for(0.002, POSE, chain)

    moved = dict(POSE)
    for m, d in deltas.items():
        moved[m] += d
    travel = chain.tip_position(moved) - chain.tip_position(POSE)
    assert float(np.dot(travel, a.unit_direction())) < 0


def test_a_step_is_shared_across_the_joints():
    from lib.kinematics import load_chain

    deltas = coordinated().ticks_for(0.002, POSE, load_chain())
    assert set(deltas) == set(PULL_MOTORS)
    assert sum(1 for v in deltas.values() if v != 0) >= 2


def test_no_joint_exceeds_the_per_cycle_tick_limit():
    """The second bound, in joint space. A badly conditioned pose can ask for a large joint
    motion to make a small tip motion, and the metre-space limit alone would not catch it."""
    from lib.kinematics import load_chain

    a = coordinated(max_ticks_per_cycle=10)
    deltas = a.ticks_for(0.050, POSE, load_chain())      # deliberately a huge step
    assert max(abs(v) for v in deltas.values()) <= 10


def test_scaling_down_keeps_the_direction():
    """When the limit bites, the step must get shorter, not crooked."""
    from lib.kinematics import load_chain

    chain = load_chain()
    full = coordinated(max_ticks_per_cycle=10_000).ticks_for(0.010, POSE, chain)
    clipped = coordinated(max_ticks_per_cycle=10).ticks_for(0.010, POSE, chain)

    def tip_delta(deltas):
        moved = dict(POSE)
        for m, d in deltas.items():
            moved[m] += d
        v = chain.tip_position(moved) - chain.tip_position(POSE)
        return v / np.linalg.norm(v)

    assert float(np.dot(tip_delta(full), tip_delta(clipped))) > 0.99


# ---------------------------------------------------------------- through the control loop
def loop(**kw) -> ControlLoop:
    limits = SafetyLimits(f_threshold_n=grams_force_to_newtons(200),
                          f_max_n=grams_force_to_newtons(300), max_travel_m=0.12)
    params = AdmittanceParams(damping_n_s_per_m=15.0, v_max_m_s=0.02,
                              deadband_n=0.05, max_step_m=0.002)
    return ControlLoop(limits=limits, params=params, axis=axis(), **kw)


def test_a_single_axis_loop_still_moves_one_motor(fake_station):
    async def run():
        """Backwards compatible: without a CoordinatedAxis nothing about the old path changes."""
        lp = loop()
        await lp.step(None, "bus", grams_force_to_newtons(250), 0.0, 2755, now=0.0)
        await lp.step(None, "bus", grams_force_to_newtons(250), 0.1, 2755, now=0.1)
        assert fake_station.commands
        assert {m for m, _ in fake_station.commands} == {4}

    asyncio.run(run())


def test_a_coordinated_loop_commands_every_joint(fake_station):
    async def run():
        lp = loop(coordinated=coordinated())
        await lp.step(None, "bus", grams_force_to_newtons(250), 0.0, 2755, now=0.0, pose_ticks=POSE)
        await lp.step(None, "bus", grams_force_to_newtons(250), 0.1, 2755, now=0.1, pose_ticks=POSE)
        moved = {m for m, _ in fake_station.commands}
        assert len(moved) >= 2, f"only moved {moved}"
        assert moved <= set(PULL_MOTORS)

    asyncio.run(run())


def test_without_a_pose_it_refuses_to_move(fake_station):
    async def run():
        """No pose means no inverse kinematics. Commanding the tracked joint alone would pull the
        finger along an arc instead of the tether's line, so the safe answer is to stay put."""
        lp = loop(coordinated=coordinated())
        await lp.step(None, "bus", grams_force_to_newtons(250), 0.0, 2755, now=0.0, pose_ticks=None)
        await lp.step(None, "bus", grams_force_to_newtons(250), 0.1, 2755, now=0.1, pose_ticks=None)
        assert fake_station.commands == []

    asyncio.run(run())


def test_the_gate_still_blocks_a_coordinated_pull(fake_station):
    async def run():
        """Coordination changes how the arm moves, never whether it is allowed to. A force over
        the hard ceiling must stop every joint, not just the tracked one."""
        lp = loop(coordinated=coordinated())
        decision = await lp.step(None, "bus", grams_force_to_newtons(400), 0.0, 2755,
                                 now=0.0, pose_ticks=POSE)
        assert decision.action is Action.RELEASE
        assert fake_station.commands == []

    asyncio.run(run())


def test_a_dropout_still_stops_a_coordinated_pull(fake_station):
    async def run():
        lp = loop(coordinated=coordinated())
        for i in range(6):
            await lp.step(None, "bus", None, i * 0.1, 2755, now=i * 0.1, pose_ticks=POSE)
        assert fake_station.commands == []

    asyncio.run(run())
