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
    """Records set_position and sync_write calls instead of moving an arm.

    `commands` holds every position actually commanded, from either path, as `(motor_id,
    position)` pairs - this is what the pre-existing tests in this file assert against, and it
    stays meaningful whichever register write produced it. `speed_writes` and `accel_writes`
    are the sync_write calls to RAM_GOAL_SPEED and RAM_ACC specifically, decoded back to plain
    ints, for the tests that check the coordinated path is scaling speed per joint rather than
    writing one motor at a time (`lib.station.set_position` is never called on that path at
    all any more, which is the point: see `ControlLoop._command_coordinated`).
    """

    def __init__(self):
        self.commands: list[tuple[int, int]] = []
        self.speed_writes: list[dict[int, int]] = []
        self.accel_writes: list[dict[int, int]] = []

    async def set_position(self, client, bus_serial, motor_id, position):
        self.commands.append((motor_id, position))

    async def sync_write(self, client, bus_serial, address, values):
        from lib.station import RAM_ACC, RAM_GOAL_POSITION, RAM_GOAL_SPEED

        decoded = {m: int.from_bytes(v, byteorder="little") for m, v in values.items()}
        if address == RAM_GOAL_POSITION:
            self.commands.extend(decoded.items())
        elif address == RAM_GOAL_SPEED:
            self.speed_writes.append(decoded)
        elif address == RAM_ACC:
            self.accel_writes.append(decoded)


@pytest.fixture
def fake_station(monkeypatch):
    import lib.station

    fake = FakeStation()
    monkeypatch.setattr(lib.station, "set_position", fake.set_position)
    monkeypatch.setattr(lib.station, "sync_write", fake.sync_write)
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


# ---------------------------------------------------------------- choosing joints per pose
def test_for_pose_picks_the_joints_itself():
    """What makes this usable without knowing the pose in advance: the caller supplies the
    pose and the pull, not a list of motor ids."""
    from lib.kinematics import load_chain

    chain = load_chain()
    direction = chain.tip_velocity_direction(4, POSE)
    a = CoordinatedAxis.for_pose(POSE, direction, chain)
    assert len(a.motors) == 3
    rank, _ = chain.pull_quality(a.motors, POSE, direction)
    assert rank == 3


def test_for_pose_refuses_a_pull_this_pose_cannot_follow():
    """Refusing is the point: the alternative is an arm that tracks the first few millimetres
    and then wanders off the tether's line with a finger attached."""
    from lib.kinematics import load_chain

    chain = load_chain()
    centred = {m: 2048 for m in range(1, 8)}
    direction = chain.tip_velocity_direction(4, centred)
    sideways = np.cross(direction, [0.0, 0.0, 1.0])
    sideways = sideways / np.linalg.norm(sideways)

    with pytest.raises(ValueError, match="cannot pull straight"):
        CoordinatedAxis.for_pose(centred, sideways, chain)


def test_for_pose_carries_the_direction_through():
    from lib.kinematics import load_chain

    chain = load_chain()
    direction = chain.tip_velocity_direction(4, POSE)
    a = CoordinatedAxis.for_pose(POSE, direction, chain)
    assert np.allclose(a.unit_direction(), direction / np.linalg.norm(direction), atol=1e-9)


# ---------------------------------------------------------------- which stroke gets checked
def test_for_pose_checks_the_stroke_it_is_told_to_travel():
    """Feasibility is not symmetric, and a yielding controller only ever travels one way.

    At the pose the arm stood in on 2026-11-15, joints [1, 2, 4] give way along the tether for
    a full 90 mm at 54 mm per radian, while travelling the same line in the opposite sense from
    the same pose collapses to 1 mm per radian after about 30 mm. Checking the stroke that
    never happens would refuse a pull that is entirely sound, so `for_pose` takes the travel
    direction explicitly rather than assuming it equals `direction`.
    """
    from lib.kinematics import load_chain, tether_tension_direction

    chain = load_chain()
    pose = {1: 2049, 2: 1469, 3: 2468, 4: 2772, 5: 1879, 6: 1525, 7: 3275}
    tension = tether_tension_direction(chain, pose)

    with pytest.raises(ValueError, match="cannot pull straight"):
        CoordinatedAxis.for_pose(pose, tension, chain, distance_m=0.090)

    a = CoordinatedAxis.for_pose(pose, tension, chain, distance_m=0.090,
                                 travel_direction=-tension)
    assert np.allclose(a.unit_direction(), tension / np.linalg.norm(tension))


def test_for_pose_still_defaults_to_checking_the_direction_given():
    from lib.kinematics import load_chain

    chain = load_chain()
    direction = chain.tip_velocity_direction(4, POSE)
    assert (CoordinatedAxis.for_pose(POSE, direction, chain).motors
            == CoordinatedAxis.for_pose(POSE, direction, chain,
                                        travel_direction=direction).motors)


# ---------------------------------------------------------------- sub-tick accumulation
def straightness(step_m: float, distance_m: float, use_residual: bool) -> float:
    """Worst deviation from the tether's line, pulling `distance_m` in steps of `step_m`."""
    from lib.kinematics import load_chain, tether_tension_direction

    chain = load_chain()
    pose = {1: 2045, 2: 1468, 3: 2458, 4: 2775, 5: 1877, 6: 1524, 7: 3274}
    tension = tether_tension_direction(chain, pose)
    a = CoordinatedAxis.for_pose(pose, tension, chain, distance_m=distance_m,
                                 travel_direction=-tension)

    start = chain.tip_position(pose)
    residual: dict[int, float] = {} if use_residual else None
    worst = 0.0
    for _ in range(4000):
        tip = chain.tip_position(pose)
        travel = tip - start
        along = float(np.dot(travel, -tension))
        worst = max(worst, float(np.linalg.norm(travel - along * -tension)))
        if along >= distance_m:
            break
        deltas = a.ticks_for(step_m, pose, chain, residual=residual)
        if not any(deltas.values()):
            break
        for motor_id, delta in deltas.items():
            pose[motor_id] += delta
    return worst


def test_small_steps_wander_off_the_line_without_the_residual():
    """The failure this guards against, stated as the measurement that found it. A straight
    line is a ratio between joints and the servo takes whole ticks, so rounding each joint
    independently every cycle discards a fixed share of the motion in a direction that does
    not average out."""
    assert straightness(0.0002, 0.034, use_residual=False) > 0.005


def test_accumulating_the_remainder_keeps_small_steps_straight():
    """Same pull, same step size, with the remainder carried: back inside the 2 mm the rest of
    this module treats as straight."""
    assert straightness(0.0002, 0.034, use_residual=True) < 0.002


def test_the_residual_does_not_spoil_steps_that_were_already_fine():
    """A 1 mm step rounds well on its own; carrying the remainder must not make it worse."""
    assert straightness(0.001, 0.034, use_residual=True) <= 0.002


def test_the_remainder_stays_under_one_tick_per_joint():
    """It is a rounding remainder, not a backlog. If it could grow it would eventually hand
    the arm a jump instead of a step."""
    from lib.kinematics import load_chain

    chain = load_chain()
    a = coordinated()
    residual: dict[int, float] = {}
    pose = dict(POSE)
    for _ in range(200):
        for motor_id, delta in a.ticks_for(0.0002, pose, chain, residual=residual).items():
            pose[motor_id] += delta
        assert all(abs(v) <= 0.5 + 1e-9 for v in residual.values()), residual


def test_the_per_cycle_clamp_drops_the_remainder_rather_than_banking_it():
    """The clamp is a safety bound, not a rounding error. Carrying the clamped-away motion
    forward would return it as a burst on a later cycle, which is what the bound prevents."""
    from lib.kinematics import load_chain

    a = coordinated(max_ticks_per_cycle=5)
    residual = {m: 0.4 for m in PULL_MOTORS}
    a.ticks_for(0.050, POSE, load_chain(), residual=residual)
    assert residual == {}


def test_the_loop_carries_the_residual_between_cycles(fake_station):
    async def run():
        lp = loop(coordinated=coordinated())
        assert lp._tick_residual == {}
        for i in range(8):
            await lp.step(None, "bus", grams_force_to_newtons(250), i * 0.1, 2755,
                          now=i * 0.1, pose_ticks=POSE)
        assert lp._tick_residual, "the loop threw the sub-tick remainder away each cycle"

    asyncio.run(run())


# ---------------------------------------------------------------- real joint-range checking
def test_for_pose_refuses_when_tick_ranges_rules_out_a_joint():
    """The bug this closes: `choose_pull_motors`/`can_pull_straight` had no idea a joint's
    real travel is narrower than the URDF believes, so two joints could quietly compensate
    for a third pinned at its stop while every geometric check kept reporting a clean pull."""
    from lib.kinematics import load_chain, tether_tension_direction

    chain = load_chain()
    pose = {1: 2049, 2: 1469, 3: 2468, 4: 2772, 5: 1879, 6: 1525, 7: 3275}
    tension = tether_tension_direction(chain, pose)
    motors = chain.choose_pull_motors(pose, -tension)
    narrow = {m: (0, 4096) for m in motors}
    narrow[motors[0]] = (pose[motors[0]] - 2, pose[motors[0]] + 2)

    with pytest.raises(ValueError, match="cannot pull straight"):
        CoordinatedAxis.for_pose(pose, tension, chain, distance_m=0.040,
                                 travel_direction=-tension, tick_ranges=narrow)


def test_for_pose_still_works_with_tick_ranges_omitted():
    """Backwards compatible: every earlier test in this file calls for_pose without
    tick_ranges and must keep passing unchanged."""
    from lib.kinematics import load_chain, tether_tension_direction

    chain = load_chain()
    pose = {1: 2049, 2: 1469, 3: 2468, 4: 2772, 5: 1879, 6: 1525, 7: 3275}
    tension = tether_tension_direction(chain, pose)
    a = CoordinatedAxis.for_pose(pose, tension, chain, distance_m=0.040,
                                 travel_direction=-tension)
    assert len(a.motors) == 3


def test_for_pose_accepts_a_wide_enough_tick_range():
    from lib.kinematics import load_chain, tether_tension_direction

    chain = load_chain()
    pose = {1: 2049, 2: 1469, 3: 2468, 4: 2772, 5: 1879, 6: 1525, 7: 3275}
    tension = tether_tension_direction(chain, pose)
    motors = chain.choose_pull_motors(pose, -tension)
    wide = {m: (0, 4096) for m in motors}
    a = CoordinatedAxis.for_pose(pose, tension, chain, distance_m=0.040,
                                 travel_direction=-tension, tick_ranges=wide)
    assert len(a.motors) == 3


# ---------------------------------------------------------------- speed/accel through the loop
def test_the_coordinated_loop_scales_speed_per_joint(fake_station):
    """The whole point of porting this from scripts/arm/jog_coordinated.py: the real controller
    must get the same per-joint speed matching, not just the diagnostic script. Without it,
    whichever joint travels furthest this cycle is still moving once the others have stopped,
    which is the wobble `scaled_speeds` exists to remove."""
    async def run():
        # Two calls: the first only primes last_t (dt is 0, so step_m is 0 and nothing is
        # commanded yet); the second is where an actual step, and the writes to check, happen.
        lp = loop(coordinated=coordinated())
        await lp.step(None, "bus", grams_force_to_newtons(250), 0.0, 2755, now=0.0, pose_ticks=POSE)
        await lp.step(None, "bus", grams_force_to_newtons(250), 0.1, 2755, now=0.1, pose_ticks=POSE)
        assert fake_station.speed_writes, "no RAM_GOAL_SPEED written on a coordinated step"
        speeds = fake_station.speed_writes[-1]
        assert set(speeds) == set(PULL_MOTORS)
        assert len(set(speeds.values())) > 1, (
            f"every joint got the same speed {speeds}, so nothing is actually being scaled"
        )

    asyncio.run(run())


def test_the_coordinated_loop_sets_accel_once_not_every_cycle(fake_station):
    async def run():
        lp = loop(coordinated=coordinated())
        for i in range(5):
            await lp.step(None, "bus", grams_force_to_newtons(250), i * 0.1, 2755,
                          now=i * 0.1, pose_ticks=POSE)
        assert len(fake_station.accel_writes) == 1
        assert fake_station.accel_writes[0] == {m: coordinated().accel for m in PULL_MOTORS}

    asyncio.run(run())


def test_custom_speed_and_accel_are_respected(fake_station):
    async def run():
        lp = loop(coordinated=coordinated(speed=900, accel=20))
        await lp.step(None, "bus", grams_force_to_newtons(250), 0.0, 2755, now=0.0, pose_ticks=POSE)
        await lp.step(None, "bus", grams_force_to_newtons(250), 0.1, 2755, now=0.1, pose_ticks=POSE)
        assert fake_station.accel_writes[0] == {m: 20 for m in PULL_MOTORS}
        assert max(fake_station.speed_writes[-1].values()) == 900

    asyncio.run(run())


def test_the_single_axis_loop_never_calls_sync_write(fake_station):
    """Backwards compatible in the other direction too: the single-motor path has no reason
    to touch the speed/accel registers at all, and must keep using plain set_position."""
    async def run():
        lp = loop()
        await lp.step(None, "bus", grams_force_to_newtons(250), 0.0, 2755, now=0.0)
        await lp.step(None, "bus", grams_force_to_newtons(250), 0.1, 2755, now=0.1)
        assert fake_station.speed_writes == []
        assert fake_station.accel_writes == []
        assert fake_station.commands

    asyncio.run(run())


def test_zero_speed_and_accel_are_rejected():
    with pytest.raises(ValueError, match="speed must be positive"):
        CoordinatedAxis(motors=PULL_MOTORS, direction=DIRECTION, speed=0)
    with pytest.raises(ValueError, match="accel must be positive"):
        CoordinatedAxis(motors=PULL_MOTORS, direction=DIRECTION, accel=0)


# ---------------------------------------------------------------- recovery through coordination
def test_a_coordinated_loop_takes_slack_up_when_recovery_is_enabled(fake_station):
    """The return stroke uses the same three-joint solve as giving way, just in the opposite
    sense along the tether: a negative step through `ticks_for` is already motion along
    +direction, so nothing coordinated needs to special-case recovery."""
    async def run():
        lp = loop(coordinated=coordinated())
        lp.recover_gain = 1.0
        # Yield first so there is ground to recover (tracked motor 4, sign=-1: ticks down).
        await lp.step(None, "bus", grams_force_to_newtons(250), 0.0, 2755, now=0.0, pose_ticks=POSE)
        await lp.step(None, "bus", grams_force_to_newtons(250), 0.1, 2755, now=0.1, pose_ticks=POSE)
        n_before = len(fake_station.commands)
        # Finger eases off well under the cap with ~50 ticks of yielded ground on motor 4.
        await lp.step(None, "bus", grams_force_to_newtons(50), 0.2, 2705, now=0.2, pose_ticks=POSE)
        assert len(fake_station.commands) > n_before, "no recovery command was sent"

    asyncio.run(run())
