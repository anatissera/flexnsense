"""How `scripts/session/run_session.py` turns a live pose into something the control loop can drive.

The part worth testing without hardware is the resolution step: given a pose, which joints
pull, which one gets tracked, how many ticks to the metre, and which way releases. Every one
of those is a number the arm then moves on, and three of the four used to come from a bench
measurement that this arm's geometry makes impossible to take on a single joint.
"""
import asyncio
import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest

from session.tuning import load_tuning

REPO_ROOT = Path(__file__).resolve().parents[2]
URDF = REPO_ROOT / "norma-core/hardware/elrobot/simulation/elrobot_follower.urdf"

pytestmark = pytest.mark.skipif(
    not URDF.is_file(),
    reason="needs the norma-core clone (hardware/elrobot/simulation); see SETUP.md",
)

#: The pose the arm was actually standing in on 2026-11-15, read from Station.
POSE = {1: 2049, 2: 1469, 3: 2468, 4: 2772, 5: 1879, 6: 1525, 7: 3275, 8: 1301}


def load_script():
    """Import the script by path: `scripts/` is not a package, so it is not importable."""
    spec = importlib.util.spec_from_file_location(
        "run_session", REPO_ROOT / "scripts/session/run_session.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_session():
    return load_script()


def build_args(run_session, argv):
    """main()'s own parser, run without running main(). Keeps the CLI contract tested against
    the real parser rather than a copy of it that could drift."""
    import argparse
    import contextlib
    import io

    captured = {}
    real_parse = argparse.ArgumentParser.parse_args

    def spy(self, a=None, namespace=None):
        captured["args"] = real_parse(self, a, namespace)
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()), pytest.raises(SystemExit):
            run_session.main(argv)
    finally:
        argparse.ArgumentParser.parse_args = real_parse
    return captured["args"]


# ---------------------------------------------------------------- the coordinated resolution
def resolve(run_session, bus_serial=None, **overrides):
    import types

    args = types.SimpleNamespace(motor=None, tether_axis=None, max_travel_mm=40.0,
                                 coordinated_speed=700, coordinated_accel=50, level=False)
    for k, v in overrides.items():
        setattr(args, k, v)
    return run_session.resolve_coordinated(args, POSE, bus_serial)


def test_it_picks_three_joints_and_tracks_one_of_them(run_session, capsys):
    axis, coordinated = resolve(run_session)
    capsys.readouterr()

    assert len(coordinated.motors) == 3
    assert axis.motor_id in coordinated.motors, (
        "the tracked joint has to be one that moves, or its register never changes and the "
        "travel limit never trips"
    )


def test_the_direction_is_the_tether_not_an_arbitrary_axis(run_session, capsys):
    """Defaulting to the gripper's own axis is what removes the step where a person reads a
    direction off a 3D view and types it in, which is where this went wrong on the bench."""
    from lib.kinematics import load_chain, tether_tension_direction

    _, coordinated = resolve(run_session)
    capsys.readouterr()
    expected = tether_tension_direction(load_chain(URDF), POSE)
    assert np.allclose(coordinated.unit_direction(), expected, atol=1e-9)


def test_yielding_actually_slackens_the_tether(run_session, capsys):
    """The one that matters most. A step in the yield direction must move the gripper towards
    the finger, closing the gap, not away from it."""
    from lib.kinematics import load_chain

    chain = load_chain(URDF)
    axis, coordinated = resolve(run_session)
    capsys.readouterr()

    deltas = coordinated.ticks_for(0.002, POSE, chain)
    moved = dict(POSE)
    for motor_id, delta in deltas.items():
        moved[motor_id] += delta
    travel = chain.tip_position(moved) - chain.tip_position(POSE)
    assert float(np.dot(travel, coordinated.unit_direction())) < 0


def test_the_tracked_joints_sign_agrees_with_the_coordinated_step(run_session, capsys):
    """The loop commands the coordinated step but measures and releases on one joint, so the
    two have to mean the same thing by "give way"."""
    from lib.kinematics import load_chain

    chain = load_chain(URDF)
    axis, coordinated = resolve(run_session)
    capsys.readouterr()

    deltas = coordinated.ticks_for(0.002, POSE, chain)
    if deltas.get(axis.motor_id):
        assert np.sign(deltas[axis.motor_id]) == np.sign(axis.ticks_for(0.002))


def test_it_agrees_with_the_yield_sign_measured_on_the_bench(run_session, capsys):
    """motor 4's yield sign was measured as -1 at this pose by the camera run. Whenever motor
    4 is the tracked joint, the geometry must not contradict the measurement."""
    axis, _ = resolve(run_session, motor=4)
    capsys.readouterr()
    assert axis.sign == -1


def test_ticks_per_metre_is_along_the_tether(run_session, capsys):
    """Not the joint's total speed: a joint that moves mostly sideways delivers less travel
    along the pull per tick than its raw speed suggests, and counting the raw speed would make
    the loop believe it had given way further than it had."""
    from lib.kinematics import load_chain

    chain = load_chain(URDF)
    axis, coordinated = resolve(run_session)
    capsys.readouterr()

    expected = chain.ticks_per_metre_along(axis.motor_id, POSE, coordinated.unit_direction())
    assert np.isclose(axis.ticks_per_metre, expected)


# ---------------------------------------------------------------- what it refuses
def test_a_motor_outside_the_pull_is_refused(run_session, capsys):
    """Tracking a joint the pull does not move leaves travel pinned at zero, so the travel
    limit silently stops being a limit."""
    _, coordinated = resolve(run_session)
    capsys.readouterr()
    outside = next(m for m in range(1, 8) if m not in coordinated.motors)

    with pytest.raises(SystemExit, match="never move"):
        resolve(run_session, motor=outside)


def test_a_zero_tether_axis_is_refused(run_session):
    with pytest.raises(SystemExit, match="zero vector"):
        resolve(run_session, tether_axis="0,0,0")


def test_a_pull_this_pose_cannot_follow_is_refused(run_session):
    """Refusing beats discovering it partway through a pull with a finger attached."""
    from lib.kinematics import load_chain

    chain = load_chain(URDF)
    direction = chain.tip_velocity_direction(4, POSE)
    sideways = np.cross(direction, chain.tip_velocity_direction(1, POSE))
    sideways = sideways / np.linalg.norm(sideways)
    axis_arg = ",".join(str(v) for v in sideways)

    with pytest.raises(SystemExit, match="cannot follow|--max-travel-mm|wanders|gain"):
        resolve(run_session, tether_axis=axis_arg, max_travel_mm=90.0)


def test_an_explicit_axis_overrides_the_gripper_axis(run_session, capsys):
    _, coordinated = resolve(run_session, tether_axis="0,0,-1")
    capsys.readouterr()
    assert np.allclose(coordinated.unit_direction(), [0.0, 0.0, -1.0])


# ---------------------------------------------------------------- the CLI contract
def test_motor_is_not_required_when_coordinated(run_session):
    """The whole point: the pose chooses the joints, so there is nothing for a person to know
    about which motor pulls along the tether."""
    args = build_args(run_session, ["--coordinated"])
    assert args.motor is None
    assert args.coordinated is True


def test_motor_is_still_accepted_for_the_single_joint_path(run_session):
    args = build_args(run_session, ["--no-coordinated", "--motor", "4"])
    assert args.motor == 4
    assert args.coordinated is False


# ---------------------------------------------------------------- the pose has to be complete
def test_a_complete_pose_has_no_missing_joints(run_session):
    assert run_session.missing_joints(POSE) == []


def test_the_gripper_jaws_do_not_count_towards_a_complete_pose(run_session):
    """Eight motors stream on this bus. Counting readings instead of naming them is how a pose
    missing an arm joint passed for complete, which is the bug this pins."""
    with_jaws_without_motor_7 = {m: POSE[m] for m in (1, 2, 3, 4, 5, 6, 8)}
    assert len(with_jaws_without_motor_7) == 7
    assert run_session.missing_joints(with_jaws_without_motor_7) == [7]


def test_no_pose_at_all_is_reported_as_every_joint_missing(run_session):
    assert run_session.missing_joints(None) == list(run_session.ARM_MOTORS)
    assert run_session.missing_joints({}) == list(run_session.ARM_MOTORS)


def test_a_missing_joint_would_have_changed_the_pull(run_session):
    """Why a partial pose is refused outright rather than used with a warning. The chain is
    serial, so one absent angle rotates every link below it: dropping motor 2 swings the
    computed tether direction by about 60 degrees, which is a different line entirely."""
    from lib.kinematics import load_chain, tether_tension_direction

    chain = load_chain(URDF)
    full = tether_tension_direction(chain, POSE)
    partial = tether_tension_direction(chain, {m: v for m, v in POSE.items() if m != 2})
    assert float(np.dot(full, partial)) < 0.7


# ---------------------------------------------------------------- the real joint-range check
def test_resolve_coordinated_loads_the_real_tick_ranges(run_session, capsys):
    """Without bus_serial, no range file can be found, so the pull is planned without this
    check at all - which is the state every run of this function was in before tick_ranges
    existed. With the real bus serial, a joint running past its real travel mid-pull would be
    caught instead of silently bending the path."""
    axis, coordinated = resolve(run_session, motor=4,
                                bus_serial="5B8E114979" if (REPO_ROOT / "config/calibration" /
                                                            "follower-5B8E114979.json").is_file()
                                else None)
    out = capsys.readouterr().out
    assert "no calibration" not in out
    assert axis.motor_id == 4


def test_resolve_coordinated_warns_rather_than_fails_with_no_calibration_file(run_session, capsys):
    axis, coordinated = resolve(run_session, bus_serial="no-such-bus-on-this-bench")
    out = capsys.readouterr().out
    assert "no calibration" in out
    assert "WITHOUT a joint-range check" in out
    assert len(coordinated.motors) == 3


# ---------------------------------------------------------------- the main loop must yield
def test_the_arm_is_actually_commanded_when_the_force_is_over_the_cap(tmp_path, monkeypatch):
    """Reproduces, without a camera or an arm, the bug found on the bench: a session that
    printed a rising force and a nonzero 'step' column for three full reps while the arm
    never moved, ending in 'arm travelled 0.0 mm'.

    The cause was `scripts/session/run_session.py`'s main loop never yielding to the event loop
    during the session (grab frame, process, decide - all synchronous, and `ControlLoop.step`
    performs no I/O until it has a position to command against). A coroutine that never
    suspends never lets a concurrently scheduled task take a turn, so the background task
    that reads the arm's real position from Station starved, `state["ticks"]` stayed `None`
    for the whole session, and `step()`'s `position_ticks is not None` guard - correctly -
    refused to command the arm even while the printed step column tracked the admittance
    law's intended motion.

    The assertion is on the outcome that failed on the bench, not on scheduler internals:
    with the force pinned over the cap, the fake arm must receive position commands during
    the session. The end-of-run home check also reads positions and masks the starvation if
    one asserts on the position stream being consumed at all, which is why this asserts on
    `set_position` and disables the home check.
    """
    import asyncio
    from types import SimpleNamespace

    import lib.camera
    import lib.station
    import gauge_reader.cli

    commands: list[tuple[int, int]] = []

    async def fake_motor_stream():
        pos = 2000
        while True:
            await asyncio.sleep(0)
            yield "FAKEBUS", 4, SimpleNamespace(position=pos)

    monkeypatch.setattr(lib.station, "follow_motor_state", lambda client: fake_motor_stream())

    async def fake_connect(host=None):
        return object()

    async def fake_set_position(client, bus_serial, motor_id, position):
        commands.append((motor_id, position))

    class FakeSource:
        def read(self):
            return True, "frame", time.monotonic(), None

        def release(self):
            pass

    class FakePipeline:
        def process(self, frame, ts):
            # Pinned over the 100 gf default cap: every cycle wants to yield, so the only
            # thing that can stop the arm being commanded is not having a position - the
            # exact state the starved follower left the session in on the bench.
            return SimpleNamespace(found=True, newtons=1.5)

    monkeypatch.setattr(lib.station, "connect", fake_connect)
    monkeypatch.setattr(lib.station, "set_position", fake_set_position)
    monkeypatch.setattr(lib.camera, "open_source", lambda camera=None: FakeSource())
    monkeypatch.setattr(gauge_reader.cli, "_load_calibration", lambda path: {})
    monkeypatch.setattr(gauge_reader.cli, "make_pipeline", lambda calib, filt=None: FakePipeline())
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "yes")

    calib = tmp_path / "calib.json"
    calib.write_text("{}")

    run_session = load_script()
    args = build_args(run_session, [
        "--no-coordinated", "--motor", "4", "--ticks-per-metre", "4000", "--yield-sign", "-1",
        "--bus-serial", "FAKEBUS", "--station-host", "fake",
        "--calib", str(calib), "--max-seconds", "0.5", "--reps", "1",
        "--no-yield-check", "--no-home-check",
    ])
    asyncio.run(run_session.run(args))

    session_commands = [c for c in commands if c[0] == 4]
    assert session_commands, (
        "the force sat over the cap for the whole session and the arm never received a "
        "single position command: the main loop is starving the position follower again"
    )


def test_a_bad_confirmation_still_aborts_without_touching_the_fake_arm(tmp_path, monkeypatch):
    """Sanity check on the harness itself: with 'no' instead of 'yes', run() must abort before
    the camera opens or any position is ever commanded, same as it always has."""
    import asyncio

    import lib.camera
    import lib.station
    import gauge_reader.cli

    opened = []

    class FakeSource:
        def read(self):
            return True, "frame", time.monotonic(), None

        def release(self):
            opened.append("released")

    async def fake_connect(host=None):
        return object()

    monkeypatch.setattr(lib.station, "connect", fake_connect)
    monkeypatch.setattr(lib.station, "set_position", lambda *a, **kw: None)
    monkeypatch.setattr(lib.camera, "open_source", lambda camera=None: opened.append("opened") or FakeSource())
    monkeypatch.setattr(gauge_reader.cli, "_load_calibration", lambda path: {})
    monkeypatch.setattr(gauge_reader.cli, "make_pipeline", lambda calib, filt=None: None)
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "no")

    run_session = load_script()
    calib = tmp_path / "calib.json"
    calib.write_text("{}")
    args = build_args(run_session, [
        "--no-coordinated", "--motor", "4", "--ticks-per-metre", "4000", "--yield-sign", "-1",
        "--bus-serial", "FAKEBUS", "--station-host", "fake",
        "--calib", str(calib), "--max-seconds", "0.3", "--reps", "1",
    ])

    assert asyncio.run(run_session.run(args)) == 1
    assert opened == [], "the camera opened even though the prompt was declined"


# ---------------------------------------------------------------- returning home afterwards
def session_fakes(monkeypatch, final_pose: dict[int, float], force_n: float = 0.2):
    """Install the plumbing shared by every end-to-end `run()` test in this section.

    `final_pose` is a fixed pose `follow_motor_state` reports for every subscription: the
    one during the session (so the controller never commands anything, with a force held
    under the cap) and the one `run()`'s own end-of-session `read_pose` calls afterwards to
    decide whether to walk home.
    """
    from types import SimpleNamespace

    import gauge_reader.cli
    import lib.camera
    import lib.station

    commands: list[tuple[int, int]] = []

    async def fake_motor_stream():
        while True:
            for motor_id, position in final_pose.items():
                await asyncio.sleep(0)
                yield "FAKEBUS", motor_id, SimpleNamespace(position=position)

    async def fake_connect(host=None):
        return object()

    async def fake_set_position(client, bus_serial, motor_id, position):
        commands.append((motor_id, position))

    class FakeSource:
        def read(self):
            return True, "frame", time.monotonic(), None

        def release(self):
            pass

    class FakePipeline:
        def process(self, frame, ts):
            return SimpleNamespace(found=True, newtons=force_n)

    monkeypatch.setattr(lib.station, "follow_motor_state", lambda client: fake_motor_stream())
    monkeypatch.setattr(lib.station, "connect", fake_connect)
    monkeypatch.setattr(lib.station, "set_position", fake_set_position)
    monkeypatch.setattr(lib.camera, "open_source", lambda camera=None: FakeSource())
    monkeypatch.setattr(gauge_reader.cli, "_load_calibration", lambda path: {})
    monkeypatch.setattr(gauge_reader.cli, "make_pipeline", lambda calib, filt=None: FakePipeline())
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "yes")
    return commands


def run_a_short_session(run_session, tmp_path, extra_args=()):
    calib = tmp_path / "calib.json"
    calib.write_text("{}")
    args = build_args(run_session, [
        "--no-coordinated", "--motor", "4", "--ticks-per-metre", "4000", "--yield-sign", "-1",
        "--bus-serial", "FAKEBUS", "--station-host", "fake",
        "--calib", str(calib), "--max-seconds", "0.2", "--reps", "1",
        "--go-home-settle", "0",
        *extra_args,
    ])
    asyncio.run(run_session.run(args))


def test_it_walks_home_when_the_session_ends_away_from_it(tmp_path, monkeypatch):
    """`commands` also carries the finally block's own unload-the-spring back-off on the
    tracked motor (`ControlLoop.release`), which fires before the home-check runs and is
    not what this test is about. Taking the LAST command per motor, rather than demanding
    the list be exactly the home walk, is what keeps this test about the home return
    specifically instead of also re-asserting release's existence."""
    import session.home_pose

    away = {1: 2000, 2: 2000, 3: 2000, 4: 2772, 5: 2000, 6: 2000, 7: 2000}
    home = {1: 2050, 2: 2050, 3: 2050, 4: 2050, 5: 2050, 6: 2050, 7: 2050}
    monkeypatch.setattr(session.home_pose, "load_home_pose", lambda *a, **kw: dict(home))

    commands = session_fakes(monkeypatch, away)
    run_session = load_script()
    run_a_short_session(run_session, tmp_path)

    last_commanded: dict[int, int] = {}
    for motor_id, position in commands:
        last_commanded[motor_id] = position
    assert last_commanded == home, f"expected the last command per joint to be home, got {last_commanded}"


def test_it_does_nothing_extra_when_already_home(tmp_path, monkeypatch):
    """Only the finally block's release back-off should fire - no joint but the tracked one
    gets a command at all, because there is no home walk to take."""
    import session.home_pose

    home = {1: 2050, 2: 2050, 3: 2050, 4: 2772, 5: 2050, 6: 2050, 7: 2050}
    monkeypatch.setattr(session.home_pose, "load_home_pose", lambda *a, **kw: dict(home))

    commands = session_fakes(monkeypatch, home)
    run_session = load_script()
    run_a_short_session(run_session, tmp_path)

    assert {m for m, _ in commands} == {4}, (
        f"only the release back-off (motor 4) should have fired, got {commands}"
    )


def test_no_home_check_skips_the_return_entirely(tmp_path, monkeypatch):
    """Same shape as the already-home case: the release back-off still happens (it is
    unconditional, by design), but nothing related to the home pose does."""
    import session.home_pose

    away = {1: 2000, 2: 2000, 3: 2000, 4: 2772, 5: 2000, 6: 2000, 7: 2000}
    home = {1: 2050, 2: 2050, 3: 2050, 4: 2050, 5: 2050, 6: 2050, 7: 2050}
    monkeypatch.setattr(session.home_pose, "load_home_pose", lambda *a, **kw: dict(home))

    commands = session_fakes(monkeypatch, away)
    run_session = load_script()
    run_a_short_session(run_session, tmp_path, extra_args=["--no-home-check"])

    assert {m for m, _ in commands} == {4}, (
        f"--no-home-check must not command any home-related motion, got {commands}"
    )
    assert all(position != home[4] for _, position in commands), (
        "the single command must be the release back-off, not a move toward home"
    )


def test_a_missing_joint_on_the_way_home_aborts_without_crashing(tmp_path, monkeypatch, capsys):
    import session.home_pose

    # Only six of the home pose's seven joints are ever reported.
    partial = {1: 2000, 2: 2000, 3: 2000, 4: 2772, 5: 2000, 6: 2000}
    home = {1: 2050, 2: 2050, 3: 2050, 4: 2050, 5: 2050, 6: 2050, 7: 2050}
    monkeypatch.setattr(session.home_pose, "load_home_pose", lambda *a, **kw: dict(home))

    commands = session_fakes(monkeypatch, partial)
    run_session = load_script()
    run_a_short_session(run_session, tmp_path)

    assert {m for m, _ in commands} == {4}, (
        f"only the release back-off should have fired, got {commands}"
    )
    assert "could not read motor" in capsys.readouterr().out


# ---------------------------------------------------------------- the dead-position warning
def test_a_dead_position_stream_is_warned_about_loudly(tmp_path, monkeypatch, capsys):
    """Reproduces, without a camera or an arm, the failure found on the bench a second time:
    Station's position stream went dead for the whole session (a real, intermittent
    connectivity issue, not the scheduling starvation fixed earlier), so the force and the
    printed 'step' column - both driven by the camera and the admittance law alone - kept
    changing normally while the arm was never actually commanded. That looked exactly like a
    working session until 'arm travelled 0.0 mm' at the very end.

    `loop.step()` already refuses to command anything without a position, which is correct;
    what was missing was telling the operator mid-session rather than leaving them to notice
    only from the final distance. This asserts the warning appears, not that the arm moves -
    the arm correctly not moving here is the point.
    """
    import asyncio
    from types import SimpleNamespace

    import gauge_reader.cli
    import lib.camera
    import lib.station

    async def dies_midway_stream():
        """Delivers normally, then goes quiet, modelling a bus that drops mid-session.

        Not dead from the start: that case is now refused before the session begins
        (`test_it_refuses_to_start_when_no_position_ever_arrives`), so it can never reach
        the warning. The warning exists for the stream that works long enough to get going
        and then stops, which is invisible otherwise.
        """
        for _ in range(40):
            await asyncio.sleep(0.005)
            yield "FAKEBUS", 4, SimpleNamespace(position=2000)
        await asyncio.sleep(30)

    async def fake_connect(host=None):
        return object()

    async def fake_set_position(client, bus_serial, motor_id, position):
        pass

    class FakeSource:
        def read(self):
            return True, "frame", time.monotonic(), None

        def release(self):
            pass

    class FakePipeline:
        def process(self, frame, ts):
            return SimpleNamespace(found=True, newtons=0.2)

    monkeypatch.setattr(lib.station, "follow_motor_state", lambda client: dies_midway_stream())
    monkeypatch.setattr(lib.station, "connect", fake_connect)
    monkeypatch.setattr(lib.station, "set_position", fake_set_position)
    monkeypatch.setattr(lib.camera, "open_source", lambda camera=None: FakeSource())
    monkeypatch.setattr(gauge_reader.cli, "_load_calibration", lambda path: {})
    monkeypatch.setattr(gauge_reader.cli, "make_pipeline", lambda calib, filt=None: FakePipeline())
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "yes")

    calib = tmp_path / "calib.json"
    calib.write_text("{}")

    run_session = load_script()
    args = build_args(run_session, [
        "--no-coordinated", "--motor", "4", "--ticks-per-metre", "4000", "--yield-sign", "-1",
        "--bus-serial", "FAKEBUS", "--station-host", "fake",
        "--calib", str(calib), "--max-seconds", "0.6", "--reps", "1",
        "--position-stale-warning-s", "0.2", "--no-home-check",
    ])
    asyncio.run(run_session.run(args))

    out = capsys.readouterr().out
    assert "no real arm position" in out
    assert "none received yet this session" not in out, (
        "the stream did deliver before it died, so the warning must not claim otherwise"
    )


# ------------------------------------------------- the camera must not starve the position reader
def test_a_slow_camera_does_not_starve_the_position_reader(tmp_path, monkeypatch):
    """Reproduces the bench failure where three full reps ran, the force climbed past the
    cap, the step column tracked the admittance law, and the arm received nothing.

    The cause was the frame grab and the gauge measurement running inline on the event loop
    with only `await asyncio.sleep(0)` between iterations. One turn per frame is not enough
    for Station's position subscription to even finish starting: instrumented on the real
    bench, its first reading arrived 3.4 s in with a trivial pipeline and never within 22 s
    with the real one. `ControlLoop.step` then correctly refuses to command an arm whose
    position it does not know, which is invisible in the output because the force and step
    columns come from the camera alone.

    The fake camera here blocks for 100 ms per frame, like the real one, and the fake Station
    only delivers positions when the event loop actually gets to run. The assertion is the
    outcome that failed: with the force held over the cap, the arm must be commanded.
    """
    import asyncio
    from types import SimpleNamespace

    import gauge_reader.cli
    import lib.camera
    import lib.station

    commands: list[tuple[int, int]] = []

    async def fake_motor_stream():
        # Bare `sleep(0)` hops, not timed sleeps: a timed sleep fires on schedule even when
        # the loop is starved, so it would not model this failure at all. Station's real
        # subscription advances a hop at a time through a chain of tasks (socket, read task,
        # follow task, generator), so it needs many consecutive turns of the event loop
        # before its first reading, which is exactly what the starved loop never gave it.
        for _ in range(400):
            await asyncio.sleep(0)
        while True:
            for motor_id in (1, 2, 3, 4, 5, 6, 7):
                await asyncio.sleep(0)
                yield "FAKEBUS", motor_id, SimpleNamespace(position=2000 + motor_id)

    async def fake_connect(host=None):
        return object()

    async def fake_set_position(client, bus_serial, motor_id, position):
        commands.append((motor_id, position))

    class SlowSource:
        def read(self):
            time.sleep(0.1)          # a real 1080p grab, blocking, as it is on the bench
            return True, "frame", time.monotonic(), None

        def release(self):
            pass

    class SlowPipeline:
        def process(self, frame, ts):
            time.sleep(0.05)         # the real sticker detection's CPU cost
            return SimpleNamespace(found=True, newtons=2.2)   # 224 gf, over the cap

    monkeypatch.setattr(lib.station, "follow_motor_state", lambda client: fake_motor_stream())
    monkeypatch.setattr(lib.station, "connect", fake_connect)
    monkeypatch.setattr(lib.station, "set_position", fake_set_position)
    monkeypatch.setattr(lib.camera, "open_source", lambda camera=None: SlowSource())
    monkeypatch.setattr(gauge_reader.cli, "_load_calibration", lambda path: {})
    monkeypatch.setattr(gauge_reader.cli, "make_pipeline", lambda calib, filt=None: SlowPipeline())
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "yes")

    calib = tmp_path / "calib.json"
    calib.write_text("{}")

    run_session = load_script()
    args = build_args(run_session, [
        "--no-coordinated", "--motor", "4", "--ticks-per-metre", "4000", "--yield-sign", "-1",
        "--bus-serial", "FAKEBUS", "--station-host", "fake",
        "--calib", str(calib), "--max-seconds", "3", "--reps", "1",
        "--no-home-check", "--no-yield-check",
    ])
    asyncio.run(run_session.run(args))

    assert commands, (
        "the force sat over the cap for the whole session and the arm was never commanded: "
        "the blocking camera work is starving the position reader again"
    )


def test_it_refuses_to_start_when_no_position_ever_arrives(tmp_path, monkeypatch):
    """The second half of the same failure: if the position never arrives at all, the session
    must refuse up front rather than run to completion looking normal."""
    import asyncio
    from types import SimpleNamespace

    import gauge_reader.cli
    import lib.camera
    import lib.station

    commands: list[tuple[int, int]] = []

    async def dead_stream():
        await asyncio.sleep(30)
        return
        yield  # pragma: no cover - makes this an async generator

    async def fake_connect(host=None):
        return object()

    async def fake_set_position(client, bus_serial, motor_id, position):
        commands.append((motor_id, position))

    class FakeSource:
        def read(self):
            return True, "frame", time.monotonic(), None

        def release(self):
            pass

    class FakePipeline:
        def process(self, frame, ts):
            return SimpleNamespace(found=True, newtons=2.2)

    monkeypatch.setattr(lib.station, "follow_motor_state", lambda client: dead_stream())
    monkeypatch.setattr(lib.station, "connect", fake_connect)
    monkeypatch.setattr(lib.station, "set_position", fake_set_position)
    monkeypatch.setattr(lib.camera, "open_source", lambda camera=None: FakeSource())
    monkeypatch.setattr(gauge_reader.cli, "_load_calibration", lambda path: {})
    monkeypatch.setattr(gauge_reader.cli, "make_pipeline", lambda calib, filt=None: FakePipeline())
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "yes")

    calib = tmp_path / "calib.json"
    calib.write_text("{}")

    run_session = load_script()
    args = build_args(run_session, [
        "--no-coordinated", "--motor", "4", "--ticks-per-metre", "4000", "--yield-sign", "-1",
        "--bus-serial", "FAKEBUS", "--station-host", "fake",
        "--calib", str(calib), "--max-seconds", "5", "--reps", "1",
        "--position-wait-s", "0.5", "--no-home-check",
    ])
    with pytest.raises(SystemExit, match="no arm position arrived"):
        asyncio.run(run_session.run(args))
    assert commands == [], "nothing may be commanded when the session refuses to start"


# ---------------------------------------------------------------- the separate deadman knob
def test_blind_grace_s_is_a_real_cli_flag_wired_to_rep_limits(run_session):
    """--max-dropouts only reaches SafetyLimits; it was mistaken, on the bench, for covering
    session.reps.RepMachine's own device-driven-phase deadman too, and it does not. This
    pins that --blind-grace-s exists and actually reaches RepLimits, so raising tolerance for
    a device-driven abort has an effect distinct from --max-dropouts."""
    args = build_args(run_session, ["--no-coordinated", "--motor", "4", "--blind-grace-s", "1.5"])
    assert args.blind_grace_s == 1.5


def test_blind_grace_s_defaults_to_the_conservative_value(run_session):
    args = build_args(run_session, ["--no-coordinated", "--motor", "4"])
    assert args.blind_grace_s == 0.3


# ---------------------------------------------------------------- the outlier-persist override
def test_outlier_persist_defaults_higher_than_the_readers_own_default(run_session):
    """Found on the bench: a bogus ~555 gf reading persisted for exactly 3 frames, the gauge
    reader's own default, and was trusted by its outlier gate - by design, since a signal
    that persists that long is supposed to mean a real load change, not noise. A run that
    moves the arm on what it reads has more reason to wait out a glitch than a read-only
    viewer, hence a higher default here specifically."""
    args = build_args(run_session, ["--no-coordinated", "--motor", "4"])
    assert args.outlier_persist == 5


def test_outlier_persist_is_overridable(run_session):
    args = build_args(run_session, ["--no-coordinated", "--motor", "4", "--outlier-persist", "9"])
    assert args.outlier_persist == 9


def test_build_pipeline_passes_outlier_persist_through(run_session, tmp_path):
    from gauge_reader.calibration import Calibration

    calib_path = tmp_path / "calib.json"
    Calibration.from_clicks((0, 0), (100, 0)).save(str(calib_path))

    pipeline = run_session.build_pipeline(str(calib_path), outlier_persist=9)
    assert pipeline.filt.outlier_persist == 9


def test_build_pipeline_defaults_outlier_persist_to_three(run_session, tmp_path):
    """build_pipeline's OWN default stays at the library default (3): the higher default (5)
    is a decision made once, in main()'s argument parser, not duplicated here - this pins
    that build_pipeline itself does not silently raise it a second time."""
    from gauge_reader.calibration import Calibration

    calib_path = tmp_path / "calib.json"
    Calibration.from_clicks((0, 0), (100, 0)).save(str(calib_path))

    pipeline = run_session.build_pipeline(str(calib_path))
    assert pipeline.filt.outlier_persist == 3


# ---------------------------------------------------------------- the retrace knob
def test_recover_retrace_mm_s_defaults_to_a_slow_nonzero_speed(run_session):
    """Disabled (0) was the original behaviour, and on the bench it meant a finger that
    relaxed faster than --v-max-mm-s could follow left the arm stranded tens of millimetres
    from home, with 'Relax, the arm will bring your finger back' never actually happening.
    The default here is intentionally nonzero."""
    args = build_args(run_session, ["--no-coordinated", "--motor", "4"])
    assert args.recover_retrace_mm_s == load_tuning().recover_retrace_mm_s > 0


def test_recover_retrace_mm_s_is_overridable(run_session):
    args = build_args(run_session, ["--no-coordinated", "--motor", "4", "--recover-retrace-mm-s", "0"])
    assert args.recover_retrace_mm_s == 0.0


# ------------------------------------------------------------- the pull law reaches the arm
# `pull_plan_for` is the function that was missing between session/modes.py specifying a
# device-driven pull and anything being able to perform one. Before it this script read
# neither step.law nor step.target, which is why --mode passive printed its phases while the
# arm never moved.

def pull_args(run_session, extra=()):
    return build_args(run_session, ["--no-coordinated", "--motor", "4", "--mode", "cpm",
                                    "--rom-mm", "15", *extra])


class FakeStep:
    """Just the three fields `pull_plan_for` reads off a RepStep."""

    def __init__(self, law, target):
        self.law = law
        self.target = target


def fake_machine(rom_target_m=0.015, pretension_n=0.59):
    from session.protocol import Block

    class M:
        block = Block(finger="index", mode="cpm", reps=2, cap_n=1.96, hold_s=0.5,
                      rom_target_m=rom_target_m)
    M.pretension_n = pretension_n
    return M


def test_a_yielding_cycle_asks_for_no_pull(run_session):
    """The yield law is what runs then, and it is the one already proven on the bench."""
    from session.modes import Law, Target

    plan = run_session.pull_plan_for(FakeStep(Law.YIELD, Target.NONE),
                                     fake_machine(), pull_args(run_session))
    assert plan is None


def test_a_still_cycle_asks_for_no_pull(run_session):
    from session.modes import Law, Target

    assert run_session.pull_plan_for(FakeStep(Law.STILL, Target.NONE),
                                     fake_machine(), pull_args(run_session)) is None


def test_a_rom_targeted_pull_aims_at_the_prescribed_stretch(run_session):
    """Negative, because stretching a finger toward extension raises tension, which is the
    direction a yielding arm moves away from."""
    from session.modes import Law, Target

    plan = run_session.pull_plan_for(FakeStep(Law.PULL, Target.ROM),
                                     fake_machine(rom_target_m=0.015), pull_args(run_session))
    assert plan.target_yielded_m == pytest.approx(-0.015)


def test_a_rep_start_pull_aims_at_zero(run_session):
    """The same zero session.reps.RepMachine uses to decide a reset has arrived."""
    from session.modes import Law, Target

    plan = run_session.pull_plan_for(FakeStep(Law.PULL, Target.REP_START),
                                     fake_machine(), pull_args(run_session))
    assert plan.target_yielded_m == 0.0


def test_the_two_pull_directions_are_opposite(run_session):
    """The whole reason step.target has to be read and not just step.law: in cpm both beats
    are Law.PULL and they aim opposite ways, so a caller holding only the law would run the
    return stroke outward."""
    from session.modes import Law, Target

    args, machine = pull_args(run_session), fake_machine()
    out = run_session.pull_plan_for(FakeStep(Law.PULL, Target.ROM), machine, args)
    back = run_session.pull_plan_for(FakeStep(Law.PULL, Target.REP_START), machine, args)
    assert out.target_yielded_m < back.target_yielded_m


def test_arming_is_bounded_by_a_force_not_a_position(run_session):
    """ARMED is the one pull with no position target: it takes slack up to the tether's
    resting tension and must stop there, well under the cap, or the patient is handed a finger
    already at therapeutic force while still being cued."""
    from session.modes import Law, Target

    machine = fake_machine(pretension_n=0.59)
    plan = run_session.pull_plan_for(FakeStep(Law.PULL, Target.NONE), machine,
                                     pull_args(run_session))
    assert plan.stop_above_n == pytest.approx(0.59)
    assert plan.target_yielded_m < 0, "it still needs somewhere to go while taking slack up"


def test_arming_is_slower_than_the_working_stroke(run_session):
    """It is the one pull that starts with the tether slack and a finger already through the
    hook, so it is the stroke that knows least about what it is about to touch."""
    from session.modes import Law, Target

    args = pull_args(run_session, ["--pull-mm-s", "30", "--arm-mm-s", "5"])
    machine = fake_machine()
    arming = run_session.pull_plan_for(FakeStep(Law.PULL, Target.NONE), machine, args)
    working = run_session.pull_plan_for(FakeStep(Law.PULL, Target.ROM), machine, args)
    assert arming.speed_m_s < working.speed_m_s


def test_the_pull_speed_comes_from_the_flag(run_session):
    from session.modes import Law, Target

    plan = run_session.pull_plan_for(
        FakeStep(Law.PULL, Target.ROM), fake_machine(),
        pull_args(run_session, ["--pull-mm-s", "25"]),
    )
    assert plan.speed_m_s == pytest.approx(0.025)


def test_the_pull_step_cap_is_separate_from_the_yield_one(run_session):
    """--max-step-mm bounds the yield step's smoothness against a flexing finger, and a pull
    is not that motion, so raising one must not move the other."""
    from session.modes import Law, Target

    args = pull_args(run_session, ["--max-step-mm", "1", "--pull-max-step-mm", "6"])
    plan = run_session.pull_plan_for(FakeStep(Law.PULL, Target.ROM), fake_machine(), args)
    assert plan.max_step_m == pytest.approx(0.006)
    assert args.max_step_mm == 1.0


def test_cpm_requires_a_stretch_target(run_session):
    """Under the cap nothing else tells a pulling arm to stop, so Block refuses it outright."""
    from session.protocol import Block

    with pytest.raises(ValueError, match="needs rom_target_m"):
        Block(finger="index", mode="cpm", reps=1, cap_n=1.96, hold_s=0.5)


def test_the_pull_defaults_come_from_the_tuning_profile(run_session):
    t = load_tuning()
    args = pull_args(run_session)
    assert args.pull_mm_s == t.pull_mm_s
    assert args.pull_max_step_mm == t.pull_max_step_mm
    assert args.arm_mm_s == t.arm_mm_s


# ------------------------------------------------------- loading a clinician's protocol
# The prescription half of the split: `config/protocols/` is what a clinician authored,
# `config/session-tuning.json` is what the rig needs. This is the path from the first one
# into a session.

REPO = Path(__file__).resolve().parents[2]
ACTIVE_PROTOCOL = str(REPO / "config" / "protocols" / "tendon-repair-active.json")
CPM_PROTOCOL = str(REPO / "config" / "protocols" / "age-related-weakness-cpm.json")


def block_from(run_session, argv, capsys=None):
    args = build_args(run_session, ["--no-coordinated", "--motor", "4", *argv])
    return run_session.block_for(args)


def test_without_a_protocol_the_flags_still_build_the_block(run_session, capsys):
    """The path every bench session has used so far must keep working untouched."""
    block = block_from(run_session, ["--finger", "ring", "--cap-gf", "120", "--reps", "7"])
    capsys.readouterr()
    assert (block.finger, block.reps) == ("ring", 7)
    assert block.cap_gf == pytest.approx(120, abs=0.5)


def test_a_protocol_file_supplies_the_whole_prescription(run_session, capsys):
    block = block_from(run_session, ["--protocol", ACTIVE_PROTOCOL])
    capsys.readouterr()
    assert block.finger == "index"
    assert block.mode == "active"
    assert block.cap_gf == pytest.approx(120, abs=0.5)


def test_a_protocol_beats_the_flags_for_what_was_prescribed(run_session, capsys):
    """The file is what a clinician authored and a flag is what someone typed at a prompt."""
    block = block_from(run_session, ["--protocol", ACTIVE_PROTOCOL,
                                     "--finger", "little", "--cap-gf", "400"])
    capsys.readouterr()
    assert block.finger == "index", "the protocol's finger, not the flag's"
    assert block.cap_gf == pytest.approx(120, abs=0.5), "the cap is never taken from a flag here"


def test_the_cap_cannot_be_raised_by_a_flag_while_claiming_to_run_a_protocol(run_session, capsys):
    """A cap is the number the whole prescription is built around. Changing it by flag while
    the log says 'ran tendon-repair-active' is how the log stops meaning anything."""
    block = block_from(run_session, ["--protocol", ACTIVE_PROTOCOL, "--cap-gf", "590"])
    capsys.readouterr()
    assert block.cap_gf < 200


def test_a_block_can_be_picked_out_of_a_protocol(run_session, capsys):
    """This script runs one block; sequencing a whole protocol across fingers needs arm 2."""
    block = block_from(run_session, ["--protocol", ACTIVE_PROTOCOL, "--protocol-block", "4"])
    capsys.readouterr()
    assert block.finger == "ring"


def test_a_block_that_does_not_exist_is_refused_with_the_ones_that_do(run_session):
    """An index typo should print the menu, not a traceback."""
    with pytest.raises(SystemExit, match="does not exist"):
        block_from(run_session, ["--protocol", ACTIVE_PROTOCOL, "--protocol-block", "99"])


def test_reps_can_be_shortened_for_one_run(run_session, capsys):
    """The one thing a person reasonably changes at the bench without editing a prescription."""
    block = block_from(run_session, ["--protocol", ACTIVE_PROTOCOL, "--reps", "2"])
    capsys.readouterr()
    assert block.reps == 2


def test_an_override_is_announced_rather_than_silent(run_session, capsys):
    """A session that quietly ran something other than what the file says is the thing worth
    preventing: the banner and the log have to disagree with the file out loud."""
    block_from(run_session, ["--protocol", ACTIVE_PROTOCOL, "--reps", "2"])
    printed = capsys.readouterr().out
    assert "overriding reps" in printed
    assert "5 -> 2" in printed


def test_not_overriding_announces_nothing(run_session, capsys):
    block_from(run_session, ["--protocol", ACTIVE_PROTOCOL])
    assert "overriding" not in capsys.readouterr().out


def test_the_protocol_name_is_printed_so_the_log_says_what_ran(run_session, capsys):
    block_from(run_session, ["--protocol", ACTIVE_PROTOCOL])
    printed = capsys.readouterr().out
    assert "EXAMPLE" in printed, "the example protocols say so in their name, and it must show"


def test_a_cpm_protocol_carries_its_stretch_target(run_session, capsys):
    """Without it there is nothing to tell a pulling arm where to stop, and Block refuses the
    prescription outright rather than letting the session discover that."""
    block = block_from(run_session, ["--protocol", CPM_PROTOCOL])
    capsys.readouterr()
    assert block.mode == "cpm"
    assert block.rom_target_m == pytest.approx(0.012)


def test_every_block_of_every_example_protocol_can_be_loaded(run_session, capsys):
    """Catches a protocol that parses but names a block this script cannot run."""
    from session.protocol import Protocol

    for path in (ACTIVE_PROTOCOL, CPM_PROTOCOL):
        for i in range(len(Protocol.load(path))):
            assert block_from(run_session, ["--protocol", path, "--protocol-block", str(i)])
            capsys.readouterr()
