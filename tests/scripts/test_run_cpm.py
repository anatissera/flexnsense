"""The CPM runner: a fixed path, and a camera that supervises rather than drives.

The properties worth testing without hardware are the ones that make this a different kind of
thing from `scripts/session/run_session.py`:

- the path is computed once and is the same every repetition, which is what "the same motion
  every time" actually means;
- the camera can stop the stroke and cannot steer it;
- it starts from its own pose, not from wherever the arm was left.
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
URDF = REPO / "norma-core/hardware/elrobot/simulation/elrobot_follower.urdf"

#: The pose the arm was actually standing in on 2026-11-15, read from Station.
POSE = {1: 2049, 2: 1469, 3: 2468, 4: 2772, 5: 1879, 6: 1525, 7: 3275, 8: 1301}


@pytest.fixture(scope="module")
def run_cpm():
    spec = importlib.util.spec_from_file_location("run_cpm", REPO / "scripts" / "session" / "run_cpm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_args(run_cpm, argv):
    """main()'s own parser, run without running main()."""
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
            run_cpm.main(argv)
    finally:
        argparse.ArgumentParser.parse_args = real_parse
    return captured["args"]


def build_jog_args(argv):
    """scripts/arm/jog_coordinated.py's own parser, so a test can compare against its real
    defaults rather than a second copy of 700/50 that could drift from the original."""
    spec = importlib.util.spec_from_file_location(
        "jog_coordinated", REPO / "scripts" / "arm" / "jog_coordinated.py"
    )
    jog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jog)
    return build_args(jog, argv)


# --------------------------------------------------------------------------- the CLI contract
def test_the_stroke_travels_between_the_two_named_poses(run_cpm):
    """The bench finding this replaced a fixed 15 mm line over: the two saved poses are 80 mm
    apart at the gripper, so a 15 mm stroke was showing under a fifth of the intended motion.
    The destination is a pose, not a distance, so the stroke is whatever those two poses are
    apart, and re-capturing either one changes it."""
    args = build_args(run_cpm, [])
    assert args.to == "home"
    assert not hasattr(args, "distance_mm"), (
        "the stroke is pose to pose now; a distance would be a second, disagreeing source "
        "of truth for how far it goes"
    )


def test_the_full_stroke_can_be_eased_into(run_cpm):
    """At full reach the stroke is most of a finger's measured excursion, so there has to be
    a way to run a fraction of it before running all of it on a hand."""
    assert build_args(run_cpm, []).reach == 1.0
    assert build_args(run_cpm, ["--reach", "0.3"]).reach == pytest.approx(0.3)


def test_the_step_is_measured_in_ticks_not_millimetres(run_cpm):
    """The judder found on the bench: a 1 mm step asks the three joints for 0.84, 4.16 and
    6.38 ticks, and rounding 0.84 to 1 is a 20% error in that joint's share, every cycle.
    Ticks are what the servo takes, so a step bounded in ticks is the one that can promise a
    commanded step is a real move rather than a sub-tick twitch."""
    args = build_args(run_cpm, [])
    assert args.step_ticks >= 10
    assert not hasattr(args, "step_mm")


def test_it_starts_from_its_own_pose_not_the_session_home(run_cpm):
    """The session home is where the arm is parked with nothing hooked. A CPM stroke needs
    somewhere to travel to with the finger attached, which is a different pose."""
    args = build_args(run_cpm, [])
    assert args.start_pose == "config/calibration/cpm-start-pose.json"
    assert "home-pose.json" not in args.start_pose


def test_the_camera_can_be_turned_off_entirely_for_a_dry_path_check(run_cpm):
    """The same way scripts/arm/jog_coordinated.py is used: confirm the line with nothing
    attached before putting a finger anywhere near it."""
    assert build_args(run_cpm, ["--no-camera"]).no_camera is True


def test_the_ceiling_comes_from_the_tuning_profile(run_cpm):
    from arm1.session.tuning import load_tuning

    t = load_tuning()
    args = build_args(run_cpm, [])
    assert args.ceiling_gf == t.safety.ceiling_gf
    assert args.outlier_persist == t.outlier_persist


def test_speed_and_accel_come_from_jog_coordinated_not_the_tuning_profile(run_cpm):
    """The bug the first bench run found: defaulting to config/session-tuning.json's
    coordinated_speed/accel (900/30) borrows a profile tuned for the admittance loop's
    continuous small corrections, not for this open-loop waypoint walk. With a low accel and
    a short settle, the servo never finished ramping up before the next 1 mm waypoint
    overwrote its goal, so every step looked like a jerky restart. scripts/arm/jog_coordinated.py
    tunes exactly this kind of motion and found 700/50 smooth; this has to use that, not the
    other script's number, on purpose."""
    from arm1.session.tuning import load_tuning

    t = load_tuning()
    args = build_args(run_cpm, [])
    assert (args.speed, args.accel) == (run_cpm.JOG_PROVEN_SPEED, run_cpm.JOG_PROVEN_ACCEL)
    assert (args.speed, args.accel) != (t.coordinated_speed, t.coordinated_accel)


def test_the_proven_profile_matches_jog_coordinated_py_exactly(run_cpm):
    """Not a coincidence this test can check: both should read as the same two numbers,
    because scripts/arm/jog_coordinated.py is where 700/50 was found to be smooth, and the whole
    point of borrowing it is to get that same proven motion here."""
    jog_defaults = build_jog_args([])
    assert run_cpm.JOG_PROVEN_SPEED == jog_defaults.speed
    assert run_cpm.JOG_PROVEN_ACCEL == jog_defaults.accel


def test_the_settle_is_also_the_proven_one_not_a_shorter_invented_one(run_cpm):
    """A settle shorter than jog_coordinated.py's proven 0.3s was the other half of the same
    bug: it cut each step off before the servo's own ramp (governed by --accel) had time to
    do anything, which reads as clunky regardless of how high --speed is set."""
    args = build_args(run_cpm, [])
    assert args.settle == pytest.approx(0.3)


def test_there_is_no_cap_flag_only_a_ceiling(run_cpm):
    """A cap is a force an exercise works against, and this mode has none: the stroke is
    bounded by its path. Offering one would invite the force-gated design this replaced."""
    args = build_args(run_cpm, [])
    assert not hasattr(args, "cap_gf")


def test_a_missing_start_pose_says_how_to_make_one(run_cpm, tmp_path):
    """A refusal that names the fix, rather than a traceback about a missing file."""
    import asyncio
    import types

    args = types.SimpleNamespace(
        start_pose=str(tmp_path / "nope.json"), bus_serial="FAKE", station_host="localhost",
        tuning=None,
    )
    with pytest.raises(SystemExit, match="set_home_pose.py --save --to cpm"):
        asyncio.run(run_cpm.run(args))


# ------------------------------------------------------------------ the path itself
@pytest.mark.skipif(not URDF.is_file(),
                    reason="needs the norma-core clone (hardware/elrobot/simulation)")
def test_the_round_trip_returns_to_exactly_where_it_began():
    """Not approximately: the return visits the same waypoints in reverse and ends on the
    start pose itself, so a repetition cannot drift from the line that was checked."""
    from arm1.lib.kinematics import load_chain, tether_tension_direction

    chain = load_chain(URDF)
    direction = tether_tension_direction(chain, POSE)
    motors = chain.choose_pull_motors(POSE, direction)
    waypoints = chain.line_trajectory(motors, POSE, -direction, 0.010, steps=10)
    round_trip = waypoints + list(reversed([POSE] + waypoints[:-1]))
    assert round_trip[-1] == POSE


@pytest.mark.skipif(not URDF.is_file(),
                    reason="needs the norma-core clone (hardware/elrobot/simulation)")
def test_every_repetition_follows_the_identical_path():
    """The property that distinguishes this mode from an admittance session, where the path
    is whatever the patient and the camera produced between them."""
    from arm1.lib.kinematics import load_chain, tether_tension_direction

    chain = load_chain(URDF)
    direction = tether_tension_direction(chain, POSE)
    motors = chain.choose_pull_motors(POSE, direction)
    waypoints = chain.line_trajectory(motors, POSE, -direction, 0.010, steps=10)
    round_trip = waypoints + list(reversed([POSE] + waypoints[:-1]))
    # Three repetitions is the same list three times, by construction: nothing recomputes it.
    assert round_trip * 3 == round_trip + round_trip + round_trip


@pytest.mark.skipif(not URDF.is_file(),
                    reason="needs the norma-core clone (hardware/elrobot/simulation)")
def test_the_stroke_stays_on_the_tether_line():
    """The whole reason three joints move instead of one. Checked at the distance this script
    actually defaults to, not at the 90 mm a full session needs."""
    import numpy as np

    from arm1.lib.kinematics import load_chain, tether_tension_direction

    chain = load_chain(URDF)
    direction = tether_tension_direction(chain, POSE)
    motors = chain.choose_pull_motors(POSE, direction)
    travel_dir = -direction
    waypoints = chain.line_trajectory(motors, POSE, travel_dir, 0.010, steps=10)
    start = chain.tip_position(POSE)
    worst = 0.0
    for way in waypoints:
        delta = chain.tip_position(way) - start
        along = float(np.dot(delta, travel_dir))
        worst = max(worst, float(np.linalg.norm(delta - along * travel_dir)))
    assert worst < 0.001, f"wandered {worst * 1000:.2f} mm off the line"


# ------------------------------------------------------- the camera supervises, never steers
class FakeSource:
    """Hands out a canned sequence of readings, then repeats the last one."""

    def __init__(self, newtons_sequence):
        self.sequence = list(newtons_sequence)
        self.released = False

    def read(self):
        return True, object(), time.monotonic(), None

    def release(self):
        self.released = True


class FakePipeline:
    def __init__(self, newtons_sequence):
        self.sequence = list(newtons_sequence)

    def process(self, frame, ts):
        value = self.sequence.pop(0) if self.sequence else None

        class Result:
            found = value is not None
            newtons = value

        return Result()


def supervisor_for(run_cpm, readings, ceiling_gf=200.0, blind_tolerance_s=2.0):
    from arm1.lib.units import grams_force_to_newtons

    return run_cpm.ForceSupervisor(
        FakeSource(readings), FakePipeline(readings),
        grams_force_to_newtons(ceiling_gf), blind_tolerance_s,
    )


def test_a_force_under_the_ceiling_does_not_stop_anything(run_cpm):
    from arm1.lib.units import grams_force_to_newtons

    s = supervisor_for(run_cpm, [grams_force_to_newtons(150)] * 3)
    for _ in range(3):
        s.poll_once()
    assert s.stop_reason is None


def test_a_force_over_the_ceiling_stops_the_stroke(run_cpm):
    from arm1.lib.units import grams_force_to_newtons

    s = supervisor_for(run_cpm, [grams_force_to_newtons(250)])
    s.poll_once()
    assert s.stop_reason is not None
    assert "ceiling" in s.stop_reason


def test_the_ceiling_message_names_both_numbers(run_cpm):
    """So the log says what was seen and what was allowed, not just that something tripped."""
    from arm1.lib.units import grams_force_to_newtons

    s = supervisor_for(run_cpm, [grams_force_to_newtons(250)], ceiling_gf=200.0)
    s.poll_once()
    assert "250" in s.stop_reason and "200" in s.stop_reason


def test_going_blind_for_too_long_stops_the_stroke(run_cpm):
    """A missing reading is never evidence of a safe force, which is the project's rule and
    does not stop applying because this motion is open-loop."""
    s = supervisor_for(run_cpm, [None], blind_tolerance_s=0.0)
    s.last_seen = time.monotonic() - 5.0
    s.poll_once()
    assert s.stop_reason is not None
    assert "not evidence of a safe force" in s.stop_reason


def test_a_brief_dropout_does_not_stop_the_stroke(run_cpm):
    """Generous compared with run_session.py, because nothing here depends on the reading step
    by step: a frame the camera missed is not a reason to abandon a bounded path."""
    s = supervisor_for(run_cpm, [None, None], blind_tolerance_s=5.0)
    s.poll_once()
    s.poll_once()
    assert s.stop_reason is None


def test_the_supervisor_tracks_the_peak_for_the_log(run_cpm):
    from arm1.lib.units import grams_force_to_newtons, newtons_to_grams_force

    s = supervisor_for(run_cpm, [grams_force_to_newtons(x) for x in (50, 180, 90)])
    for _ in range(3):
        s.poll_once()
    assert newtons_to_grams_force(s.peak_n) == pytest.approx(180, abs=1)


def test_the_supervisor_has_no_way_to_change_a_step(run_cpm):
    """The structural claim: it exposes a reading, a peak and a stop reason, and nothing that
    could steer the arm. A supervisor that could nudge the path would be the force-gated
    design this script was written to replace."""
    s = supervisor_for(run_cpm, [])
    public = {n for n in dir(s) if not n.startswith("_")}
    steering = {n for n in public if any(
        word in n for word in ("step", "target", "goal", "move", "velocity", "position")
    )}
    assert steering == set(), f"the supervisor should not expose {steering}"


def test_a_reading_renders_as_grams_for_the_operator(run_cpm):
    """Grams is what a person reads off the PESOLA, per docs/contributing.md's units rule."""
    from arm1.lib.units import grams_force_to_newtons

    s = supervisor_for(run_cpm, [grams_force_to_newtons(123)])
    assert s.reading.strip() == "--"
    s.poll_once()
    assert "123" in s.reading and "gf" in s.reading


# ------------------------------------------------------------------ the second pose
def test_set_home_pose_writes_to_exactly_one_of_the_two_named_poses():
    """CPM needs its own start pose, so the capture script has to be able to write a second
    slot - and only that second slot, not an arbitrary path: this repo's rule is that a pose
    is trustworthy because it was placed and measured against the real rig, so the only
    legitimate destinations are the named pair in session.home_pose.NAMED_POSES."""
    from arm1.session.home_pose import NAMED_POSES

    spec = importlib.util.spec_from_file_location(
        "set_home_pose", REPO / "scripts" / "arm" / "set_home_pose.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = (REPO / "scripts" / "arm" / "set_home_pose.py").read_text()
    assert "--path" not in source, "writing to an arbitrary path was deliberately removed"
    assert "args.to" in source
    assert set(NAMED_POSES) == {"home", "cpm"}


def test_a_saved_pose_file_loads_through_the_same_loader(tmp_path):
    """Whatever path it is written to, `session.home_pose.load_home_pose` has to read it, so
    the CPM start pose and the session home stay one format."""
    from arm1.session.home_pose import load_home_pose

    path = tmp_path / "cpm-start-pose.json"
    path.write_text(json.dumps({"ticks": {str(m): v for m, v in POSE.items() if m != 8}}))
    loaded = load_home_pose(path)
    assert loaded[1] == 2049
    assert 8 not in loaded


# ------------------------------------------------------- the two named poses, closed on purpose
def test_named_poses_is_exactly_these_two():
    """Not an open set. Adding a third pose later is a deliberate code change to this dict,
    not a string someone types at a prompt."""
    from arm1.session.home_pose import NAMED_POSES

    assert set(NAMED_POSES) == {"home", "cpm"}


def test_go_home_only_accepts_the_two_named_destinations(capsys):
    """A typed --path would let a typo or an unrelated file become somewhere the arm actually
    drives to. --to has to be closed to argparse's own choices, not just documented as such,
    which is why this calls main() directly rather than through build_args: argparse refuses
    the bad value and exits before build_args's own capture ever runs."""
    spec = importlib.util.spec_from_file_location(
        "go_home", REPO / "scripts" / "arm" / "go_home.py"
    )
    go_home = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(go_home)
    with pytest.raises(SystemExit):
        go_home.main(["--to", "/etc/passwd"])
    assert "invalid choice" in capsys.readouterr().err


def test_go_home_defaults_to_the_session_home():
    spec = importlib.util.spec_from_file_location(
        "go_home", REPO / "scripts" / "arm" / "go_home.py"
    )
    go_home = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(go_home)
    assert build_args(go_home, []).to == "home"


def test_go_home_can_target_the_cpm_start_pose():
    from arm1.session.home_pose import CPM_START_POSE_PATH, NAMED_POSES

    assert NAMED_POSES["cpm"] == CPM_START_POSE_PATH
    spec = importlib.util.spec_from_file_location(
        "go_home", REPO / "scripts" / "arm" / "go_home.py"
    )
    go_home = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(go_home)
    assert build_args(go_home, ["--to", "cpm"]).to == "cpm"


def test_set_home_pose_only_accepts_the_two_named_destinations(capsys):
    spec = importlib.util.spec_from_file_location(
        "set_home_pose", REPO / "scripts" / "arm" / "set_home_pose.py"
    )
    set_home_pose = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(set_home_pose)
    with pytest.raises(SystemExit):
        set_home_pose.main(["--to", "/etc/passwd"])
    assert "invalid choice" in capsys.readouterr().err


def test_run_cpm_and_go_home_agree_on_where_the_cpm_pose_lives(run_cpm):
    """run_cpm.py's own default --start-pose and go_home.py --to cpm have to be the same
    file, or walking there with go_home.py would not actually be where the stroke starts."""
    from arm1.session.home_pose import NAMED_POSES

    args = build_args(run_cpm, [])
    assert args.start_pose == NAMED_POSES["cpm"]


# --------------------------------------------------- the stroke's waypoints, pose to pose
def test_the_waypoints_land_exactly_on_the_destination(run_cpm):
    """Not approximately: the stroke's whole claim is that it ends on a pose someone placed
    and measured by hand, so the last waypoint has to be that pose."""
    start = {1: 1000.0, 2: 2000.0, 3: 1500.0}
    end = {1: 1100.0, 2: 1700.0, 3: 1500.0}
    ways = run_cpm.pose_waypoints(start, end, [1, 2], step_ticks=20)
    assert ways[-1] == pytest.approx(end)


def test_no_joint_moves_more_than_the_step_in_one_go(run_cpm):
    """The bound that keeps a servo from being told to cross a long way at once, which is
    the one thing this arm must never do near a hand."""
    start = {1: 1000.0, 2: 2000.0}
    end = {1: 1000.0, 2: 1400.0}
    ways = run_cpm.pose_waypoints(start, end, [1, 2], step_ticks=20)
    prev = start
    for way in ways:
        assert max(abs(way[m] - prev[m]) for m in way) <= 20 + 1e-9
        prev = way


def test_a_bigger_step_means_fewer_waypoints(run_cpm):
    start, end = {1: 0.0}, {1: 100.0}
    assert len(run_cpm.pose_waypoints(start, end, [1], 10)) > \
        len(run_cpm.pose_waypoints(start, end, [1], 50))


def test_the_waypoints_keep_float_precision(run_cpm):
    """Rounding each waypoint to whole ticks independently quantises the ratio between
    joints, which is exactly what made a small step judder. The ratio has to survive the
    path; the rounding happens once, at the bus."""
    start = {1: 0.0, 2: 0.0}
    end = {1: 100.0, 2: 33.0}
    ways = run_cpm.pose_waypoints(start, end, [1, 2], step_ticks=20)
    assert any(w[2] != round(w[2]) for w in ways), "a float ratio should survive"
    for w in ways:
        assert w[2] / w[1] == pytest.approx(33.0 / 100.0)


def test_joints_that_do_not_differ_are_still_carried_along(run_cpm):
    """Every waypoint is a whole pose, so a joint the stroke does not move still reports the
    value it is meant to hold rather than dropping out of the dict."""
    start = {1: 0.0, 2: 500.0}
    end = {1: 100.0, 2: 500.0}
    ways = run_cpm.pose_waypoints(start, end, [1], step_ticks=20)
    assert all(w[2] == 500.0 for w in ways)


def test_a_stroke_shorter_than_one_step_is_still_one_waypoint(run_cpm):
    """Zero waypoints and "it is already there" would otherwise be the same empty list."""
    ways = run_cpm.pose_waypoints({1: 0.0}, {1: 3.0}, [1], step_ticks=20)
    assert len(ways) == 1
    assert ways[0][1] == pytest.approx(3.0)


def test_every_waypoint_stays_between_the_two_poses(run_cpm):
    """The safety property that replaced can_pull_straight: each waypoint is a per-joint
    convex combination of two in-range poses, so no joint can be commanded past travel that
    either endpoint has not already proved. Checking the endpoints checks the whole path."""
    start = {1: 1000.0, 2: 2500.0}
    end = {1: 1400.0, 2: 1900.0}
    for way in run_cpm.pose_waypoints(start, end, [1, 2], step_ticks=13):
        for m in (1, 2):
            lo, hi = sorted((start[m], end[m]))
            assert lo - 1e-9 <= way[m] <= hi + 1e-9


@pytest.mark.skipif(not URDF.is_file(),
                    reason="needs the norma-core clone (hardware/elrobot/simulation)")
def test_the_real_stroke_is_a_real_motion_not_a_sub_tick_twitch(run_cpm):
    """Against the two poses actually saved on this bench. The first version asked the joints
    for 0.84, 4.16 and 6.38 ticks per step; this asserts the step is now big enough that
    rounding it at the bus is a small error rather than a dominant one."""
    from arm1.lib.kinematics import ARM_MOTORS
    from arm1.session.home_pose import NAMED_POSES, load_home_pose

    start = {m: float(v) for m, v in load_home_pose(NAMED_POSES["cpm"]).items()}
    end = {m: float(v) for m, v in load_home_pose(NAMED_POSES["home"]).items()}
    start = {m: start.get(m, 0.0) for m in ARM_MOTORS}
    end = {m: end.get(m, start[m]) for m in ARM_MOTORS}
    motors = sorted(m for m in ARM_MOTORS if abs(end[m] - start[m]) >= 1.0)

    ways = run_cpm.pose_waypoints(start, end, motors, step_ticks=20)
    biggest = max(abs(ways[0][m] - start[m]) for m in motors)
    assert biggest > 10, (
        f"the leading joint only moves {biggest:.1f} ticks per step, which is the regime "
        "where rounding at the bus dominates and the motion judders"
    )


# ------------------------------------------------ measuring what settle should actually be
def test_there_is_a_way_to_measure_a_step_rather_than_guess_settle(run_cpm):
    """lib/station.py records that RAM_GOAL_SPEED's unit is unknown, with one confirmed point
    on the whole scale. So how long a 20-tick step takes at speed 700 cannot be derived from
    anything in this repo, and a --settle picked without measuring is a guess that shows up
    as either dead time (judder) or goals landing before the last one finished (raggedness)."""
    assert build_args(run_cpm, ["--measure-step"]).measure_step is True
    assert build_args(run_cpm, []).measure_step is False


def test_measuring_is_not_the_default(run_cpm):
    """It moves the arm, so it has to be asked for."""
    assert build_args(run_cpm, []).measure_step is False
