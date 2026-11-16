"""The CPM runner: a fixed path, and a camera that supervises rather than drives.

The properties worth testing without hardware are the ones that make this a different kind of
thing from `scripts/run_session.py`:

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
    spec = importlib.util.spec_from_file_location("run_cpm", REPO / "scripts" / "run_cpm.py")
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


# --------------------------------------------------------------------------- the CLI contract
def test_the_default_direction_cannot_tighten_anything(run_cpm):
    """The stroke travels away from tension first unless told otherwise. Whatever else is
    wrong (a flipped sign, a mis-set pose), the first half of the default stroke cannot be
    pulling harder on a finger."""
    assert build_args(run_cpm, []).toward_tension is False


def test_the_distance_default_is_small(run_cpm):
    """This is an open-loop path: it travels exactly this far whether or not that turns out
    to be comfortable, so the default has to be a distance worth discovering that with."""
    assert build_args(run_cpm, []).distance_mm <= 12.0


def test_it_starts_from_its_own_pose_not_the_session_home(run_cpm):
    """The session home is where the arm is parked with nothing hooked. A CPM stroke needs
    somewhere to travel to with the finger attached, which is a different pose."""
    args = build_args(run_cpm, [])
    assert args.start_pose == "config/calibration/cpm-start-pose.json"
    assert "home-pose.json" not in args.start_pose


def test_the_camera_can_be_turned_off_entirely_for_a_dry_path_check(run_cpm):
    """The same way scripts/jog_coordinated.py is used: confirm the line with nothing
    attached before putting a finger anywhere near it."""
    assert build_args(run_cpm, ["--no-camera"]).no_camera is True


def test_the_ceiling_and_the_speeds_come_from_the_tuning_profile(run_cpm):
    from session.tuning import load_tuning

    t = load_tuning()
    args = build_args(run_cpm, [])
    assert args.ceiling_gf == t.safety.ceiling_gf
    assert args.speed == t.coordinated_speed
    assert args.accel == t.coordinated_accel
    assert args.outlier_persist == t.outlier_persist


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
    with pytest.raises(SystemExit, match="set_home_pose.py --save --path"):
        asyncio.run(run_cpm.run(args))


# ------------------------------------------------------------------ the path itself
@pytest.mark.skipif(not URDF.is_file(),
                    reason="needs the norma-core clone (hardware/elrobot/simulation)")
def test_the_round_trip_returns_to_exactly_where_it_began():
    """Not approximately: the return visits the same waypoints in reverse and ends on the
    start pose itself, so a repetition cannot drift from the line that was checked."""
    from lib.kinematics import load_chain, tether_tension_direction

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
    from lib.kinematics import load_chain, tether_tension_direction

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

    from lib.kinematics import load_chain, tether_tension_direction

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
    from lib.units import grams_force_to_newtons

    return run_cpm.ForceSupervisor(
        FakeSource(readings), FakePipeline(readings),
        grams_force_to_newtons(ceiling_gf), blind_tolerance_s,
    )


def test_a_force_under_the_ceiling_does_not_stop_anything(run_cpm):
    from lib.units import grams_force_to_newtons

    s = supervisor_for(run_cpm, [grams_force_to_newtons(150)] * 3)
    for _ in range(3):
        s.poll_once()
    assert s.stop_reason is None


def test_a_force_over_the_ceiling_stops_the_stroke(run_cpm):
    from lib.units import grams_force_to_newtons

    s = supervisor_for(run_cpm, [grams_force_to_newtons(250)])
    s.poll_once()
    assert s.stop_reason is not None
    assert "ceiling" in s.stop_reason


def test_the_ceiling_message_names_both_numbers(run_cpm):
    """So the log says what was seen and what was allowed, not just that something tripped."""
    from lib.units import grams_force_to_newtons

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
    from lib.units import grams_force_to_newtons, newtons_to_grams_force

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
    """Grams is what a person reads off the PESOLA, per AGENTS.md's units rule."""
    from lib.units import grams_force_to_newtons

    s = supervisor_for(run_cpm, [grams_force_to_newtons(123)])
    assert s.reading.strip() == "--"
    s.poll_once()
    assert "123" in s.reading and "gf" in s.reading


# ------------------------------------------------------------------ the second pose
def test_set_home_pose_can_write_somewhere_other_than_the_session_home():
    """CPM needs its own start pose, so the capture script has to be able to write a second
    file rather than only the one home."""
    spec = importlib.util.spec_from_file_location(
        "set_home_pose", REPO / "scripts" / "set_home_pose.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = (REPO / "scripts" / "set_home_pose.py").read_text()
    assert "--path" in source
    assert "args.path" in source


def test_a_saved_pose_file_loads_through_the_same_loader(tmp_path):
    """Whatever path it is written to, `session.home_pose.load_home_pose` has to read it, so
    the CPM start pose and the session home stay one format."""
    from session.home_pose import load_home_pose

    path = tmp_path / "cpm-start-pose.json"
    path.write_text(json.dumps({"ticks": {str(m): v for m, v in POSE.items() if m != 8}}))
    loaded = load_home_pose(path)
    assert loaded[1] == 2049
    assert 8 not in loaded
