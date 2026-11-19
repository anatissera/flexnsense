"""The first bridge script's start-up checks, without a robot, a camera, Station or onnxruntime: fakes and a fake clock."""

import asyncio
import sys
from pathlib import Path

import pytest

POLICIES = Path(__file__).resolve().parents[2] / "isaac-sim" / "policies"
sys.path.insert(0, str(POLICIES))

import run_pull_on_robot as bridge  # noqa: E402


class FakeClock:
    """Time that only moves when the code under test sleeps."""

    def __init__(self):
        self.now, self.sleeps = 100.0, 0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.sleeps += 1


class FakeForce:
    """Stands in for CameraForce: a list of answers (the last one repeats), a reason, and a background error."""

    def __init__(self, answers, reason="the scale was not found in the picture (no line)", error=None, die_after=None):
        self.answers, self.reason, self.error, self.calls, self.die_after = list(answers), reason, error, 0, die_after

    def __call__(self):
        self.calls += 1
        if self.die_after is not None and self.calls > self.die_after:
            self.error = "the scale camera stopped giving pictures"
        if self.error:
            return None
        return self.answers[min(self.calls - 1, len(self.answers) - 1)]

    def why(self) -> str:
        return self.error or self.reason


def run(coro):
    return asyncio.run(coro)


# ─────────────────────────────── importing needs nothing heavy ─────────────────────────────────────────

def test_importing_the_script_loads_no_hardware_or_network_package():
    import subprocess

    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import run_pull_on_robot; "
        "bad = [m for m in ('onnxruntime', 'websockets', 'cv2', 'arm1.lib.station', 'gauge_reader', 'sim_pull_runner') if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code, str(POLICIES)], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_script_finds_the_repository_root():
    assert (bridge.REPO / "arm1" / "lib" / "station.py").exists() and (bridge.REPO / "arm1" / "gauge_reader").is_dir()
    assert str(bridge.REPO) in sys.path and str(POLICIES) in sys.path


# ─────────────────────────────── which motors are missing ──────────────────────────────────────────────

def test_seven_entries_with_the_gripper_is_not_all_seven_arm_motors():
    ticks = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0, 8: 0}  # the case of issue 44: seven entries, motor 6 missing
    assert len(ticks) == 7
    assert bridge.missing_motors(ticks) == [6]


def test_no_motor_missing_when_all_seven_reported():
    assert bridge.missing_motors({m: 2048 for m in range(1, 8)}) == []
    assert bridge.missing_motors({m: 2048 for m in range(1, 9)}) == []
    assert bridge.missing_motors({}) == [1, 2, 3, 4, 5, 6, 7]


def test_wait_for_joints_returns_as_soon_as_every_arm_motor_reported():
    clock, state = FakeClock(), {"ticks": {8: 0}}

    async def sleep(seconds):
        await clock.sleep(seconds)
        state["ticks"][clock.sleeps] = 2048  # one more motor at every poll: 1, 2, ... 7

    assert run(bridge.wait_for_joints(state, timeout_s=10.0, clock=clock, sleep=sleep)) == []
    assert clock.sleeps == 7


def test_wait_for_joints_gives_up_and_names_the_missing_motor():
    clock = FakeClock()
    state = {"ticks": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0, 8: 0}}
    missing = run(bridge.wait_for_joints(state, timeout_s=10.0, clock=clock, sleep=clock.sleep))
    assert missing == [6]
    assert 10.0 <= clock.now - 100.0 < 10.2


def test_the_timeout_message_says_what_was_seen_and_what_is_missing():
    text = bridge.joints_timeout_message({1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0, 8: 0}, "5B3D050113")
    assert "seen:    [1, 2, 3, 4, 5, 7, 8]" in text
    assert "missing: [6]" in text
    assert "5B3D050113" in text and "not answering on the bus" in text
    assert "Nothing was sent" in text and "Traceback" not in text


def test_the_timeout_message_points_at_the_bus_serial_when_no_motor_was_seen():
    text = bridge.joints_timeout_message({}, "WRONG")
    assert "seen:    none" in text and "missing: [1, 2, 3, 4, 5, 6, 7]" in text
    assert "--bus" in text and "FOLLOWER_BUS_SERIAL" in text


# ─────────────────────────────── the first reading of the scale ────────────────────────────────────────

def test_wait_for_force_returns_the_first_reading():
    clock, force = FakeClock(), FakeForce([None, None, 0.42])
    assert run(bridge.wait_for_force(force, timeout_s=15.0, clock=clock, sleep=clock.sleep)) == pytest.approx(0.42)
    assert clock.sleeps == 2


def test_a_reading_of_zero_newtons_is_a_reading():
    clock, force = FakeClock(), FakeForce([0.0])
    assert run(bridge.wait_for_force(force, clock=clock, sleep=clock.sleep)) == 0.0
    assert clock.sleeps == 0


def test_wait_for_force_gives_up_after_the_timeout_with_the_real_cause():
    clock, force = FakeClock(), FakeForce([None])
    assert run(bridge.wait_for_force(force, timeout_s=15.0, clock=clock, sleep=clock.sleep)) is None
    assert 15.0 <= clock.now - 100.0 < 15.2
    text = bridge.no_force_message(force)
    assert "the scale was not found in the picture (no line)" in text and "Nothing was sent" in text


def test_wait_for_force_stops_at_once_when_the_reading_died():
    clock, force = FakeClock(), FakeForce([None], die_after=3)
    assert run(bridge.wait_for_force(force, timeout_s=15.0, clock=clock, sleep=clock.sleep)) is None
    assert clock.now - 100.0 < 1.0  # it did not sit out the 15 s
    assert "the scale camera stopped giving pictures" in bridge.no_force_message(force)


def test_a_dead_reading_never_counts_as_a_force():
    force = FakeForce([1.0], error="SystemExit: 2")
    clock = FakeClock()
    assert run(bridge.wait_for_force(force, clock=clock, sleep=clock.sleep)) is None


# ─────────────────────────────── the command line ──────────────────────────────────────────────────────

def args_with(tmp_path, *extra):
    return ["--policy", "p.onnx", "--joint_map", "j.json", *extra]


def test_the_calibration_must_exist_when_the_arguments_are_read(tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        bridge.parse_args(args_with(tmp_path, "--camera", "0", "--calib", str(tmp_path / "nope.json")))
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert "no calibration at" in err and "nope.json" in err and "gauge_reader calibrate" in err


def test_camera_is_required(tmp_path, capsys):
    calib = tmp_path / "scale_calib.json"
    calib.write_text("{}")
    with pytest.raises(SystemExit):
        bridge.parse_args(args_with(tmp_path, "--calib", str(calib)))
    assert "--camera" in capsys.readouterr().err


def test_arguments_with_a_calibration_and_a_camera(tmp_path):
    calib = tmp_path / "scale_calib.json"
    calib.write_text("{}")
    args = bridge.parse_args(args_with(tmp_path, "--camera", "iphone", "--calib", str(calib)))
    assert args.camera == "iphone" and args.calib == str(calib)
    assert args.send is False  # a dry run unless --send is given
    assert args.seconds == 20.0 and args.bus is None


def test_force_url_is_gone(tmp_path, capsys):
    calib = tmp_path / "scale_calib.json"
    calib.write_text("{}")
    with pytest.raises(SystemExit):
        bridge.parse_args(args_with(tmp_path, "--camera", "0", "--calib", str(calib), "--force_url", "ws://x"))
    assert "force_url" in capsys.readouterr().err


def test_the_default_calibration_is_the_readers_own_file():
    assert bridge.DEFAULT_CALIB == "config/scale_calib.json"
