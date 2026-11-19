"""Stop the policy and send the follower home. No Station, no robot: a fake arm that follows its goals."""

import argparse
import asyncio
import os
import subprocess
import sys
import types

import numpy as np
import pytest

import go_home
import stop_and_home

HOME = {1: 2129, 2: 1034, 3: 1818, 4: 3168, 5: 1888, 6: 2512, 7: 3079, 8: 1298}
START = {1: 2000, 2: 1500, 3: 2200, 4: 2700, 5: 1900, 6: 2300, 7: 3000, 8: 1400}
OPEN = 2400


def test_shipped_home_pose_is_the_one_read_off_station():
    bus, home, _ = go_home.load_home()
    assert bus == "5B3D050113" and home == HOME


def test_open_phase_moves_only_the_gripper():
    target, joints = go_home.phase_target("abrir", START, HOME, OPEN)
    assert joints == [8] and target[8] == OPEN
    assert all(target[m] == START[m] for m in range(1, 8))


def test_arm_phase_keeps_the_gripper_open():
    target, joints = go_home.phase_target("brazo", {**START, 8: OPEN}, HOME, OPEN)
    assert joints == list(range(1, 8)) and target[8] == OPEN
    assert all(target[m] == HOME[m] for m in joints)


def test_last_phase_is_the_whole_home_pose():
    target, joints = go_home.phase_target("cerrar", START, HOME, OPEN)
    assert target == HOME and joints == sorted(HOME)


def test_without_gripper_open_the_gripper_stays_put_until_the_end():
    for phase in ("abrir", "brazo"):
        target, _ = go_home.phase_target(phase, START, HOME, None)
        assert target[8] == START[8]


def test_reached_looks_only_at_the_joints_asked_for():
    pos = {**HOME, 8: HOME[8] + 500}
    assert go_home.reached(pos, HOME, 60, list(range(1, 8)))
    assert not go_home.reached(pos, HOME, 60, sorted(HOME))
    assert go_home.worst(pos, HOME, sorted(HOME)) == (8, 500)


class FakeArm:
    """Stands in for station_io: every motor reaches the goal it was last sent, unless stuck."""

    def __init__(self, start, stuck=()):
        self.pos, self.stuck = dict(start), set(stuck)
        self.sent: list[dict[int, int]] = []
        self.torque = None

    async def new_station_client(self, host, log):
        return self

    async def read_bus_positions(self, client, bus, motor_ids, timeout=5.0):
        return dict(self.pos)

    async def safe_enable_torque(self, client, bus, motor_ids):
        self.torque = True

    def goal_command(self, bus, motor_ids, goals):
        return dict(zip(motor_ids, goals))

    async def send_commands(self, client, cmds):
        for goal in cmds:
            self.sent.append(goal)
            self.pos.update({m: g for m, g in goal.items() if m not in self.stuck})

    async def set_torque(self, client, bus, motor_ids, enable):
        self.torque = enable


def run_home(monkeypatch, tmp_path, arm, gripper_open=OPEN, extra=()):
    pose = tmp_path / "home.json"
    pose.write_text('{"bus_serial": "BUS", "ticks": {%s}, "gripper_open": %s}'
                    % (", ".join(f'"{m}": {t}' for m, t in HOME.items()),
                       "null" if gripper_open is None else gripper_open))
    fake = types.ModuleType("station_io")
    for name in dir(arm):
        if not name.startswith("_"):
            setattr(fake, name, getattr(arm, name))
    monkeypatch.setitem(sys.modules, "station_io", fake)
    monkeypatch.setattr(go_home, "PERIOD_S", 0.0)
    monkeypatch.setattr(go_home, "phase_timeout", lambda *a: 0.5)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(go_home.asyncio, "sleep", lambda s: real_sleep(0))
    p = argparse.ArgumentParser()
    go_home.add_home_args(p)
    return go_home.run_go_home(p.parse_args(["--home-pose", str(pose), *extra]))


def test_goes_home_opening_first_and_releases_torque_there(monkeypatch, tmp_path):
    arm = FakeArm(START)
    assert run_home(monkeypatch, tmp_path, arm) == 0
    assert go_home.reached(arm.pos, HOME, 60, sorted(HOME))
    assert arm.torque is False
    opened = next(i for i, g in enumerate(arm.sent) if abs(g[8] - OPEN) <= 60)
    assert all(g[m] == START[m] for g in arm.sent[:opened] for m in range(1, 8)), "arm moved before the gripper opened"
    arrived = next(i for i, g in enumerate(arm.sent) if all(abs(g[m] - HOME[m]) <= 60 for m in range(1, 8)))
    assert all(g[8] == OPEN for g in arm.sent[opened + 1:arrived]), "gripper closed while the arm was moving"


def test_no_goal_jumps_more_than_the_step_limit(monkeypatch, tmp_path):
    arm = FakeArm(START)
    run_home(monkeypatch, tmp_path, arm)
    path = [START] + arm.sent
    for before, after in zip(path, path[1:]):
        assert all(abs(after[m] - before[m]) <= 40 for m in range(1, 8))
        assert abs(after[8] - before[8]) <= 100


def test_a_blocked_motor_fails_and_keeps_torque(monkeypatch, tmp_path):
    arm = FakeArm(START, stuck={2})
    assert run_home(monkeypatch, tmp_path, arm) == 1
    assert arm.torque is True


def test_check_moves_nothing(monkeypatch, tmp_path):
    arm = FakeArm(START)
    assert run_home(monkeypatch, tmp_path, arm, extra=["--check"]) == 0
    assert arm.sent == [] and arm.torque is None and arm.pos == START


def test_keep_torque_leaves_the_arm_holding_home(monkeypatch, tmp_path):
    arm = FakeArm(START)
    assert run_home(monkeypatch, tmp_path, arm, extra=["--keep-torque"]) == 0
    assert arm.torque is True


def test_station_down_is_a_failure_not_a_crash(monkeypatch, tmp_path):
    arm = FakeArm(START)

    async def boom(*a, **k):
        raise RuntimeError("bus no encontrado")

    arm.read_bus_positions = boom
    assert run_home(monkeypatch, tmp_path, arm) == 1
    assert arm.torque is None


@pytest.fixture
def fake_policy(tmp_path, monkeypatch):
    """A process whose command line looks like the policy and that ignores nothing."""
    script = tmp_path / "run_local.py"
    script.write_text("import time\ntime.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, str(script)])
    monkeypatch.setattr(stop_and_home, "pidfile_path", lambda bus: tmp_path / f"{bus}.pid")
    yield proc, tmp_path
    proc.kill()
    proc.wait()


def test_finds_the_policy_through_its_pidfile(fake_policy):
    proc, tmp = fake_policy
    (tmp / "BUS.pid").write_text(str(proc.pid))
    assert stop_and_home.find_policy_pids("BUS") == [proc.pid]


def test_a_stale_pidfile_pointing_at_something_else_is_not_trusted(fake_policy):
    proc, tmp = fake_policy
    (tmp / "BUS.pid").write_text(str(os.getpid()))      # pytest, not the policy
    assert os.getpid() not in stop_and_home.find_policy_pids("BUS")
    assert proc.pid in stop_and_home.find_policy_pids("BUS")       # found by scanning instead


def test_stop_ends_the_policy_and_a_zombie_counts_as_stopped(fake_policy):
    proc, _ = fake_policy
    stop_and_home.stop([proc.pid], wait_s=5.0)      # not reaped yet: still a zombie of this test
    assert not stop_and_home._alive(proc.pid)
    assert proc.wait(timeout=5) == -15              # SIGTERM was enough
