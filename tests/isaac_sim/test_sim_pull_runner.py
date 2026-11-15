"""The simulation pull policy's runner, without a robot, a camera or onnxruntime: a fake network and a fake force."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

POLICIES = Path(__file__).resolve().parents[2] / "isaac-sim" / "policies"
sys.path.insert(0, str(POLICIES))

from sim_pull_runner import Arm, ForceLost, SimPullRunner, norm_to_rad, rad_to_norm  # noqa: E402

START = [0.0, 0.04, 1.057, 0.072, 0.0, -1.169, 0.0]
META = {"kind": "pull", "rate_hz": 10, "action_scale": 0.06, "cap_n": 1.96, "tip_m": 0.10, "start_rad": START}


class FakeSession:
    """Stands in for the exported network: always answers with the same five numbers, and keeps what it was given."""

    def __init__(self, action):
        self.action, self.seen = np.asarray(action, dtype=np.float32), []

    def get_inputs(self):
        return [type("In", (), {"name": "obs"})()]

    def run(self, _, feed):
        self.seen.append(feed["obs"].copy())
        return [self.action[None]]


def make(tmp_path, action, force, directions=None):
    (tmp_path / "flex_sim.json").write_text(json.dumps(META))
    session = FakeSession(action)
    return SimPullRunner(tmp_path, force, session=session, directions=directions), session


def start_state():
    arm = Arm()
    state = np.full(8, 0.5)
    state[:7] = rad_to_norm(np.array(START), arm.lower, arm.upper, np.ones(7))
    return state, arm


def test_norm_and_radians_round_trip_in_both_directions():
    arm = Arm()
    q = np.array(START)
    for d in (np.ones(7), -np.ones(7), np.array([1, -1, 1, -1, 1, -1, 1.0])):
        assert np.allclose(norm_to_rad(rad_to_norm(q, arm.lower, arm.upper, d), arm.lower, arm.upper, d), q)


def test_start_posture_puts_the_jaws_where_training_had_them():
    pos, pointing = Arm().gripper(np.array(START))
    assert np.allclose(pos + 0.10 * pointing, [0.0, 0.30, 0.05], atol=0.003)  # 0.30 m ahead, 5 cm above the table


def test_zero_action_holds_the_arm_where_it_is(tmp_path):
    runner, session = make(tmp_path, [0, 0, 0, 0, 0], lambda: 0.0)
    state, _ = start_state()
    chunk, _ = runner.predict_from_jpegs(state, {}, "")
    assert chunk.shape == (1, 8)
    assert np.allclose(chunk[0], state, atol=1e-6)
    assert session.seen[0].shape == (1, 29)
    assert np.allclose(session.seen[0][0, :7], 0.0, atol=1e-6)  # at the start posture: nothing to report


def test_only_the_five_driven_motors_move(tmp_path):
    runner, _ = make(tmp_path, [1, 1, 1, 1, 1], lambda: 0.0)
    state, arm = start_state()
    chunk, _ = runner.predict_from_jpegs(state, {}, "")
    moved = np.abs(chunk[0] - state) > 1e-6
    assert moved.tolist() == [True, True, True, True, False, True, False, False]  # motors 1, 2, 3, 4, 6
    step = 0.06 / (arm.upper - arm.lower)  # one full action is 0.06 rad
    assert np.allclose((chunk[0] - state)[[0, 1, 2, 3, 5]], step[[0, 1, 2, 3, 5]], atol=1e-5)


def test_a_reversed_motor_moves_the_other_way(tmp_path):
    runner, _ = make(tmp_path, [1, 0, 0, 0, 0], lambda: 0.0, directions=[-1, 1, 1, 1, 1, 1, 1])
    arm = Arm()
    state = np.full(8, 0.5)
    state[:7] = rad_to_norm(np.array(START), arm.lower, arm.upper, runner.directions)
    chunk, _ = runner.predict_from_jpegs(state, {}, "")
    assert chunk[0, 0] < state[0]


def test_the_force_reaches_the_network_in_units_of_the_limit(tmp_path):
    readings = iter([0.0, 0.98])
    runner, session = make(tmp_path, [0, 0, 0, 0, 0], lambda: next(readings))
    state, _ = start_state()
    runner.predict_from_jpegs(state, {}, "")
    runner.predict_from_jpegs(state, {}, "")
    assert session.seen[1][0, 19] == pytest.approx(0.0)  # the force one frame ago
    assert session.seen[1][0, 20] == pytest.approx(0.5)  # the rise since then: 0.98 N of a 1.96 N limit


def test_a_target_never_runs_far_ahead_of_the_arm(tmp_path):
    runner, _ = make(tmp_path, [1, 0, 0, 0, 0], lambda: 0.0)
    state, arm = start_state()
    for _ in range(40):  # the arm does not follow: the state never changes
        chunk, _ = runner.predict_from_jpegs(state, {}, "")
    lead = (chunk[0, 0] - state[0]) * (arm.upper[0] - arm.lower[0])
    assert lead <= 0.6 + 1e-6


def test_no_force_reading_stops_instead_of_guessing(tmp_path):
    runner, _ = make(tmp_path, [0, 0, 0, 0, 0], lambda: None)
    state, _ = start_state()
    with pytest.raises(ForceLost):
        runner.predict_from_jpegs(state, {}, "")
