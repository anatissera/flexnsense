"""The URDF chain, and the two conventions that let a tick reading become geometry.

These matter because `docs/elrobot.md` says the id-to-joint mapping is unverified and must not
be assumed. Everything here either checks the URDF against this physical arm's measured
calibration, or against what the arm was observed to do by hand.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from lib.kinematics import (
    CENTRE_TICKS,
    TICKS_PER_TURN,
    check_against_calibration,
    load_chain,
    radians_to_ticks,
    ticks_to_radians,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
URDF = REPO_ROOT / "norma-core/hardware/elrobot/simulation/elrobot_follower.urdf"
CALIB = REPO_ROOT / "config/calibration/follower-5B8E114979.json"

#: The pose read off Station's web UI on 2026-11-15, used for the cross-check against what the
#: arm was observed to do by hand.
OBSERVED_POSE = {1: 1976, 2: 1719, 3: 2526, 4: 2755, 5: 1907, 6: 1270, 7: 1282}

#: The smallest set of joints that can follow an arbitrary 3D direction at this pose: three
#: independent ones. Fewer spans only a plane, which fails on the return stroke.
PULL_MOTORS = [2, 3, 4]

pytestmark = pytest.mark.skipif(
    not URDF.is_file(),
    reason="needs the norma-core clone (hardware/elrobot/simulation); see SETUP.md",
)


@pytest.fixture(scope="module")
def chain():
    return load_chain(URDF)


# ---------------------------------------------------------------- the tick convention
def test_the_centre_tick_is_the_urdf_zero():
    assert ticks_to_radians(CENTRE_TICKS) == 0.0


def test_a_full_turn_of_ticks_is_two_pi():
    assert ticks_to_radians(CENTRE_TICKS + TICKS_PER_TURN) == pytest.approx(2 * np.pi)


def test_ticks_and_radians_round_trip():
    for t in (0, 512, 2048, 3000, 4095):
        assert radians_to_ticks(ticks_to_radians(t)) == pytest.approx(t)


@pytest.mark.skipif(not CALIB.is_file(), reason="needs the committed follower calibration")
def test_every_calibrated_range_is_centred_on_the_zero_tick():
    """What makes CENTRE_TICKS a measured fact rather than a guess: the hand-made calibration
    of all eight motors is centred there, so that is where the mechanism's zero is."""
    ranges = json.loads(CALIB.read_text())["ranges"]
    for motor, (lo, hi) in ranges.items():
        assert (lo + hi) / 2 == pytest.approx(CENTRE_TICKS, abs=1), f"motor {motor}"


# ---------------------------------------------------------------- the id-to-joint mapping
@pytest.mark.skipif(not CALIB.is_file(), reason="needs the committed follower calibration")
def test_the_urdf_limits_match_this_arms_measured_ranges(chain):
    """The check that turns 'rev_motor_0N is Station's motor N' into a verified fact.

    Motor 8 is excluded: the gripper's calibrated range is set by where its jaws close, not by
    the mechanism's travel, so it legitimately differs.
    """
    ranges = json.loads(CALIB.read_text())["ranges"]
    for motor_id, measured, urdf_span, agrees in check_against_calibration(chain, ranges):
        if motor_id == 8:
            continue
        assert agrees, (
            f"motor {motor_id}: firmware says {measured:.0f} deg, URDF says {urdf_span:.0f} deg. "
            "Either the mapping is not what we think, or this arm was recalibrated"
        )


def test_the_chain_is_the_eight_motors_in_order(chain):
    assert chain.motor_ids == [1, 2, 3, 4, 5, 6, 7, 8]


# ---------------------------------------------------------------- physical sanity
def test_the_reach_matches_the_documented_arm(chain):
    """An independent check that the chain is assembled correctly: docs/elrobot.md puts the
    ElRobot's reach at about 430 mm, which nothing in the URDF parsing knows about."""
    rng = np.random.default_rng(0)
    reach = max(
        float(np.linalg.norm(chain.tip_position(
            {m: CENTRE_TICKS + rng.integers(-1000, 1000) for m in chain.motor_ids})))
        for _ in range(2000)
    )
    assert 0.35 < reach < 0.50, f"reach came out {reach*1000:.0f} mm"


def test_every_direction_is_a_unit_vector(chain):
    for motor_id in (1, 2, 3, 4, 5, 6):
        d = chain.tip_velocity_direction(motor_id, OBSERVED_POSE)
        assert np.linalg.norm(d) == pytest.approx(1.0)


def test_the_exact_jacobian_agrees_with_a_finite_difference(chain):
    """`tip_velocity_direction` uses omega x r rather than perturbing the pose. This checks the
    two against each other, so an error in the transform chain cannot hide in either."""
    for motor_id in (1, 2, 3, 4, 5, 6):
        analytic = chain.tip_velocity_direction(motor_id, OBSERVED_POSE)
        before = chain.tip_position(OBSERVED_POSE)
        nudged = dict(OBSERVED_POSE)
        nudged[motor_id] += 1
        numeric = chain.tip_position(nudged) - before
        numeric = numeric / np.linalg.norm(numeric)
        assert np.dot(analytic, numeric) == pytest.approx(1.0, abs=2e-3), f"motor {motor_id}"


# ---------------------------------------------------------------- against what the arm did
def test_motor_7_orients_the_gripper_without_moving_it(chain):
    """Observed by hand as "it only tilts the tip, it does not move the whole arm". The URDF
    explains it: motor 7's axis passes through the gripper origin, so it rotates that point
    rather than translating it. This is the strongest single confirmation that the URDF
    describes the arm we actually have."""
    assert chain.tip_speed_per_tick(7, OBSERVED_POSE) * 1000 < 0.01   # under 10 microns per tick


def test_motors_1_and_5_move_the_gripper_along_the_same_line(chain):
    """Both were observed as "left to right". The URDF agrees: their directions are nearly
    parallel, so they are redundant for the tether's purposes."""
    d1 = chain.tip_velocity_direction(1, OBSERVED_POSE)
    d5 = chain.tip_velocity_direction(5, OBSERVED_POSE)
    assert abs(float(np.dot(d1, d5))) > 0.97


def test_motors_2_and_6_move_the_gripper_along_the_same_line(chain):
    """Both observed as "up and down"."""
    d2 = chain.tip_velocity_direction(2, OBSERVED_POSE)
    d6 = chain.tip_velocity_direction(6, OBSERVED_POSE)
    assert abs(float(np.dot(d2, d6))) > 0.97


def test_motor_4_is_the_most_forward_of_the_arm_joints(chain):
    """Observed as "like 3 but moving slightly forward". In the URDF's base frame that is the
    y component, and motor 4 has the largest of any arm joint, which is why it is the candidate
    when the tether runs forward rather than across."""
    ys = {m: abs(chain.tip_velocity_direction(m, OBSERVED_POSE)[1]) for m in (1, 2, 3, 4, 5, 6)}
    assert max(ys, key=ys.get) == 4


# ---------------------------------------------------------------- the bench cross-check
def test_speed_per_tick_is_in_the_range_the_bench_procedure_expects(chain):
    """`ticks_per_metre` is measured on the bench by commanding a tick delta and reading a
    ruler; this is the same number from geometry. They should agree to the extent the pose is
    known, so a wild disagreement means one of them is wrong."""
    for motor_id in (1, 2, 3, 4, 5, 6):
        mm_per_100 = chain.tip_speed_per_tick(motor_id, OBSERVED_POSE) * 100 * 1000
        assert 1.0 < mm_per_100 < 100.0, f"motor {motor_id}: {mm_per_100:.1f} mm per 100 ticks"


def test_a_missing_motor_reading_is_treated_as_centred(chain):
    """A partial reading still gives useful geometry rather than refusing: a quiet motor on the
    bus should not stop the question being answered."""
    partial = {1: 1976, 2: 1719}
    assert chain.tip_position(partial) is not None
    full = dict.fromkeys(chain.motor_ids, CENTRE_TICKS)
    full.update(partial)
    assert np.allclose(chain.tip_position(partial), chain.tip_position(full))


# ---------------------------------------------------------------- straight-line pulls
def test_one_joint_alone_cannot_pull_straight(chain):
    """The finding that made inverse kinematics necessary. A revolute joint moves the tip along
    an arc; over the 72 to 88 mm one full-range repetition needs (config/calibration/bench.md)
    that arc leaves the tether's axis by more than 20 mm. The tether does not bend, so that is
    either side-load on the finger or slack."""
    direction = chain.tip_velocity_direction(4, OBSERVED_POSE)
    worst, _ = chain.line_error([4], OBSERVED_POSE, direction, 0.090, steps=60)
    assert worst > 0.015, f"only {worst*1000:.1f} mm off; the single-joint problem has changed"


def test_coordinating_three_joints_pulls_straight(chain):
    """And the fix. Same pull, same distance, three joints solved together."""
    direction = chain.tip_velocity_direction(4, OBSERVED_POSE)
    worst, shortfall = chain.line_error(PULL_MOTORS, OBSERVED_POSE, direction, 0.090, steps=60)
    assert worst < 0.002, f"deviation {worst*1000:.2f} mm"
    assert abs(shortfall) < 0.002, f"shortfall {shortfall*1000:.2f} mm"


def test_two_joints_span_only_a_plane(chain):
    """Why PULL_MOTORS has three entries and not two. Two revolute joints give a rank-2
    jacobian, so the reachable directions are a plane: a pull that happens to lie in it works,
    and the return stroke, which does not, misses by millimetres. Three independent joints are
    the minimum that can follow an arbitrary 3D direction."""
    assert np.linalg.matrix_rank(chain.jacobian([3, 4], OBSERVED_POSE), tol=1e-6) == 2
    assert np.linalg.matrix_rank(chain.jacobian(PULL_MOTORS, OBSERVED_POSE), tol=1e-6) == 3

    direction = chain.tip_velocity_direction(4, OBSERVED_POSE)
    back_two, _ = chain.line_error([3, 4], OBSERVED_POSE, -direction, 0.090, steps=60)
    back_three, _ = chain.line_error(PULL_MOTORS, OBSERVED_POSE, -direction, 0.090, steps=60)
    assert back_two > 0.003, f"two joints managed {back_two*1000:.2f} mm; has the pose changed?"
    assert back_three < 0.002, f"three joints: {back_three*1000:.2f} mm"


def test_more_joints_do_not_make_it_worse(chain):
    """A redundant arm has many solutions; the damped least-squares pick should stay sane as
    joints are added rather than wandering off."""
    direction = chain.tip_velocity_direction(4, OBSERVED_POSE)
    for motors in (PULL_MOTORS, [2, 3, 4, 6], [1, 2, 3, 4, 5, 6, 7]):
        worst, _ = chain.line_error(motors, OBSERVED_POSE, direction, 0.090, steps=60)
        assert worst < 0.003, f"{motors}: {worst*1000:.2f} mm"


def test_the_jacobian_columns_are_the_single_joint_directions(chain):
    """The matrix is just those vectors side by side, so an assembly mistake shows up here."""
    motors = [2, 3, 4]
    J = chain.jacobian(motors, OBSERVED_POSE)
    assert J.shape == (3, 3)
    for i, m in enumerate(motors):
        col = J[:, i] / np.linalg.norm(J[:, i])
        assert np.allclose(col, chain.tip_velocity_direction(m, OBSERVED_POSE), atol=1e-9)


def test_damping_trades_accuracy_for_bounded_joint_motion(chain):
    """The trade `ik_step` is deliberately making, and why one step is not the thing to judge.

    This arm is redundant and at this pose the jacobian is badly conditioned: its smallest
    singular value is about 1e-4 against a largest of 0.24, so one direction is nearly
    unreachable. Chasing it exactly is possible but asks for radians of joint motion to produce
    a millimetre at the tip, which next to a healing finger is the dangerous answer, not the
    clever one. Damping suppresses that direction, at the cost of a residual in it.

    What matters is the finished path, not the single step: `line_trajectory` re-solves at each
    waypoint, and `test_coordinating_three_joints_pulls_straight` shows the result lands within
    a millimetre over 90 mm.
    """
    want = np.array([0.0, -1.0, -1.0]) / np.sqrt(2) * 0.001
    J = chain.jacobian(PULL_MOTORS, OBSERVED_POSE)
    assert np.linalg.cond(J) > 100, "pose is better conditioned than it was; re-tune this test"

    damped = chain.ik_step(PULL_MOTORS, OBSERVED_POSE, want)
    greedy = chain.ik_step(PULL_MOTORS, OBSERVED_POSE, want, damping=1e-7)
    biggest_damped = max(abs(v) for v in damped.values())
    biggest_greedy = max(abs(v) for v in greedy.values())
    assert biggest_damped < 50, f"a 1 mm step asked for {biggest_damped:.0f} ticks"
    assert biggest_greedy > 10 * biggest_damped, (
        "undamped should be far more aggressive; if it is not, the pose is well conditioned "
        "and this test is no longer exercising the trade-off"
    )


def test_an_ik_step_moves_the_tip_roughly_where_asked(chain):
    """The step is still a real solve, checked against the forward kinematics: most of the
    requested motion arrives, and what is missing lies in the suppressed direction."""
    want = np.array([0.0, -1.0, -1.0]) / np.sqrt(2) * 0.001
    dq = chain.ik_step(PULL_MOTORS, OBSERVED_POSE, want)
    moved = dict(OBSERVED_POSE)
    for m, d in dq.items():
        moved[m] += d
    got = chain.tip_position(moved) - chain.tip_position(OBSERVED_POSE)
    along = float(np.dot(got, want / np.linalg.norm(want)))
    assert along > 0.9 * float(np.linalg.norm(want)), "most of the step should arrive"


def test_pulling_backwards_retraces_the_same_line(chain):
    """The reset beat of a repetition runs the pull in reverse, so it has to be as straight."""
    direction = -chain.tip_velocity_direction(4, OBSERVED_POSE)
    worst, _ = chain.line_error(PULL_MOTORS, OBSERVED_POSE, direction, 0.090, steps=60)
    assert worst < 0.002


def test_two_joints_cannot_reach_out_of_their_plane(chain):
    """The consequence of rank 2, stated as geometry rather than as a numerical accident.

    An earlier version of this test asserted that the undamped solve raises or blows up. It
    does neither: numpy returns the minimum-norm least-squares answer, which looks perfectly
    reasonable. The real and reliable property is that a rank-2 jacobian simply cannot produce
    motion outside its plane, however the linear algebra is done, which is why `PULL_MOTORS`
    has three entries.
    """
    J = chain.jacobian([3, 4], OBSERVED_POSE)
    assert np.linalg.matrix_rank(J, tol=1e-6) == 2

    # The direction those two joints cannot reach: the normal to the plane they span.
    _, _, vt = np.linalg.svd(J.T)
    unreachable = vt[-1]

    dq = chain.ik_step([3, 4], OBSERVED_POSE, unreachable * 0.001)
    moved = dict(OBSERVED_POSE)
    for m, d in dq.items():
        moved[m] += d
    got = chain.tip_position(moved) - chain.tip_position(OBSERVED_POSE)
    assert float(np.linalg.norm(got)) < 1e-5, (
        "two joints produced motion along their own null direction, which is impossible; "
        "the jacobian is wrong"
    )


def test_a_trajectory_has_one_waypoint_per_step(chain):
    direction = chain.tip_velocity_direction(4, OBSERVED_POSE)
    way = chain.line_trajectory(PULL_MOTORS, OBSERVED_POSE, direction, 0.090, steps=30)
    assert len(way) == 30
    assert all(set(w) >= set(PULL_MOTORS) for w in way)


def test_zero_steps_is_rejected(chain):
    with pytest.raises(ValueError):
        chain.line_trajectory(PULL_MOTORS, OBSERVED_POSE, np.array([0, 1, 0.0]), 0.09, steps=0)
