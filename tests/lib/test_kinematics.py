"""The URDF chain, and the two conventions that let a tick reading become geometry.

These matter because `docs/elrobot.md` says the id-to-joint mapping is unverified and must not
be assumed. Everything here either checks the URDF against this physical arm's measured
calibration, or against what the arm was observed to do by hand.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from arm1.lib.kinematics import (
    CENTRE_TICKS,
    TICKS_PER_TURN,
    check_against_calibration,
    load_chain,
    load_tick_ranges,
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


# ---------------------------------------------------------------- choosing joints per pose
CENTRED_POSE = {m: 2048 for m in range(1, 8)}
FOLDED_POSE = {1: 2500, 2: 1500, 3: 2800, 4: 1800, 5: 2200, 6: 1600, 7: 2400}


def test_the_chosen_set_always_spans_three_dimensions(chain):
    """The property that makes automatic selection safe: whatever pose it is handed, the
    joints it picks can follow an arbitrary direction, including the return stroke."""
    for pose in (OBSERVED_POSE, CENTRED_POSE, FOLDED_POSE):
        motors = chain.choose_pull_motors(pose)
        assert len(motors) == 3
        rank, _ = chain.pull_quality(motors, pose, np.array([0.0, -1.0, 0.0]))
        assert rank == 3, f"{motors} at {pose} has rank {rank}"


def test_different_poses_can_want_different_joints(chain):
    """Why the set is not a constant in a config file."""
    picks = {tuple(chain.choose_pull_motors(p))
             for p in (OBSERVED_POSE, CENTRED_POSE, FOLDED_POSE)}
    assert len(picks) > 1, "every pose chose the same joints; the selection may be ignoring pose"


def test_the_chosen_set_tracks_both_strokes(chain):
    """Out and back, since a repetition runs both and an earlier greedy selection passed the
    outward stroke at 0.4 mm while missing the return by 14.8 mm."""
    for pose in (OBSERVED_POSE, FOLDED_POSE):
        motors = chain.choose_pull_motors(pose)
        direction = chain.tip_velocity_direction(4, pose)
        out, _ = chain.line_error(motors, pose, direction, 0.090, steps=60)
        back, _ = chain.line_error(motors, pose, -direction, 0.090, steps=60)
        assert out < 0.002, f"{pose}: outward {out*1000:.2f} mm"
        assert back < 0.002, f"{pose}: return {back*1000:.2f} mm"


def test_selection_beats_naming_joints_by_hand(chain):
    """The automatic pick should be at least as well conditioned as the hand-chosen [2, 3, 4],
    otherwise there is no reason to prefer it."""
    for pose in (OBSERVED_POSE, CENTRED_POSE, FOLDED_POSE):
        auto = chain.choose_pull_motors(pose)
        _, cond_auto = chain.pull_quality(auto, pose, np.array([0.0, -1.0, 0.0]))
        _, cond_hand = chain.pull_quality([2, 3, 4], pose, np.array([0.0, -1.0, 0.0]))
        assert cond_auto <= cond_hand * 1.05, f"{pose}: auto {cond_auto:.0f} vs hand {cond_hand:.0f}"


# ---------------------------------------------------------------- refusing an impossible pull
def test_a_pull_that_degrades_into_a_singularity_is_refused(chain):
    """The case that motivated checking the whole path rather than the starting pose.

    This direction starts at a healthy 82 mm per radian and falls to about 1 mm per radian
    partway through a 90 mm pull, with the path wandering 8.6 mm off the line. Checking only
    the start pose called it fine.
    """
    direction = chain.tip_velocity_direction(4, CENTRED_POSE)
    sideways = np.cross(direction, [0.0, 0.0, 1.0])
    sideways = sideways / np.linalg.norm(sideways)
    motors = chain.choose_pull_motors(CENTRED_POSE)

    deviation, _ = chain.line_error(motors, CENTRED_POSE, sideways, 0.090, steps=60)
    assert deviation > 0.005, "the hard direction got easier; this test needs a new one"

    ok, why = chain.can_pull_straight(motors, CENTRED_POSE, sideways, 0.090)
    assert not ok
    assert "singularity" in why or "wanders" in why


def test_a_good_pull_is_accepted_with_its_numbers(chain):
    direction = chain.tip_velocity_direction(4, OBSERVED_POSE)
    motors = chain.choose_pull_motors(OBSERVED_POSE)
    ok, why = chain.can_pull_straight(motors, OBSERVED_POSE, direction, 0.090)
    assert ok, why
    assert "mm per radian" in why


def test_fewer_than_three_joints_is_refused_before_any_geometry(chain):
    ok, why = chain.can_pull_straight([3, 4], OBSERVED_POSE,
                                      np.array([0.0, -1.0, 0.0]), 0.090)
    assert not ok
    assert "plane" in why


def test_a_shorter_pull_can_be_possible_where_a_long_one_is_not(chain):
    """The refusal is about the path, so shortening it is one of the suggested remedies and
    has to actually work."""
    direction = chain.tip_velocity_direction(4, CENTRED_POSE)
    sideways = np.cross(direction, [0.0, 0.0, 1.0])
    sideways = sideways / np.linalg.norm(sideways)
    motors = chain.choose_pull_motors(CENTRED_POSE)
    assert not chain.can_pull_straight(motors, CENTRED_POSE, sideways, 0.090)[0]
    assert chain.can_pull_straight(motors, CENTRED_POSE, sideways, 0.010)[0]


# ---------------------------------------------------------------- the tether's own line
#: The pose the camera measurement of 2026-11-15 ran in, and the yield sign it measured for
#: motor 4: negative ticks lowered the force. The cross-check below is the only independent
#: evidence this repo has for which way out of the gripper the tether runs.
MEASURED_POSE = {1: 2049, 2: 1469, 3: 2468, 4: 2772, 5: 1879, 6: 1525, 7: 3275}
MEASURED_YIELD_SIGN_MOTOR_4 = -1


def test_the_gripper_axis_is_a_unit_vector(chain):
    assert np.isclose(np.linalg.norm(chain.gripper_axis(OBSERVED_POSE)), 1.0)


def test_the_gripper_axis_points_along_the_forearm(chain):
    """Which of the tip link's three local axes points out of the jaws is a URDF convention,
    so it is pinned against something computed independently: the vector from the wrist joint
    to the tip. They agree in every pose, including folded and fully extended ones, which the
    other two columns do not."""
    poses = [OBSERVED_POSE, MEASURED_POSE, CENTRED_POSE,
             {1: 1500, 2: 2500, 3: 1600, 4: 2300, 5: 2048, 6: 2600, 7: 2048},
             {1: 2048, 2: 1200, 3: 2900, 4: 2900, 5: 2048, 6: 1200, 7: 2048}]
    for pose in poses:
        forearm = chain.tip_position(pose) - chain.joint_origin(6, pose)[1]
        forearm = forearm / np.linalg.norm(forearm)
        assert float(np.dot(chain.gripper_axis(pose), forearm)) > 0.9, pose


def test_the_gripper_axis_turns_with_the_arm(chain):
    """It is the gripper's own direction, not a fixed direction in the room. If it did not
    move with the arm it would be useless as the tether's axis."""
    turned = {**OBSERVED_POSE, 1: OBSERVED_POSE[1] + 500}
    assert float(np.dot(chain.gripper_axis(OBSERVED_POSE),
                        chain.gripper_axis(turned))) < 0.95


def test_the_tension_direction_opposes_the_gripper_axis(chain):
    """The finger sits out along the gripper's axis, so travelling that way closes the gap and
    slackens the line. Tension increases on the way back."""
    from arm1.lib.kinematics import tether_tension_direction

    assert np.allclose(tether_tension_direction(chain, OBSERVED_POSE),
                       -chain.gripper_axis(OBSERVED_POSE))


def test_the_tension_direction_agrees_with_the_measured_yield_sign(chain):
    """The one bench fact behind all of this. The camera run found that negative ticks on
    motor 4 lowered the force at MEASURED_POSE; the geometry has to say the same thing, or the
    sense of the whole coordinated pull is backwards and yielding would tighten the tether."""
    from arm1.lib.kinematics import tether_tension_direction

    direction = tether_tension_direction(chain, MEASURED_POSE)
    assert chain.yield_sign(4, MEASURED_POSE, direction) == MEASURED_YIELD_SIGN_MOTOR_4


def test_the_yield_sign_flips_with_the_pull_direction(chain):
    direction = chain.tip_velocity_direction(4, OBSERVED_POSE)
    assert chain.yield_sign(4, OBSERVED_POSE, direction) == -1
    assert chain.yield_sign(4, OBSERVED_POSE, -direction) == 1


def test_a_joint_perpendicular_to_the_pull_has_no_yield_sign(chain):
    """Refusing beats returning +1: there is no tick direction that releases this pull, and a
    loop tracking such a joint would yield forever without the force ever coming down."""
    direction = chain.tip_velocity_direction(4, OBSERVED_POSE)
    sideways = np.cross(direction, chain.tip_velocity_direction(1, OBSERVED_POSE))
    with pytest.raises(ValueError, match="perpendicular"):
        chain.yield_sign(4, OBSERVED_POSE, sideways)


# ---------------------------------------------------------------- ticks per metre of the pull
def test_ticks_per_metre_along_counts_only_motion_along_the_pull(chain):
    """A joint moving mostly sideways to the tether contributes less travel per tick than its
    total speed suggests, so the number has to be larger, never smaller."""
    direction = chain.gripper_axis(OBSERVED_POSE)
    total = 1.0 / chain.tip_speed_per_tick(4, OBSERVED_POSE)
    along = chain.ticks_per_metre_along(4, OBSERVED_POSE, direction)
    assert along > total


def test_ticks_per_metre_along_matches_the_total_when_the_joint_follows_the_pull(chain):
    """The two agree exactly when the joint happens to move straight along the direction."""
    direction = chain.tip_velocity_direction(4, OBSERVED_POSE)
    assert np.isclose(chain.ticks_per_metre_along(4, OBSERVED_POSE, direction),
                      1.0 / chain.tip_speed_per_tick(4, OBSERVED_POSE))


def test_ticks_per_metre_along_is_consistent_with_a_finite_difference(chain):
    direction = chain.gripper_axis(OBSERVED_POSE)
    tpm = chain.ticks_per_metre_along(4, OBSERVED_POSE, direction)
    # Five ticks, about 1 mm: the quantity is a derivative, and over a longer move the joint's
    # arc curves away from the direction by enough to show (50 ticks is already 3 percent out,
    # which is the same arcing that makes a single-joint pull unusable in the first place).
    moved = {**OBSERVED_POSE, 4: OBSERVED_POSE[4] + 5}
    travel = chain.tip_position(moved) - chain.tip_position(OBSERVED_POSE)
    along = abs(float(np.dot(travel, direction)))
    assert np.isclose(5 / along, tpm, rtol=0.01)


def test_a_joint_perpendicular_to_the_pull_has_no_ticks_per_metre(chain):
    """The reciprocal of nearly zero is a huge number that passes every later check, so this
    has to fail here rather than hand the loop a travel scale that can never trip its limit."""
    direction = chain.tip_velocity_direction(4, OBSERVED_POSE)
    sideways = np.cross(direction, chain.tip_velocity_direction(1, OBSERVED_POSE))
    with pytest.raises(ValueError, match="micrometres per tick"):
        chain.ticks_per_metre_along(4, OBSERVED_POSE, sideways)


# ---------------------------------------------------------------- which joint to track
def test_the_tracked_joint_is_the_one_that_measures_the_pull_best(chain):
    direction = chain.gripper_axis(OBSERVED_POSE)
    motors = chain.choose_pull_motors(OBSERVED_POSE, direction)
    tracked = chain.best_tracked_motor(motors, OBSERVED_POSE, direction)

    assert tracked in motors
    best = chain.ticks_per_metre_along(tracked, OBSERVED_POSE, direction)
    for other in motors:
        assert chain.ticks_per_metre_along(other, OBSERVED_POSE, direction) >= best * 0.999


def test_tracking_motor_7_is_never_chosen(chain):
    """Motor 7 only orients the gripper, so its register would never move. Tracking it would
    leave the travel limit permanently at zero."""
    direction = chain.gripper_axis(OBSERVED_POSE)
    assert chain.best_tracked_motor([4, 7], OBSERVED_POSE, direction) == 4


def test_arm_motors_excludes_the_gripper_jaws():
    from arm1.lib.kinematics import ARM_MOTORS

    assert ARM_MOTORS == (1, 2, 3, 4, 5, 6, 7)
    assert 8 not in ARM_MOTORS


# ---------------------------------------------------------------- real joint-range checking
def test_can_pull_straight_ignores_range_by_default(chain):
    """Unchanged behaviour when tick_ranges is not passed: every test above this one in the
    file keeps working without knowing this parameter exists."""
    pose = {1: 2049, 2: 1469, 3: 2468, 4: 2772, 5: 1879, 6: 1525, 7: 3275}
    direction = chain.tip_velocity_direction(4, pose)
    motors = chain.choose_pull_motors(pose, direction)
    ok, _ = chain.can_pull_straight(motors, pose, direction, 0.040)
    assert ok


def test_can_pull_straight_refuses_a_joint_run_past_its_range():
    """The failure this exists to catch: a pull that passes every other check (rank, gain,
    straightness) because all of those are computed from the *commanded* ticks, with no idea
    that one of those commands is physically impossible. Built from a direction and distance
    chosen to run motor 2 (which moves ~14 mm per 100 ticks here) outside a deliberately
    narrow synthetic range, so the test does not depend on this arm's real calibration file."""
    chain = load_chain(URDF)
    pose = {1: 2049, 2: 1469, 3: 2468, 4: 2772, 5: 1879, 6: 1525, 7: 3275}
    direction = chain.tip_velocity_direction(2, pose)
    motors = chain.choose_pull_motors(pose, direction)
    narrow = {m: (0, 4096) for m in motors}
    narrow[2] = (pose[2] - 5, pose[2] + 5)   # motor 2 can move only 5 ticks either way

    ok, why = chain.can_pull_straight(motors, pose, direction, 0.040, tick_ranges=narrow)
    assert not ok
    assert "motor 2" in why
    assert "past its" in why


def test_can_pull_straight_accepts_a_range_wide_enough_to_cover_the_whole_pull(chain):
    pose = {1: 2049, 2: 1469, 3: 2468, 4: 2772, 5: 1879, 6: 1525, 7: 3275}
    direction = chain.tip_velocity_direction(4, pose)
    motors = chain.choose_pull_motors(pose, direction)
    wide = {m: (0, 4096) for m in motors}
    ok, _ = chain.can_pull_straight(motors, pose, direction, 0.040, tick_ranges=wide)
    assert ok


def test_the_range_violation_is_reported_before_singularity_checks(chain):
    """Order matters for which message a person sees. A joint pinned against its real stop is
    the more actionable, more concrete fact; this is implicitly checked by the narrow-range
    test above also tripping even when gain/condition would otherwise pass, but is worth
    stating directly: the range check runs inside the same per-waypoint loop as the others,
    at the first waypoint that violates it, not only at the end."""
    pose = {1: 2049, 2: 1469, 3: 2468, 4: 2772, 5: 1879, 6: 1525, 7: 3275}
    direction = chain.tip_velocity_direction(2, pose)
    motors = chain.choose_pull_motors(pose, direction)
    # Wide enough that, without the range check, this pull is accepted outright.
    assert chain.can_pull_straight(motors, pose, direction, 0.040)[0]
    narrow = {m: (0, 4096) for m in motors}
    narrow[2] = (pose[2] - 5, pose[2] + 5)
    ok, why = chain.can_pull_straight(motors, pose, direction, 0.040, tick_ranges=narrow)
    assert not ok and "motor 2" in why


def test_load_tick_ranges_reads_the_real_follower_calibration():
    ranges = load_tick_ranges("5B8E114979", calib_dir=REPO_ROOT / "config/calibration")
    assert ranges[1] == (1085.0, 3011.0)
    assert set(ranges) == set(range(1, 9))


def test_load_tick_ranges_refuses_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="no calibration"):
        load_tick_ranges("not-a-real-bus", calib_dir=tmp_path)


# ---------------------------------------------------------------- levelling a pull direction
def test_levelling_zeroes_the_z_component():
    from arm1.lib.kinematics import level_direction

    assert level_direction([0.2, 0.9, 0.12])[2] == 0.0


def test_levelling_keeps_the_xy_bearing():
    """Dropping z and renormalising must not change the compass direction of the pull, only
    its length: a levelled pull that also turned sideways would be a different line."""
    from arm1.lib.kinematics import level_direction

    levelled = level_direction([0.2, 0.9, 0.12])
    assert np.isclose(levelled[0] / levelled[1], 0.2 / 0.9)


def test_a_levelled_direction_is_still_a_unit_vector():
    from arm1.lib.kinematics import level_direction

    assert np.isclose(np.linalg.norm(level_direction([0.2, 0.9, 0.12])), 1.0)


def test_an_already_horizontal_direction_is_unchanged():
    from arm1.lib.kinematics import level_direction

    assert np.allclose(level_direction([0.3, 0.9539392, 0.0]), [0.3, 0.9539392, 0.0])


def test_a_vertical_direction_cannot_be_levelled():
    """Nothing is left in the horizontal plane to renormalise. Returning a zero vector would
    hand a degenerate pull to `choose_pull_motors` instead of refusing."""
    from arm1.lib.kinematics import level_direction

    with pytest.raises(ValueError, match="vertical"):
        level_direction([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="vertical"):
        level_direction([1e-9, 1e-9, 1.0])


def test_levelling_the_real_gripper_axis_removes_a_real_tilt(chain):
    """Why this exists at all, on this arm's own numbers: the gripper's axis at the bench
    pose is tilted about 0.11 out of horizontal, which is millimetres of climb over a long
    pull, and is what made a session wander upward where the jog had tracked straight."""
    from arm1.lib.kinematics import level_direction, tether_tension_direction

    pose = {1: 2049, 2: 1469, 3: 2468, 4: 2772, 5: 1879, 6: 1525, 7: 3275}
    tension = tether_tension_direction(chain, pose)
    assert abs(tension[2]) > 0.05, "expected a real tilt to level away"
    assert level_direction(tension)[2] == 0.0
