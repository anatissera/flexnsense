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
