"""The home pose: the configuration every session starts and ends in.

The point of having one is that control quality stops depending on how the arm was last left.
These check that the saved pose is actually good, not merely present, because a home pose that
is badly conditioned is worse than none: it makes a bad pull repeatable.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from session.home_pose import (
    HOME_POSE_PATH,
    distance_from_home,
    home_pose_metadata,
    is_home,
    load_home_pose,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HOME_FILE = REPO_ROOT / HOME_POSE_PATH
CALIB = REPO_ROOT / "config/calibration/follower-5B8E114979.json"
URDF = REPO_ROOT / "norma-core/hardware/elrobot/simulation/elrobot_follower.urdf"

pytestmark = pytest.mark.skipif(not HOME_FILE.is_file(),
                                reason="no home pose saved; run scripts/find_home_pose.py --save")


def test_it_names_every_arm_joint():
    """All seven, so no joint is left wherever it happened to be."""
    assert sorted(load_home_pose()) == [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.skipif(not CALIB.is_file(), reason="needs the follower calibration")
def test_every_joint_is_inside_its_calibrated_range_with_room_to_spare():
    """A home pose against a joint limit cannot yield in one direction, which is exactly what
    the controller needs to be able to do."""
    home = load_home_pose()
    ranges = json.loads(CALIB.read_text())["ranges"]
    for motor, ticks in home.items():
        lo, hi = ranges[str(motor)]
        assert lo < ticks < hi, f"motor {motor} at {ticks}, range {lo}-{hi}"
        assert min(ticks - lo, hi - ticks) > 300, (
            f"motor {motor} has only {min(ticks - lo, hi - ticks)} ticks of headroom"
        )


@pytest.mark.skipif(not URDF.is_file(), reason="needs the norma-core clone")
def test_it_really_does_pull_straight_in_every_direction():
    """The claim the saved metadata makes, re-derived rather than trusted. This is the whole
    reason this pose was chosen over any other."""
    from lib.kinematics import load_chain

    chain = load_chain(URDF)
    home = load_home_pose()
    motors = chain.choose_pull_motors(home)

    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(10):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        for sign in (1, -1):
            err, _ = chain.line_error(motors, home, d * sign, 0.090, steps=40)
            worst = max(worst, err)
    assert worst < 0.002, f"worst deviation {worst*1000:.2f} mm over 90 mm"


@pytest.mark.skipif(not URDF.is_file(), reason="needs the norma-core clone")
def test_it_is_better_than_the_obvious_alternative():
    """The all-centred pose is what anyone would pick without searching, and it is 8.6 mm off
    in its worst direction. If the saved pose ever stops beating it, the search needs re-running.
    """
    from lib.kinematics import load_chain

    chain = load_chain(URDF)
    home = load_home_pose()
    centred = {m: 2048 for m in range(1, 8)}

    def worst_for(pose):
        motors = chain.choose_pull_motors(pose)
        rng = np.random.default_rng(7)
        worst = 0.0
        for _ in range(10):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            for sign in (1, -1):
                err, _ = chain.line_error(motors, pose, d * sign, 0.090, steps=40)
                worst = max(worst, err)
        return worst

    assert worst_for(home) < worst_for(centred)


def test_the_saved_metadata_matches_what_it_claims():
    meta = home_pose_metadata()
    assert meta["worst_deviation_mm"] < 2.0
    assert len(meta["pull_motors"]) == 3
    assert meta["directions_tested"] >= 20


# ---------------------------------------------------------------- is the arm home
def test_the_home_pose_is_home():
    home = load_home_pose()
    assert distance_from_home(home, home) == 0
    assert is_home(home, home=home)


def test_a_small_offset_still_counts_as_home():
    """A servo holding position sits a few ticks off its goal, so demanding exactness would
    mean never being home."""
    home = load_home_pose()
    nudged = {m: t + 5 for m, t in home.items()}
    assert is_home(nudged, tolerance_ticks=20, home=home)


def test_a_large_offset_does_not():
    home = load_home_pose()
    far = dict(home)
    far[4] += 500
    assert not is_home(far, tolerance_ticks=20, home=home)
    assert distance_from_home(far, home) == 500


def test_the_distance_is_the_worst_joint_not_the_average():
    """One joint far out of place is not excused by the others being right: that is a different
    pose, and the pull geometry depends on all of them."""
    home = load_home_pose()
    one_off = dict(home)
    one_off[2] += 400
    assert distance_from_home(one_off, home) == 400


def test_a_missing_joint_reading_is_treated_as_being_home():
    """Deliberate, and safe in context: `go_home.py` refuses outright when a joint is silent,
    so this never decides anything on its own. Returning a huge distance instead would make a
    quiet bus look like a wildly wrong pose."""
    home = load_home_pose()
    partial = {m: t for m, t in home.items() if m != 5}
    assert distance_from_home(partial, home) == 0
