"""The 21 landmarks to MyoHand flexion angles mapping, on synthetic hands. No camera, no model."""
from __future__ import annotations

import math

import pytest

from hand_tracking.frame import FLEXION_JOINTS
from hand_tracking.pose import bend, landmarks_to_pose, tip_wrist_mm

# Each finger's base landmark index and the direction it radiates in, roughly hand shaped.
FINGER_DIRS = {
    1: (0.6, 0.5, 0.0),    # thumb:  CMC, MCP, IP, TIP
    5: (0.2, 1.0, 0.0),    # index:  MCP, PIP, DIP, TIP
    9: (0.0, 1.0, 0.0),    # middle
    13: (-0.2, 1.0, 0.0),  # ring
    17: (-0.4, 0.9, 0.0),  # pinky
}
# Distances from the wrist along that direction, so every joint in the chain is collinear.
SPANS = (0.9, 1.3, 1.6, 1.8)
PERP = (0.0, 0.0, 1.0)  # perpendicular to every direction above, which all lie in the xy plane

# acos is ill conditioned near 1, so a perfectly straight joint lands within about 1e-8 of zero
# rather than on it. That is 6e-7 degrees, far below anything a camera can resolve.
STRAIGHT = 1e-6


def unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = math.sqrt(sum(c * c for c in v))
    return (v[0] / n, v[1] / n, v[2] / n)


def straight_hand() -> list[tuple[float, float, float]]:
    """Wrist at the origin, every finger a straight line radiating from it, so every angle is 0."""
    points = [(0.0, 0.0, 0.0)] * 21
    for base, direction in FINGER_DIRS.items():
        u = unit(direction)
        for offset, span in enumerate(SPANS):
            points[base + offset] = (u[0] * span, u[1] * span, u[2] * span)
    return points


def bent_at(base: int, joint: int) -> list[tuple[float, float, float]]:
    """A straight hand with one chain turned a right angle at `joint` landmarks past `base`."""
    points = straight_hand()
    pivot = points[base + joint]
    for offset in range(joint + 1, 4):
        step = SPANS[offset] - SPANS[joint]
        points[base + offset] = (pivot[0] + PERP[0] * step, pivot[1] + PERP[1] * step, pivot[2] + PERP[2] * step)
    return points


def test_bend_is_zero_for_collinear_bones_and_safe_on_degenerate_ones():
    assert bend((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)) == pytest.approx(0.0)
    assert bend((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == pytest.approx(math.pi / 2)
    assert bend((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)) == pytest.approx(math.pi)
    assert bend((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == 0.0  # zero-length bone, not a crash


def test_a_straight_hand_has_every_joint_at_zero():
    pose = landmarks_to_pose(straight_hand())
    assert set(pose) == set(FLEXION_JOINTS)
    for name, angle in pose.items():
        assert angle == pytest.approx(0.0, abs=STRAIGHT), name


def test_bending_the_index_pip_moves_only_that_joint():
    pose = landmarks_to_pose(bent_at(5, 1))  # index PIP is one landmark past the MCP
    assert pose["pm2_flexion_r"] == pytest.approx(math.pi / 2)
    assert pose["mcp2_flexion_r"] == pytest.approx(0.0, abs=STRAIGHT)
    assert pose["md2_flexion_r"] == pytest.approx(0.0, abs=STRAIGHT)
    # The other three fingers are untouched.
    for digit in (3, 4, 5):
        assert pose[f"pm{digit}_flexion_r"] == pytest.approx(0.0, abs=STRAIGHT)


def test_each_finger_maps_to_its_own_digit():
    for digit, base in zip((2, 3, 4, 5), (5, 9, 13, 17)):
        pose = landmarks_to_pose(bent_at(base, 1))
        assert pose[f"pm{digit}_flexion_r"] == pytest.approx(math.pi / 2)


def test_the_thumb_flexes_negative():
    """Flexion is the negative direction for the thumb's joints in the MyoHand model."""
    pose = landmarks_to_pose(bent_at(1, 2))  # bend at the IP
    assert pose["ip_flexion_r"] == pytest.approx(-math.pi / 2)
    assert pose["cmc_flexion_r"] == pytest.approx(0.0, abs=STRAIGHT)
    assert pose["mp_flexion_r"] == pytest.approx(0.0, abs=STRAIGHT)


def test_tip_wrist_distance_is_millimetres():
    # The index tip sits SPANS[-1] metres from the wrist along a unit direction.
    assert tip_wrist_mm(straight_hand()) == pytest.approx(SPANS[-1] * 1000.0)


def test_too_few_landmarks_is_an_error_not_a_wrong_answer():
    with pytest.raises(ValueError):
        landmarks_to_pose(straight_hand()[:20])
