import math

import pytest

from hand_tracking.landmarks import (
    FINGERS,
    INDEX,
    all_flexion_deg,
    finger_flexion_deg,
    flexion_deg,
    joint_angle_deg,
    thumb_flexion_deg,
)

Point = tuple[float, float, float]


def test_joint_angle_straight_line_is_180():
    a, b, c = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)
    assert joint_angle_deg(a, b, c) == pytest.approx(180.0)


def test_joint_angle_right_angle_bend():
    a, b, c = (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    assert joint_angle_deg(a, b, c) == pytest.approx(90.0)


def test_joint_angle_folded_flat_is_0():
    a, b, c = (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)  # c coincides with a: folded back onto itself
    assert joint_angle_deg(a, b, c) == pytest.approx(0.0)


def test_joint_angle_degenerate_point_is_nan():
    a, b, c = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)  # a == b: zero-length segment
    assert math.isnan(joint_angle_deg(a, b, c))


def test_flexion_is_180_minus_joint_angle():
    right_angle = (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    assert flexion_deg(*right_angle) == pytest.approx(90.0)
    straight = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)
    assert flexion_deg(*straight) == pytest.approx(0.0)


def _bend(flex_deg: float, vertex: Point = (0.0, 0.0, 0.0)) -> tuple[Point, Point, Point]:
    """3 points (a, vertex, c), unit-length arms, whose flexion_deg is exactly `flex_deg`."""
    vx, vy, vz = vertex
    a = (vx + 1.0, vy, vz)
    theta = math.radians(180.0 - flex_deg)  # the joint angle this flexion corresponds to
    c = (vx + math.cos(theta), vy + math.sin(theta), vz)
    return a, vertex, c


def _straight_hand() -> list[Point]:
    """21 points, every finger perfectly straight (flexion 0 for all five)."""
    points: list[Point] = [(0.0, 0.0, 0.0)] * 21  # index 0 (WRIST) plus slack for unused indices
    for row, (name, idxs) in enumerate(FINGERS.items()):
        for col, idx in enumerate(idxs):
            points[idx] = (float(col), float(row) * 10.0, 0.0)  # rows far apart: no accidental cross-finger angles
    return points


def _bend_finger(points: list[Point], name: str, flex_deg: float) -> None:
    """Overwrite one finger's joints so its representative flexion angle (PIP, or IP for the thumb) is `flex_deg`."""
    idxs = FINGERS[name]
    a_idx, vertex_idx, c_idx = (idxs[1], idxs[2], idxs[3]) if name == "thumb" else (idxs[0], idxs[1], idxs[2])
    row_z = points[vertex_idx]
    points[a_idx], points[vertex_idx], points[c_idx] = _bend(flex_deg, vertex=row_z)


def test_finger_flexion_straight_is_0():
    points = _straight_hand()
    assert finger_flexion_deg(points, INDEX) == pytest.approx(0.0)


def test_finger_flexion_bent_90():
    points = _straight_hand()
    _bend_finger(points, "index", 90.0)
    assert finger_flexion_deg(points, INDEX) == pytest.approx(90.0)


def test_thumb_flexion_straight_is_0():
    points = _straight_hand()
    assert thumb_flexion_deg(points) == pytest.approx(0.0)


def test_thumb_flexion_bent_45():
    points = _straight_hand()
    _bend_finger(points, "thumb", 45.0)
    assert thumb_flexion_deg(points) == pytest.approx(45.0)


def test_all_flexion_deg_straight_hand_is_all_zero():
    result = all_flexion_deg(_straight_hand())
    assert set(result) == {"thumb", "index", "middle", "ring", "pinky"}
    for name, angle in result.items():
        assert angle == pytest.approx(0.0), f"{name}: expected ~0, got {angle}"


def test_all_flexion_deg_only_bent_finger_changes():
    points = _straight_hand()
    _bend_finger(points, "middle", 60.0)
    result = all_flexion_deg(points)
    assert result["middle"] == pytest.approx(60.0)
    for name in ("thumb", "index", "ring", "pinky"):
        assert result[name] == pytest.approx(0.0)
