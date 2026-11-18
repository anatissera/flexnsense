"""The parts of the WiLoR source that are plain geometry: no torch, no GPU, no checkpoint.

Importing hand_tracking.wilor_source must work on a machine with none of WiLoR's dependencies
installed, which is what lets these run in CI and in the repo's own venv.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from arm1.hand_tracking.frame import FLEXION_JOINTS
from arm1.hand_tracking.wilor_source import (
    build_frame,
    project_to_normalized,
    rotation_matrix_to_axis_angle,
)

from tests.hand_tracking.test_pose import straight_hand


def rodrigues(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    """Axis-angle to rotation matrix, written out here so the round trip is a real check."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def test_identity_is_no_rotation():
    assert rotation_matrix_to_axis_angle(np.eye(3)) == (0.0, 0.0, 0.0)


def test_quarter_turn_about_z():
    v = rotation_matrix_to_axis_angle(rodrigues((0, 0, 1), math.pi / 2))
    assert v == pytest.approx((0.0, 0.0, math.pi / 2))


@pytest.mark.parametrize("axis", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0.3, -0.7, 0.6)])
@pytest.mark.parametrize("angle", [0.1, 1.0, 2.0, 3.0])
def test_round_trip_through_a_rotation_matrix(axis, angle):
    v = np.array(rotation_matrix_to_axis_angle(rodrigues(axis, angle)))
    assert np.linalg.norm(v) == pytest.approx(angle, abs=1e-6)
    expected = np.asarray(axis, dtype=float) / np.linalg.norm(axis) * angle
    assert v == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("axis", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.6, 0.8, 0.0)])
def test_half_turns_take_the_degenerate_branch_and_still_come_back(axis):
    """At pi the skew part of the matrix vanishes, which is the case worth pinning down."""
    v = np.array(rotation_matrix_to_axis_angle(rodrigues(axis, math.pi)))
    assert np.linalg.norm(v) == pytest.approx(math.pi, abs=1e-5)
    unit = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    # a and -a describe the same half turn, so either sign is correct.
    assert np.abs(np.dot(v / math.pi, unit)) == pytest.approx(1.0, abs=1e-5)


def test_a_point_on_the_optical_axis_lands_in_the_middle_of_the_image():
    out = project_to_normalized([[0.0, 0.0, 1.0]], [0.0, 0.0, 0.0], focal_length=500, width=640, height=480)
    assert out[0][0] == pytest.approx(0.5)
    assert out[0][1] == pytest.approx(0.5)
    assert out[0][2] == pytest.approx(1.0)  # z stays the camera-space depth in metres


def test_projection_matches_the_pinhole_the_demo_uses():
    # x_px = f * X/Z + width/2, then divided by width.
    out = project_to_normalized([[0.1, -0.2, 2.0]], [0.0, 0.0, 0.0], focal_length=500, width=640, height=480)
    assert out[0][0] == pytest.approx((500 * 0.1 / 2.0 + 320) / 640)
    assert out[0][1] == pytest.approx((500 * -0.2 / 2.0 + 240) / 480)


def test_camera_translation_shifts_the_projection():
    at_origin = project_to_normalized([[0.0, 0.0, 1.0]], [0.0, 0.0, 0.0], 500, 640, 480)
    shifted = project_to_normalized([[0.0, 0.0, 1.0]], [0.5, 0.0, 0.0], 500, 640, 480)
    assert shifted[0][0] > at_origin[0][0]


def test_a_joint_at_the_pinhole_does_not_divide_by_zero():
    out = project_to_normalized([[0.0, 0.0, 0.0]], [0.0, 0.0, 0.0], 500, 640, 480)
    assert all(math.isfinite(c) for c in out[0])


def wilor_frame(is_right: bool = True):
    """A frame built from a straight synthetic hand and identity MANO rotations."""
    return build_frame(
        t_ms=250.0,
        is_right=is_right,
        keypoints_3d=np.array(straight_hand()),
        cam_t=np.array([0.0, 0.0, 0.6]),
        focal_length=500.0,
        width=640,
        height=480,
        hand_pose=np.tile(np.eye(3), (15, 1, 1)),
        global_orient=np.eye(3),
        betas=np.zeros(10),
    )


def test_a_wilor_frame_is_a_handframe_with_the_mano_block_filled():
    d = wilor_frame().to_dict()
    assert d["source"] == "wilor"
    assert d["handedness"] == "right"
    assert d["schema"] == 1
    assert len(d["landmarks"]) == 21
    assert len(d["world"]) == 21
    assert len(d["mano"]["handPose"]) == 15
    assert all(len(row) == 3 for row in d["mano"]["handPose"])
    assert len(d["mano"]["globalOrient"]) == 3
    assert len(d["mano"]["betas"]) == 10


def test_a_wilor_frame_carries_the_same_angles_a_mediapipe_one_would():
    """The whole point of the shared layout: the same 21 points give the same 15 joints."""
    d = wilor_frame().to_dict()
    assert set(d["angles"]) == set(FLEXION_JOINTS)
    for name, angle in d["angles"].items():
        assert angle == pytest.approx(0.0, abs=1e-4), name  # the synthetic hand is straight
    assert d["tipWristMm"] == pytest.approx(1800.0, abs=0.01)


def test_handedness_follows_the_detector():
    assert wilor_frame(is_right=False).to_dict()["handedness"] == "left"


def test_identity_rotations_come_out_as_zero_axis_angles():
    mano = wilor_frame().to_dict()["mano"]
    assert mano["handPose"] == [[0.0, 0.0, 0.0]] * 15
    assert mano["globalOrient"] == [0.0, 0.0, 0.0]
