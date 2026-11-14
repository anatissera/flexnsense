"""The MediaPipe source's pure parts, with fakes standing in for mediapipe's result objects.

mediapipe is a heavy optional dependency, so importing hand_tracking.mediapipe_source must not
need it. These run on a machine without it installed.
"""
from __future__ import annotations

import pytest

from hand_tracking.frame import FLEXION_JOINTS
from hand_tracking.mediapipe_source import build_frame, handedness_of, to_points

from tests.hand_tracking.test_pose import straight_hand


class FakeLandmark:
    """Stands in for mediapipe's NormalizedLandmark, which also carries fields we ignore."""

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z
        self.visibility = 0.9  # present on the real object, and not part of a HandFrame


class FakeCategory:
    def __init__(self, category_name: str) -> None:
        self.category_name = category_name


def fake_landmarks(points):
    return [FakeLandmark(*p) for p in points]


def test_landmark_objects_become_plain_tuples():
    points = to_points(fake_landmarks([(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)]))
    assert points == [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)]


@pytest.mark.parametrize("name,expected", [("Left", "left"), ("Right", "right"), ("left", "left")])
def test_handedness_is_lowercased(name, expected):
    assert handedness_of([FakeCategory(name)]) == expected


@pytest.mark.parametrize("categories", [None, [], [FakeCategory("Unknown")], [object()]])
def test_handedness_is_none_when_mediapipe_does_not_say(categories):
    assert handedness_of(categories) is None


def test_a_mediapipe_frame_has_no_mano_block():
    """MediaPipe estimates landmarks rather than fitting a hand model, so it has no MANO params."""
    d = build_frame(12.0, fake_landmarks(straight_hand()), fake_landmarks(straight_hand()),
                    [FakeCategory("Right")]).to_dict()
    assert d["source"] == "mediapipe"
    assert "mano" not in d
    assert d["handedness"] == "right"
    assert d["t"] == pytest.approx(12.0)


def test_it_fills_the_same_angles_a_wilor_frame_would():
    frame = build_frame(0.0, fake_landmarks(straight_hand()), fake_landmarks(straight_hand()), None)
    d = frame.to_dict()
    assert set(d["angles"]) == set(FLEXION_JOINTS)
    for name, angle in d["angles"].items():
        assert angle == pytest.approx(0.0, abs=1e-4), name
    assert d["tipWristMm"] == pytest.approx(1800.0, abs=0.01)


def test_angles_come_from_the_world_landmarks_not_the_image_ones():
    """The image landmarks are distorted by aspect ratio, so an angle from them would be wrong."""
    flat = [(x, y, 0.0) for x, y, _ in straight_hand()]  # a plausible but different set of points
    frame = build_frame(0.0, fake_landmarks(flat), fake_landmarks(straight_hand()), None)
    straight = build_frame(0.0, fake_landmarks(straight_hand()), fake_landmarks(straight_hand()), None)
    assert frame.to_dict()["angles"] == straight.to_dict()["angles"]
