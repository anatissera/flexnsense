"""The HandFrame wire format, and the two contracts it has to stay inside.

No camera and no mediapipe here: a HandFrame is built from plain numbers, and the two contract
tests read committed files (the MuJoCo model and the TypeScript mirror).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from arm1.hand_tracking.frame import (
    ANGLE_PLACES,
    FINGERTIPS,
    FLEXION_JOINTS,
    LANDMARK_PLACES,
    SCHEMA_VERSION,
    WORLD_PLACES,
    HandFrame,
    ManoParams,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MYOHAND_XML = REPO_ROOT / "dashboard" / "public" / "models" / "myohand" / "myohand_r.xml"
HAND_FRAME_TS = REPO_ROOT / "dashboard" / "app" / "lib" / "handFrame.ts"


def make_frame(**overrides) -> HandFrame:
    """A frame with 21 distinguishable points, so an index mix-up shows up as a wrong number."""
    points = [(i / 100, i / 200, i / 400) for i in range(21)]
    kwargs = dict(
        source="mediapipe",
        t_ms=1234.5678,
        handedness="right",
        landmarks=points,
        world=points,
        angles={"mcp2_flexion_r": 0.123456, "ip_flexion_r": -0.654321},
        tip_wrist_mm=87.654321,
    )
    kwargs.update(overrides)
    return HandFrame(**kwargs)


def test_to_dict_has_the_documented_keys():
    d = make_frame().to_dict()
    assert set(d) == {"schema", "source", "t", "handedness", "landmarks", "world", "angles", "tipWristMm"}
    assert d["schema"] == SCHEMA_VERSION
    assert d["source"] == "mediapipe"
    assert d["handedness"] == "right"
    assert len(d["landmarks"]) == 21
    assert len(d["world"]) == 21
    assert set(d["landmarks"][0]) == {"x", "y", "z"}


def test_mano_is_absent_unless_the_source_produced_it():
    assert "mano" not in make_frame().to_dict()

    mano = ManoParams(
        hand_pose=[[0.1, 0.2, 0.3]] * 15,
        global_orient=[0.4, 0.5, 0.6],
        betas=list(range(10)),
    )
    d = make_frame(source="wilor", mano=mano).to_dict()
    assert d["mano"]["handPose"] == [[0.1, 0.2, 0.3]] * 15
    assert len(d["mano"]["handPose"]) == 15
    assert d["mano"]["globalOrient"] == [0.4, 0.5, 0.6]
    assert len(d["mano"]["betas"]) == 10
    # A reader that only knows the base fields still sees a complete frame.
    assert d["angles"] and len(d["world"]) == 21


def test_numbers_are_rounded_to_the_documented_precision():
    d = make_frame(
        landmarks=[(0.123456789, 0.0, 0.0)] * 21,
        world=[(0.123456789, 0.0, 0.0)] * 21,
    ).to_dict()
    assert d["landmarks"][0]["x"] == pytest.approx(round(0.123456789, LANDMARK_PLACES))
    assert d["world"][0]["x"] == pytest.approx(round(0.123456789, WORLD_PLACES))
    assert d["angles"]["mcp2_flexion_r"] == pytest.approx(round(0.123456, ANGLE_PLACES))
    assert d["tipWristMm"] == pytest.approx(87.65)
    assert d["t"] == pytest.approx(1234.57)


def test_angles_keys_are_a_subset_of_the_known_joints():
    assert set(make_frame().to_dict()["angles"]) <= set(FLEXION_JOINTS)


def test_every_flexion_joint_exists_in_the_myohand_model():
    """The angles a frame carries are only useful if MuJoCo can look the names up."""
    names = {j.get("name") for j in ET.parse(MYOHAND_XML).getroot().iter("joint")}
    assert set(FLEXION_JOINTS) <= names
    assert len(FLEXION_JOINTS) == 15


def _ts_const(source: str, name: str) -> str:
    match = re.search(rf"export const {name} = \[(.*?)\] as const;", source, re.DOTALL)
    assert match, f"{name} not found in {HAND_FRAME_TS}"
    return match.group(1)


def test_typescript_mirror_declares_the_same_contract():
    """handFrame.ts is the other half of this schema; silent drift between the two is the risk."""
    source = HAND_FRAME_TS.read_text()
    assert tuple(re.findall(r'"([a-z0-9_]+)"', _ts_const(source, "FLEXION_JOINTS"))) == FLEXION_JOINTS
    assert tuple(int(n) for n in re.findall(r"\d+", _ts_const(source, "FINGERTIPS"))) == FINGERTIPS
    # The wire keys the TypeScript side writes, as spelled in buildHandFrame.
    for key in ("schema", "source", "handedness", "landmarks", "world", "angles", "tipWristMm"):
        assert key in source
