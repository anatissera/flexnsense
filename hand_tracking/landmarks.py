"""Pure geometry on MediaPipe's 21 hand landmarks: one flexion angle per finger.

No mediapipe or opencv import here, so this is testable with plain (x, y, z) tuples - see
tests/hand_tracking/test_landmarks.py for synthetic examples.

Use the *world* landmarks (metres, hand-centred, equal scale on all three axes), not the
normalized image landmarks (x/y normalized to image width/height separately, z a rough
relative depth) - those distort angles. MediaPipe's HandLandmarkerResult exposes both;
`hand_world_landmarks` is the one to pass in here.

MediaPipe's 21-point layout (same indices for image and world landmarks):
    0       WRIST
    1-4     THUMB:  CMC, MCP, IP, TIP
    5-8     INDEX:  MCP, PIP, DIP, TIP
    9-12    MIDDLE: MCP, PIP, DIP, TIP
    13-16   RING:   MCP, PIP, DIP, TIP
    17-20   PINKY:  MCP, PIP, DIP, TIP
"""
from __future__ import annotations

import math

WRIST = 0
THUMB = (1, 2, 3, 4)      # CMC, MCP, IP, TIP
INDEX = (5, 6, 7, 8)      # MCP, PIP, DIP, TIP
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = {"thumb": THUMB, "index": INDEX, "middle": MIDDLE, "ring": RING, "pinky": PINKY}

Point = tuple[float, float, float]


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v: Point) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def joint_angle_deg(a: Point, b: Point, c: Point) -> float:
    """Angle a-b-c at vertex b, in degrees: 180 = perfectly straight, 0 = folded flat onto itself."""
    u, v = _sub(a, b), _sub(c, b)
    nu, nv = _norm(u), _norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    cos = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def flexion_deg(a: Point, b: Point, c: Point) -> float:
    """Flexion at joint b: 0 = fully extended (straight), larger = more bent, up to ~180."""
    angle = joint_angle_deg(a, b, c)
    return angle if math.isnan(angle) else 180.0 - angle


def finger_flexion_deg(points: list[Point], finger: tuple[int, int, int, int]) -> float:
    """PIP-joint flexion for one of INDEX/MIDDLE/RING/PINKY: angle at PIP, between MCP and DIP."""
    mcp, pip, dip, _tip = finger
    return flexion_deg(points[mcp], points[pip], points[dip])


def thumb_flexion_deg(points: list[Point]) -> float:
    """IP-joint flexion for the thumb (it has no DIP): angle at IP, between MCP and TIP."""
    _cmc, mcp, ip, tip = THUMB
    return flexion_deg(points[mcp], points[ip], points[tip])


def all_flexion_deg(points: list[Point]) -> dict[str, float]:
    """One representative flexion angle per finger: PIP for the four fingers, IP for the thumb."""
    return {
        name: thumb_flexion_deg(points) if name == "thumb" else finger_flexion_deg(points, idxs)
        for name, idxs in FINGERS.items()
    }
