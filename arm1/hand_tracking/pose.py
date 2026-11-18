"""21 hand landmarks to the 15 MyoHand flexion angles, in radians.

This is what fills a HandFrame's ``angles``, and it is deliberately the only place the mapping
lives on the Python side: MediaPipe and WiLoR both hand over 21 points in the same layout, so
they both get the same angles out and an angle means the same thing whichever one produced it.

``landmarksToPose`` in ``dashboard/app/lib/../components/HandTracker.tsx`` is the TypeScript twin
of :func:`landmarks_to_pose`, used by the in-browser tracker. Keep the two identical.

The 21-point layout is MediaPipe's, which is also OpenPose's, which is also what WiLoR emits
(its MANO wrapper remaps to OpenPose order, see hand_tracking/wilor_source.py):

    0        wrist
    1-4      thumb:  CMC, MCP, IP, TIP
    5-8      index:  MCP, PIP, DIP, TIP
    9-12     middle: MCP, PIP, DIP, TIP
    13-16    ring:   MCP, PIP, DIP, TIP
    17-20    pinky:  MCP, PIP, DIP, TIP

Pass metric, hand centred points (MediaPipe's ``hand_world_landmarks``, WiLoR's
``pred_keypoints_3d``), never the normalized image ones: see hand_tracking/landmarks.py for why.
"""
from __future__ import annotations

import math
from typing import Sequence

Point = tuple[float, float, float]

#: Degenerate bone length below which an angle is meaningless and reported as 0.
_EPS = 1e-6


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def bend(u: Point, v: Point) -> float:
    """Angle between two bone vectors: 0 when they are collinear (a straight joint), up to pi."""
    du = math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
    dv = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if du < _EPS or dv < _EPS:
        return 0.0
    cos = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (du * dv)
    return math.acos(max(-1.0, min(1.0, cos)))


def landmarks_to_pose(points: Sequence[Point]) -> dict[str, float]:
    """21 landmarks to ``{joint name: flexion angle in radians}``, one entry per FLEXION_JOINT.

    Fingers 2-5: MCP is wrist->MCP against the proximal phalanx, PIP is proximal against middle,
    DIP is middle against distal. The thumb's three angles come out negative, because flexion is
    the negative direction for those joints in the MyoHand model.
    """
    if len(points) < 21:
        raise ValueError(f"need 21 landmarks, got {len(points)}")

    def bone(i: int) -> Point:
        return _sub(points[i + 1], points[i])

    pose: dict[str, float] = {}
    for k, digit in enumerate((2, 3, 4, 5)):
        b = 5 + 4 * k  # MCP landmark index: 5, 9, 13, 17
        pose[f"mcp{digit}_flexion_r"] = bend(_sub(points[b], points[0]), bone(b))
        pose[f"pm{digit}_flexion_r"] = bend(bone(b), bone(b + 1))
        pose[f"md{digit}_flexion_r"] = bend(bone(b + 1), bone(b + 2))
    pose["cmc_flexion_r"] = -bend(bone(0), bone(1))
    pose["mp_flexion_r"] = -bend(bone(1), bone(2))
    pose["ip_flexion_r"] = -bend(bone(2), bone(3))
    return pose


def tip_wrist_mm(points: Sequence[Point]) -> float:
    """Index fingertip to wrist distance in millimetres, from metric points."""
    d = _sub(points[8], points[0])
    return math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) * 1000.0
