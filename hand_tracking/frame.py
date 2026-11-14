"""HandFrame: the one shape every hand tracker emits, whatever model produced it.

The dashboard's in-browser MediaPipe tracker fills the base fields; a WiLoR backend fills the
same base fields plus ``mano``. Anything that reads only the base fields therefore works with
either source, which is what keeps the two swappable.

``dashboard/app/lib/handFrame.ts`` is the TypeScript mirror of this module. The two must agree
on every JSON key, on ``FLEXION_JOINTS`` and on ``FINGERTIPS``; ``tests/hand_tracking/test_frame.py``
reads that file and fails if they drift apart.

Units: angles in radians (what both the MyoHand MuJoCo joints and MANO use), ``world`` in metres,
``tip_wrist_mm`` in millimetres, ``t_ms`` in milliseconds since tracking started.

Note this is radians, while ``hand_tracking.landmarks`` and the CLI's CSV report degrees. The CSV
is read by a person; a HandFrame is fed to MuJoCo and to MANO, and both want radians.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

SCHEMA_VERSION = 1

#: Fingertip indices in the shared 21-point layout: thumb, index, middle, ring, pinky.
FINGERTIPS = (4, 8, 12, 16, 20)

#: The 15 MyoHand flexion joints a tracked hand drives, in MuJoCo's naming. Every one of these
#: exists in dashboard/public/models/myohand/myohand_r.xml.
#:
#: That model also carries cmc_abduction_r and mcp2..5_abduction_r, which a single flexion angle
#: per joint cannot fill. A MANO source can, because its rotations are full 3 DOF: that is what
#: ManoParams.hand_pose is reserved for.
FLEXION_JOINTS = (
    "mcp2_flexion_r", "pm2_flexion_r", "md2_flexion_r",
    "mcp3_flexion_r", "pm3_flexion_r", "md3_flexion_r",
    "mcp4_flexion_r", "pm4_flexion_r", "md4_flexion_r",
    "mcp5_flexion_r", "pm5_flexion_r", "md5_flexion_r",
    "cmc_flexion_r", "mp_flexion_r", "ip_flexion_r",
)

# Enough precision to be lossless in practice, short enough to log or stream: 4 places on a
# normalized coordinate is a tenth of a pixel at 1000 px, 5 places on metres is 0.01 mm, and
# 4 places on radians is 0.006 degrees.
LANDMARK_PLACES = 4
WORLD_PLACES = 5
ANGLE_PLACES = 4
MM_PLACES = 2

Point = tuple[float, float, float]


def _points(points: Sequence[Point], places: int) -> list[dict[str, float]]:
    return [{"x": round(p[0], places), "y": round(p[1], places), "z": round(p[2], places)} for p in points]


@dataclass(frozen=True)
class ManoParams:
    """Rotations and shape from a MANO-based model. Only a MANO source (WiLoR) produces these."""

    #: 15 x 3 axis-angle rotations in radians, one per MANO hand joint, full 3 DOF each.
    #:
    #: Careful: these are in MANO's own joint order, which is NOT the 21-point order the rest of
    #: a HandFrame uses. MANO goes index, middle, pinky, ring, thumb, three joints each, base to
    #: tip. Only ``landmarks`` and ``world`` are remapped to the shared layout.
    hand_pose: Sequence[Sequence[float]]
    #: Hand orientation in camera space, axis-angle, radians.
    global_orient: Sequence[float]
    #: The 10 MANO shape coefficients, i.e. this subject's hand proportions.
    betas: Sequence[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "handPose": [list(row) for row in self.hand_pose],
            "globalOrient": list(self.global_orient),
            "betas": list(self.betas),
        }


@dataclass(frozen=True)
class HandFrame:
    """One tracked frame. Use :meth:`to_dict` to get the JSON the dashboard consumes."""

    #: "mediapipe" or "wilor".
    source: str
    #: Milliseconds since tracking started.
    t_ms: float
    #: "left", "right", or None when the tracker does not report it.
    handedness: str | None
    #: 21 points in normalized image coordinates (0..1 of width and height). For drawing only.
    #: Only x and y are contractual: z is a relative depth from MediaPipe and a camera-space
    #: depth in metres from WiLoR. Read ``world`` for anything quantitative.
    landmarks: Sequence[Point]
    #: 21 points in metres, hand centred. Angles come from these, never from ``landmarks``.
    world: Sequence[Point]
    #: Joint name to flexion angle in radians. Keys are a subset of :data:`FLEXION_JOINTS`.
    angles: dict[str, float]
    #: Index fingertip to wrist distance, millimetres.
    tip_wrist_mm: float
    mano: ManoParams | None = None

    def to_dict(self) -> dict[str, Any]:
        """The wire format, rounded to the precision the schema documents."""
        frame: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "source": self.source,
            "t": round(self.t_ms, MM_PLACES),
            "handedness": self.handedness,
            "landmarks": _points(self.landmarks, LANDMARK_PLACES),
            "world": _points(self.world, WORLD_PLACES),
            "angles": {name: round(value, ANGLE_PLACES) for name, value in self.angles.items()},
            "tipWristMm": round(self.tip_wrist_mm, MM_PLACES),
        }
        if self.mano is not None:
            frame["mano"] = self.mano.to_dict()
        return frame
