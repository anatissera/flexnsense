"""How far a fingertip travels along its own pull direction, from a sequence of tracked frames.

Pure geometry, no mediapipe/cv2/numpy-shaped-surprises here beyond plain arithmetic - this is
what makes it testable with synthetic points (tests/hand_tracking/test_excursion.py), the same
discipline as `hand_tracking.landmarks` and `hand_tracking.pose`.

The measurement this exists for (docs/admittance-design.md's `excursion_m`, and
`session.calibration.BenchCalibration.excursion_m`): move one finger slowly from full extension
to full flexion while `hand_tracking run --jsonl` records every frame, then find out how far the
tip moved *along the direction it actually moved in*.

Two things rule out the obvious shortcut of just differencing the first and last frame's tip
position:

- MediaPipe's ``world`` landmarks are hand-centred, so their origin drifts a little as the hand's
  overall pose changes during the motion. Differencing two raw ``world`` points mixes that origin
  drift into the fingertip's own travel. Taking tip-minus-wrist first removes it, because the
  drift cancels between the two points of each frame.
- The two endpoints alone throw away every other frame, including whichever one was actually the
  most extended or most flexed - a slow, deliberate motion rarely peaks exactly where recording
  started and stopped.

So this fits a single axis through the *whole* tip-minus-wrist trajectory (its direction of
greatest spread, via SVD - equivalent to PCA on three points or more) and reports the spread of
all frames projected onto that axis. That axis is also, physically, a good proxy for the tether
axis: it is the actual direction the fingertip moved while flexing, which is what the tether
pulls against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

Point = tuple[float, float, float]

#: hand_tracking's own vocabulary for the fifth digit is "pinky" (hand_tracking/landmarks.py);
#: this module and its callers speak that, not the clinical "little" - see session.fingers for
#: where the two vocabularies get reconciled downstream.
FINGERTIP_BY_NAME = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}

WRIST = 0


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v: Point) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def tip_minus_wrist(points: Sequence[Point], finger: str) -> Point:
    """One frame's fingertip position relative to the wrist, which cancels the world origin drift."""
    if finger not in FINGERTIP_BY_NAME:
        raise ValueError(f"unknown finger {finger!r}; known: {sorted(FINGERTIP_BY_NAME)}")
    if len(points) < 21:
        raise ValueError(f"need 21 landmarks, got {len(points)}")
    return _sub(points[FINGERTIP_BY_NAME[finger]], points[WRIST])


@dataclass(frozen=True)
class ExcursionResult:
    """The fit through one finger's tip-minus-wrist trajectory across a recording."""

    #: Spread of the trajectory along its own dominant direction, in metres. The headline number.
    excursion_m: float
    #: Unit vector, the fitted axis, in the same hand-centred frame the input points were in.
    #: Sign is arbitrary (SVD does not orient it); only the line it defines is meaningful.
    axis: tuple[float, float, float]
    #: Fraction of the trajectory's total variance this one axis accounts for, 0..1. Low means
    #: the motion was not close to a single straight line - a flinch, a rotated wrist, a second
    #: finger moving too - and `excursion_m` is then a line fitted through something that was not
    #: really 1-D motion, which should be shown rather than hidden.
    variance_explained: float
    #: RMS distance of the trajectory from the fitted axis, in metres: how much the motion
    #: wandered off-line. Compare it to `excursion_m` - a few percent is a clean pull, anything
    #: approaching `excursion_m` itself means the axis is not trustworthy.
    perpendicular_rms_m: float
    n_samples: int


def fit_excursion(vectors: Sequence[Point]) -> ExcursionResult:
    """Fit the dominant axis through a sequence of tip-minus-wrist vectors and its excursion.

    Needs at least 2 samples (a straight line has no spread to speak of with just 1), but a real
    recording at 30 Hz over a few seconds of slow motion gives tens to hundreds.
    """
    n = len(vectors)
    if n < 2:
        raise ValueError(f"need at least 2 samples, got {n}")

    mean = tuple(sum(v[i] for v in vectors) / n for i in range(3))
    centered = [_sub(v, mean) for v in vectors]

    # Power iteration on the 3x3 covariance matrix for the dominant eigenvector: no numpy
    # dependency needed for a 3x3 symmetric matrix, and this module stays import-light on
    # purpose (see module docstring on why it has to run with no mediapipe/cv2 import).
    cov = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for v in centered:
        for i in range(3):
            for j in range(3):
                cov[i][j] += v[i] * v[j]
    for i in range(3):
        for j in range(3):
            cov[i][j] /= n

    axis = _dominant_eigenvector(cov)
    projections = [v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2] for v in centered]
    excursion = max(projections) - min(projections)

    total_var = sum(cov[i][i] for i in range(3))
    along_var = sum(p * p for p in projections) / n
    variance_explained = along_var / total_var if total_var > 0 else 1.0

    perp_sq = [
        _norm(_sub(v, tuple(p * a for a in axis))) ** 2
        for v, p in zip(centered, projections)
    ]
    perpendicular_rms = math.sqrt(sum(perp_sq) / n)

    return ExcursionResult(
        excursion_m=excursion,
        axis=axis,
        variance_explained=variance_explained,
        perpendicular_rms_m=perpendicular_rms,
        n_samples=n,
    )


def _dominant_eigenvector(cov: list[list[float]], iterations: int = 200) -> tuple[float, float, float]:
    """Power iteration: the eigenvector for the largest eigenvalue of a 3x3 symmetric matrix.

    Good enough here because the matrix is tiny and the recording's trajectory is expected to
    have one axis that dominates the other two by a wide margin - that dominance is exactly what
    `variance_explained` reports back, so a case where power iteration would converge slowly
    (near-equal top eigenvalues) is already a case this function flags as untrustworthy.
    """
    x, y, z = 1.0, 1.0, 1.0
    for _ in range(iterations):
        nx = cov[0][0] * x + cov[0][1] * y + cov[0][2] * z
        ny = cov[1][0] * x + cov[1][1] * y + cov[1][2] * z
        nz = cov[2][0] * x + cov[2][1] * y + cov[2][2] * z
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm < 1e-18:
            return (0.0, 0.0, 0.0)
        x, y, z = nx / norm, ny / norm, nz / norm
    return (x, y, z)
