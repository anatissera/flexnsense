"""Shared geometry for the scale reader. This is the contract between detector, calibration and the CLI.

Coordinates are OpenCV image pixels: x to the right, y down, floats allowed (sub-pixel).

The *axis* runs along the scale tube, through the line the red marker slides on:
  p0 = the point on that line at the 0 g printed mark
  p1 = the point on that line at the full-scale (600 g) printed mark
The marker moves from p0 towards p1 as the load grows.

`t` (in pixels) is the signed distance of a point along the axis, measured from p0 in the direction p0 -> p1.
`n` (in pixels) is the signed distance perpendicular to the axis (positive = axis direction rotated +90 degrees).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Axis:
    p0: tuple[float, float]
    p1: tuple[float, float]

    @property
    def length(self) -> float:
        """Distance p0 -> p1 in pixels."""
        return float(np.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1]))

    @property
    def unit(self) -> np.ndarray:
        """Unit vector from p0 to p1."""
        d = np.array(self.p1, dtype=np.float64) - np.array(self.p0, dtype=np.float64)
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError("Axis p0 and p1 are the same point")
        return d / n

    @property
    def normal(self) -> np.ndarray:
        """Unit vector perpendicular to the axis (unit rotated +90 degrees in image coordinates)."""
        u = self.unit
        return np.array([-u[1], u[0]])

    @property
    def angle_deg(self) -> float:
        """Angle of p0 -> p1 in degrees, as atan2(dy, dx) in image coordinates."""
        u = self.unit
        return float(np.degrees(np.arctan2(u[1], u[0])))

    def project(self, point: tuple[float, float]) -> tuple[float, float]:
        """Image point -> (t, n) in axis coordinates."""
        v = np.array(point, dtype=np.float64) - np.array(self.p0, dtype=np.float64)
        return float(v @ self.unit), float(v @ self.normal)

    def point_at(self, t: float, n: float = 0.0) -> tuple[float, float]:
        """(t, n) in axis coordinates -> image point."""
        p = np.array(self.p0, dtype=np.float64) + t * self.unit + n * self.normal
        return float(p[0]), float(p[1])

    def to_dict(self) -> dict:
        return {"p0": list(self.p0), "p1": list(self.p1)}

    @classmethod
    def from_dict(cls, d: dict) -> "Axis":
        return cls(tuple(map(float, d["p0"])), tuple(map(float, d["p1"])))
