"""Calibration: marker position along the axis (t_px) -> grams.

The Pesola spring is linear (Hooke's law), so grams = offset_g + slope_g_per_px * t_px.

Workflow:
    cal = Calibration.from_clicks(p0, p1)          # user clicks the 0 g and 600 g printed lines
    cal.tare(t_px_with_nothing_hanging)            # optional: zero the reading, slope unchanged
    cal.add_sample(t_px, 200.0); cal.fit()         # optional: refine with known weights
    cal.save(DEFAULT_CALIB_PATH)

Sign convention everywhere (residuals, check()): error = reading - truth, in grams.

Hand-held scale: StickerCalibration works in u instead of t_px. u is the marker position as a fraction between
two red stickers on the tube (s1 just before the 0 g line -> u = 0, s2 just past the 600 g line -> u = 1), so it
does not change when the scale moves, turns or comes closer. grams = offset_g + slope_g_per_u * u.

    cal = StickerCalibration.from_clicks(s1, s2, p0, p1)   # stickers + the 0 g / 600 g lines on one frozen frame
    cal.tare(u_empty); cal.add_sample(u, 200.0); cal.fit(); cal.save(path)
    cal = load_calibration(path)                            # Calibration (schema 1) or StickerCalibration (schema 2)
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from arm1.lib.geometry import Axis

DEFAULT_CALIB_PATH = "config/scale_calib.json"
SCHEMA = "scale-calib/1"
SCHEMA_STICKERS = "scale-calib/2"

# Least-squares slope needs samples spread along the axis; below this the slope is too noisy to trust.
MIN_FIT_SPREAD_PX = 20.0
# Same idea in sticker units (u = 0..1 between the stickers; 0.03 is about 3.7 mm on a Pesola 600 g).
MIN_FIT_SPREAD_U = 0.03


@dataclass
class FitReport:
    n: int
    slope_g_per_px: float
    offset_g: float
    residuals_g: list[float]        # per sample, reading - truth, after the fit
    max_abs_residual_g: float
    rms_g: float


@dataclass
class Calibration:
    axis: Axis
    g0: float = 0.0                 # grams printed at p0
    g1: float = 600.0               # grams printed at p1
    slope_g_per_px: float = 0.0     # set by from_clicks: (g1 - g0) / axis.length
    offset_g: float = 0.0           # grams at t = 0; from_clicks sets g0
    tare_t_px: float | None = None
    samples: list[tuple[float, float]] = field(default_factory=list)   # (t_px, grams) known-weight samples
    created_at: str = ""            # ISO timestamp
    notes: str = ""

    # ---- construction -------------------------------------------------------------------------
    @classmethod
    def from_clicks(cls, p0, p1, g0: float = 0.0, g1: float = 600.0) -> "Calibration":
        axis = Axis((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))
        length = axis.length
        if length <= 0:
            raise ValueError("p0 and p1 are the same point")
        return cls(
            axis=axis,
            g0=float(g0),
            g1=float(g1),
            slope_g_per_px=(float(g1) - float(g0)) / length,
            offset_g=float(g0),
            created_at=_now_iso(),
        )

    # ---- conversion ---------------------------------------------------------------------------
    def to_grams(self, t_px: float) -> float:
        return self.offset_g + self.slope_g_per_px * float(t_px)

    def to_t(self, grams: float) -> float:
        if self.slope_g_per_px == 0:
            raise ValueError("calibration slope is zero")
        return (float(grams) - self.offset_g) / self.slope_g_per_px

    # ---- adjustment ---------------------------------------------------------------------------
    def tare(self, t_px_at_zero_load: float) -> None:
        """Shift the offset so the given position reads 0 g. The slope is unchanged."""
        t = float(t_px_at_zero_load)
        self.offset_g = -self.slope_g_per_px * t
        self.tare_t_px = t

    def add_sample(self, t_px: float, grams: float) -> None:
        self.samples.append((float(t_px), float(grams)))

    def fit(self) -> FitReport:
        """Refit from the known-weight samples (see module docstring for the rules)."""
        n = len(self.samples)
        if n == 0:
            return FitReport(0, self.slope_g_per_px, self.offset_g, [], 0.0, 0.0)

        t = np.array([s[0] for s in self.samples], dtype=np.float64)
        g = np.array([s[1] for s in self.samples], dtype=np.float64)

        if n >= 2 and float(t.max() - t.min()) >= MIN_FIT_SPREAD_PX:
            # Least squares g = offset + slope * t.
            A = np.column_stack([np.ones_like(t), t])
            (offset, slope), *_ = np.linalg.lstsq(A, g, rcond=None)
            self.slope_g_per_px = float(slope)
            self.offset_g = float(offset)
        else:
            # One sample, or several bunched together: keep the slope, shift the offset onto their mean.
            self.offset_g = float(g.mean() - self.slope_g_per_px * t.mean())

        residuals = [self.to_grams(ti) - gi for ti, gi in zip(t.tolist(), g.tolist())]
        abs_r = [abs(r) for r in residuals]
        return FitReport(
            n=n,
            slope_g_per_px=self.slope_g_per_px,
            offset_g=self.offset_g,
            residuals_g=residuals,
            max_abs_residual_g=max(abs_r),
            rms_g=math.sqrt(sum(r * r for r in residuals) / n),
        )

    def check(self, t_px: float, grams_true: float) -> float:
        """Signed error of the reading at t_px against a known weight (reading - truth), in grams."""
        return self.to_grams(t_px) - float(grams_true)

    # ---- persistence --------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "axis": self.axis.to_dict(),
            "g0": self.g0,
            "g1": self.g1,
            "slope_g_per_px": self.slope_g_per_px,
            "offset_g": self.offset_g,
            "tare_t_px": self.tare_t_px,
            "samples": [[t, g] for t, g in self.samples],
            "created_at": self.created_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        schema = d.get("schema")
        if schema == SCHEMA_STICKERS:
            raise ValueError(f"this is a sticker calibration ({SCHEMA_STICKERS!r}), not a pixel one; "
                             "use gauge_reader.calibration.load_calibration() or StickerCalibration.load()")
        if schema != SCHEMA:
            raise ValueError(f"unsupported calibration schema {schema!r}, expected {SCHEMA!r}")
        tare = d.get("tare_t_px")
        return cls(
            axis=Axis.from_dict(d["axis"]),
            g0=float(d.get("g0", 0.0)),
            g1=float(d.get("g1", 600.0)),
            slope_g_per_px=float(d["slope_g_per_px"]),
            offset_g=float(d["offset_g"]),
            tare_t_px=None if tare is None else float(tare),
            samples=[(float(t), float(g)) for t, g in d.get("samples", [])],
            created_at=str(d.get("created_at", "")),
            notes=str(d.get("notes", "")),
        )

    def save(self, path=DEFAULT_CALIB_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)   # atomic: a crash mid-write never leaves a half-written calibration

    @classmethod
    def load(cls, path=DEFAULT_CALIB_PATH) -> "Calibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class StickerCalibration:
    """u (marker position between the stickers, 0 at s1, 1 at s2) -> grams. Pose-invariant.

    fit() returns a FitReport whose slope_g_per_px field holds the slope in grams per unit u.
    """
    u0: float                       # u of the 0 g line
    u1: float                       # u of the full-scale line
    g0: float = 0.0
    g1: float = 600.0
    slope_g_per_u: float = 0.0      # from_clicks: (g1 - g0) / (u1 - u0)
    offset_g: float = 0.0           # grams at u = 0
    tare_u: float | None = None
    samples: list[tuple[float, float]] = field(default_factory=list)   # (u, grams) known-weight samples
    s1_ref: tuple[float, float] | None = None   # sticker image positions at calibration (orientation hint)
    s2_ref: tuple[float, float] | None = None
    ref_profile: dict | None = None             # look of hand-clicked references (stickers.StickerProfile.to_dict)
    created_at: str = ""
    notes: str = ""
    mode: str = "stickers"

    # ---- construction -------------------------------------------------------------------------
    @classmethod
    def from_clicks(cls, s1, s2, p0, p1, g0: float = 0.0, g1: float = 600.0,
                    ref_profile: dict | None = None) -> "StickerCalibration":
        """s1, s2: sticker centres; p0, p1: points on the 0 g and full-scale lines (all image pixels, one frame)."""
        s1 = (float(s1[0]), float(s1[1]))
        s2 = (float(s2[0]), float(s2[1]))
        u0 = project_u(s1, s2, p0)
        u1 = project_u(s1, s2, p1)
        if abs(u1 - u0) < 1e-9:
            raise ValueError("p0 and p1 project to the same place between the stickers")
        g0, g1 = float(g0), float(g1)
        slope = (g1 - g0) / (u1 - u0)
        return cls(
            u0=u0,
            u1=u1,
            g0=g0,
            g1=g1,
            slope_g_per_u=slope,
            offset_g=g0 - slope * u0,
            s1_ref=s1,
            s2_ref=s2,
            ref_profile=ref_profile,
            created_at=_now_iso(),
        )

    # ---- conversion ---------------------------------------------------------------------------
    def to_grams(self, u: float) -> float:
        return self.offset_g + self.slope_g_per_u * float(u)

    def to_u(self, grams: float) -> float:
        if self.slope_g_per_u == 0:
            raise ValueError("calibration slope is zero")
        return (float(grams) - self.offset_g) / self.slope_g_per_u

    # ---- adjustment ---------------------------------------------------------------------------
    def tare(self, u_at_zero_load: float) -> None:
        """Shift the offset so the given u reads 0 g. The slope is unchanged."""
        u = float(u_at_zero_load)
        self.offset_g = -self.slope_g_per_u * u
        self.tare_u = u

    def add_sample(self, u: float, grams: float) -> None:
        self.samples.append((float(u), float(grams)))

    def fit(self) -> FitReport:
        """Same rules as Calibration.fit, with a minimum u spread of MIN_FIT_SPREAD_U for a slope refit."""
        n = len(self.samples)
        if n == 0:
            return FitReport(0, self.slope_g_per_u, self.offset_g, [], 0.0, 0.0)

        u = np.array([s[0] for s in self.samples], dtype=np.float64)
        g = np.array([s[1] for s in self.samples], dtype=np.float64)

        if n >= 2 and float(u.max() - u.min()) >= MIN_FIT_SPREAD_U:
            A = np.column_stack([np.ones_like(u), u])
            (offset, slope), *_ = np.linalg.lstsq(A, g, rcond=None)
            self.slope_g_per_u = float(slope)
            self.offset_g = float(offset)
        else:
            self.offset_g = float(g.mean() - self.slope_g_per_u * u.mean())

        residuals = [self.to_grams(ui) - gi for ui, gi in zip(u.tolist(), g.tolist())]
        return FitReport(
            n=n,
            slope_g_per_px=self.slope_g_per_u,
            offset_g=self.offset_g,
            residuals_g=residuals,
            max_abs_residual_g=max(abs(r) for r in residuals),
            rms_g=math.sqrt(sum(r * r for r in residuals) / n),
        )

    def check(self, u: float, grams_true: float) -> float:
        """Signed error of the reading at u against a known weight (reading - truth), in grams."""
        return self.to_grams(u) - float(grams_true)

    def u_range(self, margin: float = 0.05) -> tuple[float, float]:
        """Plausible u interval for the marker: the 0 g..full-scale span (plus tare/samples), padded by margin."""
        us = [self.u0, self.u1]
        if self.tare_u is not None:
            us.append(self.tare_u)
        us.extend(s[0] for s in self.samples)
        return min(us) - float(margin), max(us) + float(margin)

    # ---- persistence --------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_STICKERS,
            "mode": "stickers",
            "u0": self.u0,
            "u1": self.u1,
            "g0": self.g0,
            "g1": self.g1,
            "slope_g_per_u": self.slope_g_per_u,
            "offset_g": self.offset_g,
            "tare_u": self.tare_u,
            "samples": [[u, g] for u, g in self.samples],
            "s1_ref": None if self.s1_ref is None else list(self.s1_ref),
            "s2_ref": None if self.s2_ref is None else list(self.s2_ref),
            "ref_profile": self.ref_profile,
            "created_at": self.created_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StickerCalibration":
        schema = d.get("schema")
        if schema == SCHEMA:
            raise ValueError(f"this is a pixel calibration ({SCHEMA!r}), not a sticker one; "
                             "use gauge_reader.calibration.load_calibration() or Calibration.load()")
        if schema != SCHEMA_STICKERS or d.get("mode", "stickers") != "stickers":
            raise ValueError(f"unsupported calibration schema {schema!r} / mode {d.get('mode')!r}, "
                             f"expected {SCHEMA_STICKERS!r} / 'stickers'")
        tare = d.get("tare_u")

        def pt(v):
            return None if v is None else (float(v[0]), float(v[1]))

        return cls(
            u0=float(d["u0"]),
            u1=float(d["u1"]),
            g0=float(d.get("g0", 0.0)),
            g1=float(d.get("g1", 600.0)),
            slope_g_per_u=float(d["slope_g_per_u"]),
            offset_g=float(d["offset_g"]),
            tare_u=None if tare is None else float(tare),
            samples=[(float(u), float(g)) for u, g in d.get("samples", [])],
            s1_ref=pt(d.get("s1_ref")),
            s2_ref=pt(d.get("s2_ref")),
            ref_profile=d.get("ref_profile"),
            created_at=str(d.get("created_at", "")),
            notes=str(d.get("notes", "")),
            mode="stickers",
        )

    def save(self, path=DEFAULT_CALIB_PATH) -> None:
        _atomic_write_json(Path(path), self.to_dict())

    @classmethod
    def load(cls, path=DEFAULT_CALIB_PATH) -> "StickerCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def project_u(s1, s2, point) -> float:
    """Image point -> u: its projection onto the line s1 -> s2, as a fraction (0 at s1, 1 at s2)."""
    a = np.array(s1, dtype=np.float64)
    d = np.array(s2, dtype=np.float64) - a
    dd = float(d @ d)
    if dd == 0:
        raise ValueError("s1 and s2 are the same point")
    return float((np.array(point, dtype=np.float64) - a) @ d / dd)


def load_calibration(path=DEFAULT_CALIB_PATH) -> "Calibration | StickerCalibration":
    """Load either kind of calibration file, chosen by its schema/mode."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = d.get("schema") if isinstance(d, dict) else None
    if schema == SCHEMA:
        return Calibration.from_dict(d)
    if schema == SCHEMA_STICKERS and d.get("mode", "stickers") == "stickers":
        return StickerCalibration.from_dict(d)
    mode = d.get("mode") if isinstance(d, dict) else None
    raise ValueError(f"unknown calibration file {path}: schema {schema!r}, mode {mode!r}")


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)   # atomic: a crash mid-write never leaves a half-written calibration


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
