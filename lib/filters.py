"""Smoothing of per-frame gram readings: a 1D constant-velocity Kalman filter with outlier gating.

State x = [grams, grams/s]. Timestamps may be irregular; dt comes from them.
Process noise is white acceleration with std `accel_std_g_s2` (discrete white-noise-acceleration model).

Outliers: a measurement whose innovation exceeds `outlier_sigma` standard deviations is held back. A single
spike is dropped. If `outlier_persist` consecutive measurements are all held back, it is a real load change:
the filter is re-initialised from those held-back measurements, so a step is followed in a few frames instead
of being smoothed over for a second. Smaller changes pass the gate; if `outlier_persist` consecutive accepted
innovations exceed `maneuver_sigma` with the same sign, the filter is re-initialised the same way (a 10 g
step is then followed in ~0.07 s instead of ~1 s).

Default tuning (30 fps, detector noise ~2 g): measurement_std 2 g, accel_std 20 g/s^2 -> filtered std is
~0.3x the raw noise at rest, |velocity| at rest stays < 5 g/s, and large steps are followed in 3 frames.

Gaps: `raw_grams=None` only predicts. If no measurement has arrived for longer than `max_gap_s`, the output
is None and the next measurement re-initialises the filter.

Stability: raw readings over the last `stable_window_s` must cover >= 80 % of it, their std must be below
`stable_std_g`, and |velocity| must be below `stable_max_velocity_g_s`. Held-back outliers count towards the
coverage but not the std (a single dropped spike does not make a resting load "unstable").
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

STABLE_COVERAGE = 0.8
_INIT_VEL_STD_G_S = 1000.0     # velocity unknown on (re)initialisation


@dataclass
class FilteredReading:
    grams: float | None         # filtered value; None when there is no recent measurement
    raw: float | None
    velocity_g_s: float
    stable: bool
    std_g: float                # std of accepted raw readings over the stable window (nan if too few)


class ReadingFilter:
    def __init__(
        self,
        measurement_std_g: float = 2.0,
        accel_std_g_s2: float = 20.0,
        stable_window_s: float = 0.5,
        stable_std_g: float = 4.0,          # ~2x measurement_std_g
        stable_max_velocity_g_s: float = 5.0,
        max_gap_s: float = 0.5,
        outlier_sigma: float = 6.0,
        outlier_persist: int = 3,
        maneuver_sigma: float = 3.0,
    ):
        self.measurement_std_g = float(measurement_std_g)
        self.accel_std_g_s2 = float(accel_std_g_s2)
        self.stable_window_s = float(stable_window_s)
        self.stable_std_g = float(stable_std_g)
        self.stable_max_velocity_g_s = float(stable_max_velocity_g_s)
        self.max_gap_s = float(max_gap_s)
        self.outlier_sigma = float(outlier_sigma)
        self.outlier_persist = max(1, int(outlier_persist))
        self.maneuver_sigma = float(maneuver_sigma)
        self.reset()

    # ------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self._x: np.ndarray | None = None          # [grams, grams/s]
        self._P: np.ndarray | None = None
        self._t: float | None = None                # time the state refers to
        self._last_meas_t: float | None = None      # time of the last non-None raw reading
        self._pending: list[tuple[float, float]] = []   # held-back (t, z) outliers
        self._drift: list[tuple[float, float]] = []     # consecutive same-sign > maneuver_sigma innovations
        self._drift_sign = 0.0
        # (t, z, accepted) raw readings for the stability window; accepted is a 1-element list (mutable).
        self._hist: deque[tuple[float, float, list[bool]]] = deque()

    @property
    def _R(self) -> float:
        return self.measurement_std_g ** 2

    def _init(self, t: float, z: float) -> None:
        self._x = np.array([z, 0.0])
        self._P = np.diag([self._R, _INIT_VEL_STD_G_S ** 2])
        self._t = t

    def _predict(self, t: float) -> None:
        dt = t - self._t
        if dt <= 0:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        q = self.accel_std_g_s2 ** 2
        Q = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        self._t = t

    def _innovation(self, z: float) -> tuple[float, float]:
        return z - self._x[0], self._P[0, 0] + self._R

    def _correct(self, z: float) -> None:
        y, S = self._innovation(z)
        K = self._P[:, 0] / S
        self._x = self._x + K * y
        # Simple covariance update is fine for a 2-state filter; symmetrise to keep it numerically clean.
        P = self._P - np.outer(K, self._P[0, :])
        self._P = (P + P.T) / 2

    # ------------------------------------------------------------------------------------------
    def update(self, raw_grams: float | None, timestamp_s: float) -> FilteredReading:
        t = float(timestamp_s)
        raw = None if raw_grams is None or not math.isfinite(raw_grams) else float(raw_grams)

        # Lost track for too long: forget the state, the next measurement starts afresh.
        if self._last_meas_t is not None and t - self._last_meas_t > self.max_gap_s:
            self._x = None
            self._pending.clear()
            self._drift.clear()

        if raw is not None:
            self._last_meas_t = t
            flag = [True]
            self._hist.append((t, raw, flag))
            if self._x is None:
                self._init(t, raw)
            else:
                self._predict(t)
                y, S = self._innovation(raw)
                if abs(y) > self.outlier_sigma * math.sqrt(S):
                    flag[0] = False
                    self._pending.append((t, raw))
                    if len(self._pending) >= self.outlier_persist:
                        self._reinit_from(self._pending)
                else:
                    self._pending.clear()
                    self._correct(raw)
                    self._track_drift(t, raw, y, S)
        elif self._x is not None:
            self._predict(t)

        self._trim_hist(t)
        return self._reading(raw, t)

    def _track_drift(self, t: float, z: float, y: float, S: float) -> None:
        """Smaller load changes pass the outlier gate but bias the innovations; a low-noise filter would take
        a second to catch up. Several consecutive same-sign innovations beyond maneuver_sigma -> restart."""
        if abs(y) > self.maneuver_sigma * math.sqrt(S):
            sign = math.copysign(1.0, y)
            if sign != self._drift_sign:
                self._drift = []
                self._drift_sign = sign
            self._drift.append((t, z))
            if len(self._drift) >= self.outlier_persist:
                self._reinit_from(self._drift)
        else:
            self._drift = []
            self._drift_sign = 0.0

    def _reinit_from(self, points: list[tuple[float, float]]) -> None:
        """A persistent deviation is a real change: restart the filter from those readings (a short line fit)."""
        pend = list(points)
        self._pending = []
        self._drift = []
        self._drift_sign = 0.0
        self._init(*pend[0])
        for tp, zp in pend[1:]:
            self._predict(tp)
            self._correct(zp)
        held = {tp for tp, _ in pend}
        for th, _, flag in self._hist:
            if th in held:
                flag[0] = True

    def _trim_hist(self, t: float) -> None:
        while self._hist and self._hist[0][0] < t - self.stable_window_s - 1e-9:
            self._hist.popleft()

    def _reading(self, raw: float | None, t: float) -> FilteredReading:
        if self._x is None:
            return FilteredReading(None, raw, 0.0, False, float("nan"))
        grams = float(self._x[0])
        vel = float(self._x[1])

        W = self.stable_window_s
        times = [h[0] for h in self._hist]
        # Coverage: the sum of gaps between consecutive readings that are short (<= 25 % of the window).
        covered = sum(b - a for a, b in zip(times, times[1:]) if b - a <= 0.25 * W)
        accepted = [h[1] for h in self._hist if h[2][0]]
        std = float(np.std(accepted, ddof=1)) if len(accepted) >= 3 else float("nan")
        stable = (
            covered >= STABLE_COVERAGE * W
            and math.isfinite(std)
            and std < self.stable_std_g
            and abs(vel) < self.stable_max_velocity_g_s
            and not self._pending
            and not self._drift   # the load has just started moving: the estimate is about to lag
        )
        return FilteredReading(grams, raw, vel, bool(stable), std)
