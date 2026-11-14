"""Red marker detector for the Pesola Medio-Line scale.

Only a narrow strip along the calibrated axis is processed: an affine warp rectifies it so that strip
column x corresponds to axis coordinate t = x - margin and strip row y to n = y - half_width. The marker
is segmented with an HSV + Lab a* red mask, the best connected component is chosen, and its centre is the
redness-weighted mean of strip coordinates over the (slightly dilated) component. The soft weight
(R - max(G, B), offset and clipped) is linear in the anti-aliased edge blend, which gives sub-pixel accuracy.

    det = RedMarkerDetector(axis)
    d = det.detect(frame_bgr)
    if d.found: grams = d.t_px / axis.length * 600
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision.geometry import Axis


@dataclass
class DetectorConfig:
    strip_half_width_px: float = 25.0   # strip half-width across the axis (n direction)
    margin_px: float = 30.0             # strip extends this far before p0 and past p1 (t direction)
    hsv_low1: tuple = (0, 100, 60)      # "weak" red: blurred / anti-aliased edges
    hsv_high1: tuple = (16, 255, 255)   # up to hue 16: the real Pesola marker is orange-red (~12)
    hsv_low2: tuple = (170, 100, 60)
    sat_strong: int = 0                 # off by default (fingertips reach it too). If >0, a blob needs pixels this saturated (real stickers ~220, marker ~200;
                                        # skin ~130, wood ~75 never reach it): hysteresis like Canny
    hsv_high2: tuple = (180, 255, 255)
    lab_a_min: int = 150                # OpenCV 8-bit Lab a* threshold (128 = neutral)
    min_area_px: int = 6
    max_area_px: int = 20000
    morph_kernel: int = 3
    # sub-pixel centre estimator
    weight_offset: float = 20.0         # soft weight = max(0, R - max(G, B) - weight_offset)
    centre_dilate_px: int = 2           # grow the chosen component so anti-aliased edges are weighted too
    redness_full_conf: float = 120.0    # mean (R - max(G,B)) at which redness confidence saturates


@dataclass
class Detection:
    found: bool
    t_px: float | None
    n_px: float | None
    point: tuple[float, float] | None
    area_px: int
    confidence: float

    @classmethod
    def none(cls, area_px: int = 0) -> "Detection":
        return cls(False, None, None, None, area_px, 0.0)


def _cfg(config: DetectorConfig | None) -> DetectorConfig:
    return config if config is not None else DetectorConfig()


def red_mask(img_bgr: np.ndarray, config: DetectorConfig | None = None) -> np.ndarray:
    """uint8 0/255 mask of red pixels: (HSV range 1 OR range 2) AND Lab a* >= lab_a_min, then open/close."""
    c = _cfg(config)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(c.hsv_low1, np.uint8), np.array(c.hsv_high1, np.uint8))
    m |= cv2.inRange(hsv, np.array(c.hsv_low2, np.uint8), np.array(c.hsv_high2, np.uint8))
    a = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[..., 1]
    m &= np.where(a >= c.lab_a_min, np.uint8(255), np.uint8(0))
    if c.morph_kernel and c.morph_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c.morph_kernel, c.morph_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if c.sat_strong and c.sat_strong > min(c.hsv_low1[1], c.hsv_low2[1]):
        # hysteresis: keep only weak-red components that contain at least one strongly saturated pixel
        strong = (m > 0) & (hsv[..., 1] >= c.sat_strong)
        if not strong.any():
            return np.zeros_like(m)
        n, lab = cv2.connectedComponents(m, connectivity=8)
        keep = np.zeros(n, bool)
        keep[np.unique(lab[strong])] = True
        keep[0] = False
        m = np.where(keep[lab], np.uint8(255), np.uint8(0))
    return m


def find_red_blobs(frame_bgr: np.ndarray, config: DetectorConfig | None = None, min_area: int = 20) -> list[tuple[float, float, int]]:
    """All red blobs in the full frame as (x, y, area), largest first. Useful for calibration / overlays."""
    m = red_mask(frame_bgr, config)
    n, _, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    blobs = [(float(cents[i][0]), float(cents[i][1]), int(stats[i, cv2.CC_STAT_AREA]))
             for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    blobs.sort(key=lambda b: -b[2])
    return blobs


class RedMarkerDetector:
    def __init__(self, axis: Axis, config: DetectorConfig | None = None):
        self.axis = axis
        self.config = _cfg(config)
        self.last_strip: np.ndarray | None = None
        self.last_mask: np.ndarray | None = None
        c = self.config
        u, nrm = axis.unit, axis.normal
        self._hw = float(c.strip_half_width_px)
        self._margin = float(c.margin_px)
        self._size = (int(np.ceil(axis.length + 2 * self._margin)) + 1, int(np.ceil(2 * self._hw)) + 1)  # (w, h)
        # strip (x, y) -> image: p0 + (x - margin) * u + (y - hw) * n   (used with WARP_INVERSE_MAP)
        ox = axis.p0[0] - self._margin * u[0] - self._hw * nrm[0]
        oy = axis.p0[1] - self._margin * u[1] - self._hw * nrm[1]
        self._M = np.array([[u[0], nrm[0], ox], [u[1], nrm[1], oy]], dtype=np.float64)
        k = 2 * int(c.centre_dilate_px) + 1
        self._dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) if c.centre_dilate_px > 0 else None

    def strip_corners(self) -> np.ndarray:
        """4x2 float array: strip outline in image coordinates (t0,-hw), (t1,-hw), (t1,+hw), (t0,+hw)."""
        t0, t1 = -self._margin, self.axis.length + self._margin
        pts = [self.axis.point_at(t0, -self._hw), self.axis.point_at(t1, -self._hw),
               self.axis.point_at(t1, self._hw), self.axis.point_at(t0, self._hw)]
        return np.array(pts, dtype=np.float64)

    def rectify(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(frame_bgr, self._M, self._size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        c = self.config
        strip = self.rectify(frame_bgr)
        mask = red_mask(strip, c)
        self.last_strip, self.last_mask = strip, mask

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best, best_key = -1, None
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < c.min_area_px or area > c.max_area_px:
                continue
            cy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2.0
            key = (area, -abs(cy - self._hw))
            if best_key is None or key > best_key:
                best, best_key = i, key
        if best < 0:
            return Detection.none()

        area = int(stats[best, cv2.CC_STAT_AREA])
        x0, y0 = int(stats[best, cv2.CC_STAT_LEFT]), int(stats[best, cv2.CC_STAT_TOP])
        w, h = int(stats[best, cv2.CC_STAT_WIDTH]), int(stats[best, cv2.CC_STAT_HEIGHT])
        pad = int(c.centre_dilate_px) + 1
        xa, ya = max(0, x0 - pad), max(0, y0 - pad)
        xb, yb = min(strip.shape[1], x0 + w + pad), min(strip.shape[0], y0 + h + pad)

        comp = (labels[ya:yb, xa:xb] == best).astype(np.uint8)
        if self._dil is not None:
            comp = cv2.dilate(comp, self._dil)
        roi = strip[ya:yb, xa:xb].astype(np.float32)
        redness = roi[..., 2] - np.maximum(roi[..., 0], roi[..., 1])
        wgt = np.clip(redness - c.weight_offset, 0, None) * comp
        s = float(wgt.sum())
        if s <= 0:
            return Detection.none(area)
        ys, xs = np.mgrid[ya:yb, xa:xb]
        cx = float((wgt * xs).sum() / s)
        cy = float((wgt * ys).sum() / s)
        t, nn = cx - self._margin, cy - self._hw

        # confidence: redness of the core pixels x area plausibility (expected ~1.5 x 9.6 mm marker band)
        core = labels[ya:yb, xa:xb] == best
        red_conf = float(np.clip(redness[core].mean() / c.redness_full_conf, 0, 1))
        ppm = self.axis.length / 100.0
        expected = 1.5 * ppm * min(9.6 * ppm, 2 * self._hw)
        ratio = area / max(expected, 1.0)
        area_conf = float(np.clip(min(ratio, 1.0 / ratio) * 1.5, 0, 1)) if ratio > 0 else 0.0
        edge_pen = 0.8 if (t < -0.5 * self._margin or t > self.axis.length + 0.5 * self._margin) else 1.0
        conf = red_conf * (0.5 + 0.5 * area_conf) * edge_pen
        return Detection(True, t, nn, self.axis.point_at(t, nn), area, conf)
