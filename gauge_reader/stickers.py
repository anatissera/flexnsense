"""Relative reading of a hand-held scale from two red reference stickers.

Two red round stickers sit on the tube: sticker 1 just before the 0 g line, sticker 2 just past the 600 g line.
The red marker (same red) slides between them. Instead of a fixed calibrated axis, every frame we find the
triplet (s1, marker, s2) and read u = projection of the marker onto s1 -> s2 as a fraction (0 at s1, 1 at s2).
The app maps u to grams with its calibration (u0 = u of the 0 g line, u1 = u of the 600 g line).

    tr = StickerTracker(StickerConfig(u_range=(u0 - 0.05, u1 + 0.05)), s1_hint=s1, s2_hint=s2)
    d = tr.detect(frame_bgr)
    if d.found: grams = (d.u - u0) / (u1 - u0) * 600

How the three red blobs are told apart:
  * candidates = connected components of detector.red_mask (skin is not red enough to pass it), with a
    redness-weighted sub-pixel centroid and binary second moments;
  * stickers are round dots of similar area (blur may stretch them, mostly ALONG the line);
  * the marker is a band ACROSS the tube: measured in the s1 -> s2 frame its spread across the line is
    clearly larger than along it, and larger than the stickers' own across/along ratio;
  * the marker lies strictly between the stickers, near their line, with u inside u_range;
  * while tracking, |s1 - s2| must stay close to the previous frame's distance.
With one sticker covered only two red blobs remain on the line, so no triplet exists and the result is
found=False, reason "sticker-hidden" -- the tracker never invents the missing sticker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from gauge_reader.detector import DetectorConfig, red_mask
from lib.geometry import Axis

Point = tuple[float, float]


@dataclass
class StickerConfig:
    red: DetectorConfig = field(default_factory=DetectorConfig)   # colour thresholds, reuse red_mask
    min_area_px: int = 8
    max_area_px: int = 20000
    max_offaxis_frac: float = 0.05     # max distance of the marker from the s1-s2 line, as a fraction of |s1-s2|
    min_sticker_dist_px: float = 60.0
    max_dist_change: float = 0.25      # max relative change of |s1-s2| vs the previous frame (when tracking)
    u_range: tuple[float, float] = (0.0, 1.0)   # plausible marker u; the app sets (u0-0.05, u1+0.05) from calibration
    roi_margin_px: float = 150.0       # search near the last position; whole frame when lost
    # --- extra knobs ---
    marker_min_aspect: float = 0.0     # marker shape rule OFF: the real Pesola marker is a capsule ALONG the tube (set 1.5 for a band across)
    marker_vs_sticker_aspect: float = 0.0   # (off with the rule above)
    sticker_max_aspect: float = 2.5    # stickers: std across / std along at most this (round dot, maybe blurred)
    sticker_max_elong: float = 3.0     # stickers: major / minor std at most this (any direction)
    sticker_area_ratio: float = 2.5    # max area ratio between the two stickers
    marker_min_area_frac: float = 0.25  # marker mask area >= this x mean sticker area (a smeared remnant is not trusted)
    max_candidates: int = 16           # largest red blobs considered for triplets
    track_max_gap: int = 15            # frames after the last success during which max_dist_change is enforced
    min_redness: float = 68.0          # mean R - max(G,B) of a blob's core: real stickers/marker 107-145, skin <= 64 (in view), heavy blur >= 73
    line_clearance_frac: float = 0.04  # calibration: a sticker must be this far (fraction of 0..600 length) beyond the
                                       # 0 g / 600 g click, so the marker resting on the 0 g line is not taken for sticker 1
    click_offaxis_frac: float = 0.12   # calibration: stickers may sit this far (fraction of 0..600 length) beside the
                                       # clicked line (users click the printed ticks at the tube edge, not the marker track)
    ambiguity_du: float = 0.01         # two different readings with scores this close -> "ambiguous"
    ambiguity_score: float = 0.15


@dataclass
class StickerDetection:
    found: bool
    u: float | None
    s1: Point | None
    s2: Point | None
    marker: Point | None
    sticker_dist_px: float | None
    axis: Axis | None                  # s1 -> s2
    confidence: float
    reason: str = ""                   # "" if found; else "no-scale" | "sticker-hidden" | "no-marker" | "ambiguous"
    blobs: list[tuple[float, float, int]] = field(default_factory=list)   # red candidates (x, y, area) for overlay

    @classmethod
    def fail(cls, reason: str, blobs=None) -> "StickerDetection":
        return cls(False, None, None, None, None, None, None, 0.0, reason, list(blobs or []))


@dataclass
class _Blob:
    c: np.ndarray        # sub-pixel centre (x, y)
    area: int
    cov: np.ndarray      # 2x2 covariance of the binary component (px^2)
    redness: float       # mean R - max(G, B) over the core
    clipped: bool = False  # touches the image border: centre is biased, never use it as a sticker or marker

    def spread(self, d: np.ndarray) -> float:
        """Standard deviation of the blob along unit direction d (incl. 1/12 px^2 pixel quantisation)."""
        return float(np.sqrt(max(d @ self.cov @ d, 0.0) + 1.0 / 12.0))

    @property
    def elong(self) -> float:
        ev = np.linalg.eigvalsh(self.cov) + 1.0 / 12.0
        return float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))


def _cfg(config: StickerConfig | None) -> StickerConfig:
    return config if config is not None else StickerConfig()


def _blobs(img: np.ndarray, cfg: StickerConfig, offset=(0, 0)) -> list[_Blob]:
    """Red components of img with sub-pixel centres (in full-frame coordinates via offset)."""
    rc = cfg.red
    mask = red_mask(img, rc)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    idx = [i for i in range(1, n) if cfg.min_area_px <= stats[i, cv2.CC_STAT_AREA] <= cfg.max_area_px]
    idx.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    idx = idx[: cfg.max_candidates]
    dil = rc.centre_dilate_px
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1)) if dil > 0 else None
    pad = int(dil) + 1
    H, W = mask.shape
    out: list[_Blob] = []
    for i in idx:
        x0, y0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        xa, ya, xb, yb = max(0, x0 - pad), max(0, y0 - pad), min(W, x0 + w + pad), min(H, y0 + h + pad)
        core = (labels[ya:yb, xa:xb] == i).astype(np.uint8)
        comp = cv2.dilate(core, kern) if kern is not None else core
        roi = img[ya:yb, xa:xb].astype(np.float32)
        redness = roi[..., 2] - np.maximum(roi[..., 0], roi[..., 1])
        wgt = np.clip(redness - rc.weight_offset, 0, None) * comp
        s = float(wgt.sum())
        if s <= 0:
            continue
        ys, xs = np.mgrid[ya:yb, xa:xb]
        cx = float((wgt * xs).sum() / s) + offset[0]
        cy = float((wgt * ys).sum() / s) + offset[1]
        m = cv2.moments(core, binaryImage=True)
        a = max(m["m00"], 1.0)
        cov = np.array([[m["mu20"] / a, m["mu11"] / a], [m["mu11"] / a, m["mu02"] / a]])
        clipped = x0 <= 0 or y0 <= 0 or x0 + w >= W or y0 + h >= H
        out.append(_Blob(np.array([cx, cy]), int(stats[i, cv2.CC_STAT_AREA]), cov, float(redness[core > 0].mean()), clipped))
    return out


def _u(point, s1, s2) -> float:
    a, b, p = (np.asarray(v, dtype=np.float64) for v in (s1, s2, point))
    d = b - a
    return float((p - a) @ d / max(d @ d, 1e-12))


def _pt(v) -> Point:
    return (float(v[0]), float(v[1]))


class StickerTracker:
    def __init__(self, config: StickerConfig | None = None, s1_hint=None, s2_hint=None):
        self.config = _cfg(config)
        self.s1_hint = None if s1_hint is None else np.asarray(s1_hint, dtype=np.float64)
        self.s2_hint = None if s2_hint is None else np.asarray(s2_hint, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        """Forget the tracked state (orientation falls back to the hints)."""
        self._s1: np.ndarray | None = None      # last good sticker positions (kept through lost frames)
        self._s2: np.ndarray | None = None
        self._dist: float | None = None
        self._gap = 10 ** 9                      # frames since the last success

    @staticmethod
    def u_of(point, s1, s2) -> float:
        """Projection of point onto s1 -> s2 as a fraction (0 at s1, 1 at s2)."""
        return _u(point, s1, s2)

    # ------------------------------------------------------------------ detection
    def detect(self, frame_bgr: np.ndarray) -> StickerDetection:
        cfg = self.config
        det = None
        if self._s1 is not None and self._gap <= 1:  # tracking: search near the last position first
            lo = np.minimum(self._s1, self._s2) - cfg.roi_margin_px
            hi = np.maximum(self._s1, self._s2) + cfg.roi_margin_px
            H, W = frame_bgr.shape[:2]
            x0, y0 = int(max(0, np.floor(lo[0]))), int(max(0, np.floor(lo[1])))
            x1, y1 = int(min(W, np.ceil(hi[0]))), int(min(H, np.ceil(hi[1])))
            if x1 - x0 > 8 and y1 - y0 > 8:
                det = self._detect_in(frame_bgr[y0:y1, x0:x1], (x0, y0))
                if not det.found:
                    det = None
        if det is None:
            det = self._detect_in(frame_bgr, (0, 0))
        if det.found:
            self._s1, self._s2 = np.array(det.s1), np.array(det.s2)
            self._dist, self._gap = det.sticker_dist_px, 0
        else:
            self._gap += 1
        return det

    def _refs(self):
        if self._s1 is not None:
            return self._s1, self._s2
        if self.s1_hint is not None and self.s2_hint is not None:
            return self.s1_hint, self.s2_hint
        return None

    def _detect_in(self, img: np.ndarray, offset) -> StickerDetection:
        cfg = self.config
        all_blobs = _blobs(img, cfg, offset)
        overlay = [(float(b.c[0]), float(b.c[1]), b.area) for b in all_blobs]
        blobs = [b for b in all_blobs if not b.clipped and b.redness >= cfg.min_redness]
        if not blobs:
            return StickerDetection.fail("no-scale", overlay)
        tracking = self._dist is not None and self._gap < cfg.track_max_gap
        refs = self._refs()
        lo_u, hi_u = cfg.u_range

        cands = []   # (score, u, s1, s2, marker, dist, orientation_undecided)
        n = len(blobs)
        for ia in range(n):
            for ib in range(ia + 1, n):
                a, b = blobs[ia], blobs[ib]
                dvec = b.c - a.c
                dist = float(np.hypot(*dvec))
                if dist < cfg.min_sticker_dist_px:
                    continue
                if max(a.area, b.area) > cfg.sticker_area_ratio * min(a.area, b.area):
                    continue
                dchg = abs(dist - self._dist) / self._dist if tracking else 0.0
                if dchg > cfg.max_dist_change:
                    continue
                d = dvec / dist
                nrm = np.array([-d[1], d[0]])
                asp_a, asp_b = a.spread(nrm) / a.spread(d), b.spread(nrm) / b.spread(d)
                if max(asp_a, asp_b) > cfg.sticker_max_aspect or max(a.elong, b.elong) > cfg.sticker_max_elong:
                    continue
                for ic in range(n):
                    if ic in (ia, ib):
                        continue
                    c = blobs[ic]
                    if c.area < cfg.marker_min_area_frac * 0.5 * (a.area + b.area):
                        continue
                    rel = c.c - a.c
                    uc = float(rel @ d) / dist
                    off = abs(float(rel @ nrm)) / dist
                    if not (0.0 < uc < 1.0) or off > cfg.max_offaxis_frac:
                        continue
                    asp_c = c.spread(nrm) / c.spread(d)
                    if asp_c < cfg.marker_min_aspect or asp_c < cfg.marker_vs_sticker_aspect * max(asp_a, asp_b):
                        continue
                    # orientation: which end is sticker 1 (the 0 g end)
                    s1, s2, u = a, b, uc
                    undecided = False
                    if refs is not None:
                        r1, r2 = refs
                        keep = np.hypot(*(a.c - r1)) + np.hypot(*(b.c - r2))
                        swap = np.hypot(*(b.c - r1)) + np.hypot(*(a.c - r2))
                        if swap < keep:
                            s1, s2, u = b, a, 1.0 - uc
                        if not (lo_u <= u <= hi_u):
                            continue
                    else:
                        ok_keep, ok_swap = lo_u <= uc <= hi_u, lo_u <= 1.0 - uc <= hi_u
                        if not (ok_keep or ok_swap):
                            continue
                        if ok_swap and not ok_keep:
                            s1, s2, u = b, a, 1.0 - uc
                        elif ok_keep and ok_swap:  # undecidable: s1 = the end left (then top) in the image
                            undecided = True
                            if (b.c[0], b.c[1]) < (a.c[0], a.c[1]):
                                s1, s2, u = b, a, 1.0 - uc
                    score = (off / cfg.max_offaxis_frac
                             + (dchg / cfg.max_dist_change if tracking else 0.0)
                             + max(asp_a, asp_b) / cfg.sticker_max_aspect
                             + cfg.marker_min_aspect / asp_c
                             + 0.5 * (max(a.area, b.area) / min(a.area, b.area) - 1.0))
                    cands.append((score, u, s1, s2, c, dist, undecided))

        if not cands:
            return StickerDetection.fail(self._why(blobs), overlay)
        cands.sort(key=lambda t: t[0])
        best = cands[0]
        for other in cands[1:]:
            if other[0] - best[0] > cfg.ambiguity_score:
                break
            if abs(other[1] - best[1]) > cfg.ambiguity_du:
                return StickerDetection.fail("ambiguous", overlay)
        score, u, s1, s2, mk, dist, undecided = best
        red = float(np.clip(min(s1.redness, s2.redness, mk.redness) / cfg.red.redness_full_conf, 0, 1))
        conf = red * float(np.clip(1.0 - 0.15 * score, 0.1, 1.0)) * (0.5 if undecided else 1.0)
        p1, p2 = _pt(s1.c), _pt(s2.c)
        return StickerDetection(True, float(u), p1, p2, _pt(mk.c), float(dist), Axis(p1, p2), conf, "", overlay)

    def _why(self, blobs: list[_Blob]) -> str:
        """Best guess why no triplet was found. Never a reading, only a reason for the UI."""
        cfg = self.config
        if not blobs:
            return "no-scale"
        # a band-shaped red blob (the marker) is visible but no sticker pair frames it -> a sticker is covered
        if any(b.elong >= max(cfg.marker_min_aspect * 1.2, 2.0) for b in blobs):
            return "sticker-hidden"
        # two round stickers at a plausible distance but nothing marker-like between them
        n = len(blobs)
        for ia in range(n):
            for ib in range(ia + 1, n):
                a, b = blobs[ia], blobs[ib]
                dist = float(np.hypot(*(b.c - a.c)))
                if dist < cfg.min_sticker_dist_px or max(a.area, b.area) > cfg.sticker_area_ratio * min(a.area, b.area):
                    continue
                if max(a.elong, b.elong) <= cfg.sticker_max_elong:
                    return "no-marker"
        return "sticker-hidden" if self._s1 is not None and n <= 2 else "no-scale"


# ---------------------------------------------------------------------- calibration helper
def find_stickers_near_line(frame_bgr: np.ndarray, p0, p1, config: StickerConfig | None = None):
    """At calibration: the user clicked the 0 g line (p0) and the 600 g line (p1). Return (s1, s2) = sub-pixel
    centres of the red blobs on that line just beyond p0 (s1) and just beyond p1 (s2); None if not both found."""
    cfg = _cfg(config)
    axis = Axis(_pt(p0), _pt(p1))
    L = axis.length
    blobs = _blobs(frame_bgr, cfg)
    best1 = best2 = None
    for b in blobs:
        t, nn = axis.project(_pt(b.c))
        if abs(nn) > max(cfg.click_offaxis_frac * L, 6.0) or b.elong > cfg.sticker_max_elong:
            continue
        if b.clipped or b.redness < cfg.min_redness:
            continue
        gap = cfg.line_clearance_frac * L
        if -0.35 * L < t < -gap and (best1 is None or t > best1[0]):
            best1 = (t, b)
        elif L + gap < t < 1.35 * L and (best2 is None or t < best2[0]):
            best2 = (t, b)
    if best1 is None or best2 is None:
        return None
    return _pt(best1[1].c), _pt(best2[1].c)
