"""Camera reader for the Pesola Medio-Line 600 g spring scale: find the red marker, show and log grams.

    python -m gauge_reader calibrate                 # stickers mode: click 0 g and 600 g, confirm stickers, tare, save
    python -m gauge_reader calibrate --mode fixed    # scale clamped still: fixed axis from the two clicks
    python -m gauge_reader run --csv out.csv         # live reading window + CSV log (mode from the calib file)
    python -m gauge_reader run --headless --video recordings/moving.mp4
    python -m gauge_reader evaluate --video recordings/moving.mp4      # end-to-end check vs truth CSV

Two modes:
  stickers (default) - the scale is hand-held and moves. Two red stickers on the tube (one before 0 g, one past
                       600 g) are tracked every frame with the marker; u = marker position between them (0..1)
                       is turned into grams by a StickerCalibration.  -> StickerPipeline
  fixed              - the scale hangs still: a fixed axis from two clicks, t_px along it -> grams.  -> Pipeline

The per-frame work is `<pipeline>.process(frame, ts)`; everything with windows lives in the `_*UI` helpers and
is never touched on headless paths.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gauge_reader import overlay
from lib import camera
from lib.camera import FrameSource
from lib.geometry import Axis
from lib.units import grams_force_to_newtons

DEFAULT_CALIB = "config/scale_calib.json"   # mirrors gauge_reader.calibration.DEFAULT_CALIB_PATH
RECORDINGS_DIR = Path("recordings")
CSV_COLUMNS = ["time_s", "frame", "found", "t_px", "raw_g", "grams", "newtons", "stable", "std_g",
               "confidence", "u", "sticker_dist_px", "reason"]    # last three: sticker mode only
MODES = ("stickers", "fixed")
HISTORY = 15            # frames used for the median in tare / known-weight samples
WINDOW = "scale reader"



# --------------------------------------------------------------------------------------------- pipeline

def _f(v, nd=3) -> str:
    return "" if v is None else f"{float(v):.{nd}f}"


@dataclass
class FrameResult:
    frame: int
    time_s: float
    detection: Any          # gauge_reader.detector.Detection
    raw_g: float | None
    reading: Any            # lib.filters.FilteredReading

    @property
    def found(self) -> bool:
        return bool(self.detection.found)

    @property
    def t_px(self) -> float | None:
        return self.detection.t_px if self.detection.found else None

    @property
    def grams(self) -> float | None:
        return getattr(self.reading, "grams", None)

    @property
    def newtons(self) -> float | None:
        """gf -> N at the boundary (lib/units.py); control code should read this, not .grams."""
        g = self.grams
        return None if g is None else grams_force_to_newtons(g)

    @property
    def stable(self) -> bool:
        return bool(getattr(self.reading, "stable", False))

    @property
    def std_g(self) -> float | None:
        s = getattr(self.reading, "std_g", None)
        return None if s is None or not np.isfinite(s) else float(s)

    u = None
    sticker_dist_px = None

    @property
    def reason(self) -> str:
        return "" if self.found else "no-marker"

    def csv_row(self) -> list[str]:
        f = _f
        return [f"{self.time_s:.4f}", str(self.frame), str(int(self.found)), f(self.t_px), f(self.raw_g),
                f(self.grams), f(self.newtons), str(int(self.stable)), f(self.std_g),
                f(self.detection.confidence), f(self.u, 6), f(self.sticker_dist_px, 2), self.reason]


class Pipeline:
    """Fixed mode: detect -> calibration.to_grams -> filter.update. Keeps the last HISTORY marker positions
    (t_px) for tare / known-weight samples."""

    mode = "fixed"
    pos_name = "t"

    def __init__(self, detector, calib, filt, history: int = HISTORY):
        self.detector = detector
        self.calib = calib
        self.filt = filt
        self.recent_t: deque[float] = deque(maxlen=history)
        self.n_frames = 0
        self.n_found = 0

    @classmethod
    def from_calibration(cls, calib, det_config=None, filt=None) -> "Pipeline":
        from gauge_reader.detector import RedMarkerDetector
        from lib.filters import ReadingFilter
        return cls(RedMarkerDetector(calib.axis, det_config), calib, filt if filt is not None else ReadingFilter())

    def process(self, frame_bgr: np.ndarray, ts: float, index: int | None = None) -> FrameResult:
        det = self.detector.detect(frame_bgr)
        raw = None
        if det.found and det.t_px is not None:
            raw = float(self.calib.to_grams(det.t_px))
            self.recent_t.append(float(det.t_px))
            self.n_found += 1
        reading = self.filt.update(raw, ts)
        idx = self.n_frames if index is None else index
        self.n_frames += 1
        return FrameResult(idx, float(ts), det, raw, reading)

    def median_t(self, min_n: int = 5) -> float | None:
        if len(self.recent_t) < min_n:
            return None
        return float(statistics.median(self.recent_t))

    def tare(self) -> float | None:
        """Scale must be unloaded. Sets zero at the median marker position of the last frames."""
        t = self.median_t()
        if t is None:
            return None
        self.calib.tare(t)
        self.filt.reset()
        return t

    def add_sample(self, grams: float):
        """Known weight hanging, at rest: add (median t, grams) and refit. Returns (t, FitReport) or None."""
        t = self.median_t()
        if t is None:
            return None
        self.calib.add_sample(t, float(grams))
        report = self.calib.fit()
        self.filt.reset()
        return t, report

    median_pos = median_t

    @staticmethod
    def fmt_pos(t: float) -> str:
        return f"t={t:.2f}px"

    def describe(self) -> str:
        a = self.calib.axis
        return (f"fixed mode: axis p0={a.p0} p1={a.p1}  length {a.length:.1f}px  angle {a.angle_deg:.1f} deg  "
                f"=> {self.calib.slope_g_per_px:.3f} g/px")

    def record_crop(self, frame_bgr: np.ndarray, r: FrameResult) -> np.ndarray | None:
        """The rectified marker strip (never the full frame)."""
        return self.detector.last_strip

    def draw(self, img: np.ndarray, r: FrameResult, *, fps=None, show_mask=False, help_lines=None, extra="",
             axis=False, frame_bgr=None) -> np.ndarray:
        if axis:
            overlay.draw_axis(img, self.calib.axis, label=True)
        return overlay.draw_run_overlay(
            img, corners=self.detector.strip_corners(), point=r.detection.point, grams=r.grams, std_g=r.std_g,
            stable=r.stable, found=r.found, fps=fps, mask=self.detector.last_mask, strip=self.detector.last_strip,
            show_mask=show_mask, help_lines=help_lines, extra=extra)


# --------------------------------------------------------------------------------------------- sticker mode

@dataclass
class StickerFrameResult:
    frame: int
    time_s: float
    detection: Any          # gauge_reader.stickers.StickerDetection
    raw_g: float | None
    reading: Any            # lib.filters.FilteredReading
    last_g: float | None = None     # filtered grams of the last found frame (shown greyed while not found)

    t_px = None

    @property
    def found(self) -> bool:
        return bool(self.detection.found)

    @property
    def u(self) -> float | None:
        return float(self.detection.u) if self.detection.found and self.detection.u is not None else None

    @property
    def sticker_dist_px(self) -> float | None:
        d = getattr(self.detection, "sticker_dist_px", None)
        return None if d is None or not np.isfinite(d) else float(d)

    @property
    def reason(self) -> str:
        return "" if self.found else (getattr(self.detection, "reason", "") or "not-found")

    @property
    def grams(self) -> float | None:
        """Filtered grams, only on found frames: while a sticker is hidden the filter's extrapolation is NOT a
        reading (it can be far off during a pull), so nothing is reported - see last_g."""
        return getattr(self.reading, "grams", None) if self.found else None

    @property
    def newtons(self) -> float | None:
        g = self.grams
        return None if g is None else grams_force_to_newtons(g)

    @property
    def stable(self) -> bool:
        return self.found and bool(getattr(self.reading, "stable", False))

    std_g = FrameResult.std_g
    csv_row = FrameResult.csv_row


# Recorded crop: the S1 -> S2 segment, rectified horizontal, with 10 % of its length before / after it and
# 10 % on each side across (the tube is ~5 %): only the scale, never the patient.
CROP_SIZE = (960, 160)


def sticker_crop(frame_bgr: np.ndarray, s1, s2, size: tuple[int, int] = CROP_SIZE) -> np.ndarray | None:
    """Rectified crop around the scale between stickers s1 and s2 (fixed output size), or None without them."""
    if s1 is None or s2 is None:
        return None
    a, b = np.asarray(s1, np.float64), np.asarray(s2, np.float64)
    d = b - a
    L = float(np.hypot(*d))
    if L < 1.0:
        return None
    W, H = size
    k = W / (1.2 * L)                   # output px per image px
    nrm = np.array([-d[1], d[0]]) / L
    src = np.float32([a, b, a + nrm * (H / 2 / k)])
    x0, x1 = 0.1 * L * k, 1.1 * L * k
    dst = np.float32([[x0, H / 2], [x1, H / 2], [x0, H]])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(frame_bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


class StickerPipeline:
    """Sticker mode: tracker.detect -> calib.to_grams(u) -> filter.update (None when not found). Keeps the last
    HISTORY marker fractions u of found frames for tare / known-weight samples."""

    mode = "stickers"
    pos_name = "u"

    def __init__(self, tracker, calib, filt, history: int = HISTORY):
        self.tracker = tracker
        self.calib = calib
        self.filt = filt
        self.recent_u: deque[float] = deque(maxlen=history)
        self.n_frames = 0
        self.n_found = 0
        self.last_s1 = self.last_s2 = None      # last seen sticker positions (recording crop during dropouts)
        self.last_g: float | None = None        # filtered grams of the last found frame

    @staticmethod
    def make_tracker(calib, config=None):
        from gauge_reader.stickers import StickerConfig, StickerTracker
        if config is None:
            config = StickerConfig(u_range=tuple(calib.u_range()))
        return StickerTracker(config, s1_hint=getattr(calib, "s1_ref", None), s2_hint=getattr(calib, "s2_ref", None))

    @classmethod
    def from_calibration(cls, calib, config=None, filt=None) -> "StickerPipeline":
        from lib.filters import ReadingFilter
        return cls(cls.make_tracker(calib, config), calib, filt if filt is not None else ReadingFilter())

    def process(self, frame_bgr: np.ndarray, ts: float, index: int | None = None) -> StickerFrameResult:
        det = self.tracker.detect(frame_bgr)
        raw = None
        if det.found and det.u is not None and np.isfinite(det.u):
            raw = float(self.calib.to_grams(det.u))
            self.recent_u.append(float(det.u))
            self.n_found += 1
        if getattr(det, "s1", None) is not None and getattr(det, "s2", None) is not None:
            self.last_s1, self.last_s2 = det.s1, det.s2
        reading = self.filt.update(raw, ts)
        if raw is not None and getattr(reading, "grams", None) is not None:
            self.last_g = float(reading.grams)
        idx = self.n_frames if index is None else index
        self.n_frames += 1
        return StickerFrameResult(idx, float(ts), det, raw, reading, self.last_g)

    def median_u(self, min_n: int = 5) -> float | None:
        if len(self.recent_u) < min_n:
            return None
        return float(statistics.median(self.recent_u))

    median_pos = median_u

    def tare(self) -> float | None:
        """Scale must be unloaded. Sets zero at the median u of the last found frames."""
        u = self.median_u()
        if u is None:
            return None
        self.calib.tare(u)
        self.filt.reset()
        return u

    def add_sample(self, grams: float):
        """Known weight hanging, at rest: add (median u, grams) and refit. Returns (u, FitReport) or None."""
        u = self.median_u()
        if u is None:
            return None
        self.calib.add_sample(u, float(grams))
        report = self.calib.fit()
        self.filt.reset()
        return u, report

    @staticmethod
    def fmt_pos(u: float) -> str:
        return f"u={u:.4f}"

    def describe(self) -> str:
        c = self.calib
        return (f"stickers mode: S1={_fmt_pt(getattr(c, 's1_ref', None))} S2={_fmt_pt(getattr(c, 's2_ref', None))}  "
                f"0 g at u={c.u0:.4f}, {c.g1:.0f} g at u={c.u1:.4f}  => {c.slope_g_per_u:.1f} g per unit u")

    def record_crop(self, frame_bgr: np.ndarray, r: StickerFrameResult) -> np.ndarray | None:
        """Rectified crop around the scale (last known sticker positions); never the full frame."""
        return sticker_crop(frame_bgr, self.last_s1, self.last_s2)

    def _red_mask_crop(self, frame_bgr):
        from gauge_reader.detector import red_mask
        crop = sticker_crop(frame_bgr, self.last_s1, self.last_s2)
        if crop is None:
            return None, None
        cfg = getattr(getattr(self.tracker, "config", None), "red", None)
        return red_mask(crop, cfg), crop

    def draw(self, img: np.ndarray, r: StickerFrameResult, *, fps=None, show_mask=False, help_lines=None, extra="",
             axis=False, frame_bgr=None) -> np.ndarray:
        d = r.detection
        mask = strip = None
        if show_mask:
            mask, strip = self._red_mask_crop(frame_bgr if frame_bgr is not None else img)
        u_marks = (self.calib.u0, self.calib.u1) if axis else None
        return overlay.draw_sticker_overlay(
            img, s1=getattr(d, "s1", None), s2=getattr(d, "s2", None), marker=getattr(d, "marker", None),
            grams=r.grams if r.found else r.last_g, std_g=r.std_g, stable=r.stable, found=r.found, reason=r.reason,
            fps=fps,
            blobs=getattr(d, "blobs", None), mask=mask, strip=strip, show_mask=show_mask, help_lines=help_lines,
            extra=extra, u_marks=u_marks)


def _fmt_pt(p) -> str:
    return "--" if p is None else f"({float(p[0]):.1f}, {float(p[1]):.1f})"


def is_sticker_calib(calib) -> bool:
    return getattr(calib, "mode", "fixed") == "stickers"


def make_pipeline(calib):
    """Pipeline (fixed) or StickerPipeline, from the calibration's mode."""
    return StickerPipeline.from_calibration(calib) if is_sticker_calib(calib) else Pipeline.from_calibration(calib)


# --------------------------------------------------------------------------------------------- sources
# Cameras and video files come from the shared library (lib/camera.py), so robot scripts can use them too.

def cmd_cameras(args) -> int:
    return camera.main()


def open_source(args) -> FrameSource:
    """The camera or video named on the command line."""
    return camera.open_source(camera=args.camera, video=getattr(args, "video", None), width=args.width, height=args.height)


# --------------------------------------------------------------------------------------------- outputs

class SessionLog:
    """CSV of per-frame results, flushed every line (a crash loses nothing)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(CSV_COLUMNS)
        self._f.flush()

    def write(self, r: FrameResult):
        self._w.writerow(r.csv_row())
        self._f.flush()

    def close(self):
        self._f.close()


class StripRecorder:
    """Writes ONLY a crop around the scale (fixed mode: the rectified marker strip; sticker mode: the rectified
    S1-S2 region), upscaled, to an mp4 - never full frames, the camera also sees the patient."""

    def __init__(self, path: str | Path, fps: float, target_width: int = 1280, max_scale: float = 4.0):
        self.path = Path(path)
        self.fps, self.target_width, self.max_scale = fps, target_width, max_scale
        self._vw = None
        self._size = None
        self.n = 0

    def write(self, strip: np.ndarray | None):
        if strip is None:
            return
        if self._vw is None:
            h, w = strip.shape[:2]
            f = float(np.clip(self.target_width / max(w, 1), 1.0, self.max_scale))
            self._size = (int(w * f) // 2 * 2, max(2, int(h * f) // 2 * 2))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self._size)
        self._vw.write(cv2.resize(strip, self._size, interpolation=cv2.INTER_CUBIC))
        self.n += 1

    def close(self):
        if self._vw is not None:
            self._vw.release()


class StatusPrinter:
    """Headless: one status line about once a second."""

    def __init__(self, period_s: float = 1.0):
        self.period = period_s
        self._last = 0.0
        self._t0 = time.monotonic()
        self._n = 0

    def maybe_print(self, r: FrameResult, force: bool = False):
        self._n += 1
        now = time.monotonic()
        if not force and now - self._last < self.period:
            return
        fps = self._n / max(now - self._t0, 1e-6)
        self._last, self._t0, self._n = now, now, 0
        print(status_line(r, fps), flush=True)


def status_line(r, fps: float | None = None) -> str:
    if not r.found or r.grams is None:
        g = overlay.reason_text(r.reason) if isinstance(r, StickerFrameResult) else "no marker"
    else:
        std = f" +/-{r.std_g:.1f}" if r.std_g is not None else ""
        g = f"{r.grams:7.1f} g{std} {'STABLE' if r.stable else 'moving'}"
    raw = f"raw {r.raw_g:7.1f}" if r.raw_g is not None else "raw    --"
    if isinstance(r, StickerFrameResult):
        pos = f"u {r.u:7.4f}" if r.u is not None else "u      --"
    else:
        pos = f"t {r.t_px:7.2f}px" if r.t_px is not None else "t      --"
    conf = getattr(r.detection, "confidence", None)
    conf = f"conf {float(conf):.2f}" if conf is not None else "conf --"
    tail = f"  {fps:5.1f} fps" if fps is not None else ""
    return f"[{r.time_s:8.2f}s #{r.frame:6d}] {g:28s} {raw}  {pos}  {conf}{tail}"


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _parse_xy(s: str) -> tuple[float, float]:
    try:
        x, y = (float(v) for v in s.replace(";", ",").split(","))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected X,Y - got {s!r}") from e
    return x, y


def _load_calibration(path: str):
    """Fixed (schema 1) Calibration or StickerCalibration (schema 2), via gauge_reader.calibration.load_calibration."""
    if not Path(path).is_file():
        raise SystemExit(f"error: no calibration at {path} - run `python -m gauge_reader calibrate` first")
    import gauge_reader.calibration as vc
    loader = getattr(vc, "load_calibration", None)
    return loader(path) if loader is not None else vc.Calibration.load(path)


def _print_fit(report, unit: str = "g/px") -> None:
    """FitReport.slope_g_per_px holds the slope in g per unit of the position (px in fixed mode, u in stickers)."""
    res = ", ".join(f"{r:+.2f}" for r in (getattr(report, "residuals_g", None) or []))
    slope = getattr(report, "slope_g_per_px", None)
    if slope is None:
        slope = getattr(report, "slope_g_per_u", float("nan"))
    print(f"fit: n={report.n}  slope={slope:.4f} {unit}  offset={report.offset_g:+.2f} g  "
          f"max|res|={report.max_abs_residual_g:.2f} g  rms={report.rms_g:.2f} g")
    if res:
        print(f"     residuals [g]: {res}")


# --------------------------------------------------------------------------------------------- GUI helpers

def _open_window(w: int, h: int, name: str = WINDOW):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    f = min(1.0, 1440 / max(w, 1), 900 / max(h, 1))
    cv2.resizeWindow(name, int(w * f), int(h * f))


def _window_closed(name: str = WINDOW) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


RUN_HELP = ["q quit   z tare (unloaded)   s save calib   space pause   m mask"]


# --------------------------------------------------------------------------------------------- run

def cmd_run(args) -> int:
    calib = _load_calibration(args.calib)
    pipe = make_pipeline(calib)
    print(f"calibration {args.calib}: {pipe.describe()}")
    src = open_source(args)
    csv_path = Path(args.csv) if args.csv else RECORDINGS_DIR / f"session-{_stamp()}.csv"
    log = SessionLog(csv_path)
    rec_name = "strip" if pipe.mode == "fixed" else "crop"
    rec = StripRecorder(RECORDINGS_DIR / f"{rec_name}-{_stamp()}.mp4", src.fps) if args.record else None
    print(f"logging to {csv_path}" + (f"; recording the scale {rec_name} to {rec.path}" if rec else ""))
    status = StatusPrinter() if args.headless else None
    if not args.headless:
        _open_window(src.width or args.width, src.height or args.height)

    paused, show_mask, dirty = False, False, False
    fps_ema, t_prev = 0.0, time.monotonic()
    last = None
    frame = None
    try:
        while True:
            if not paused:
                ok, frame, ts, idx = src.read()
                if not ok:
                    break
                r = pipe.process(frame, ts, idx)
                last = r
                log.write(r)
                if rec:
                    rec.write(pipe.record_crop(frame, r))
                now = time.monotonic()
                dt, t_prev = now - t_prev, now
                if dt > 0:
                    fps_ema = 1 / dt if fps_ema == 0 else 0.9 * fps_ema + 0.1 / dt
                if status:
                    status.maybe_print(r)
                if args.max_frames and pipe.n_frames >= args.max_frames:
                    break
            if args.headless:
                continue

            disp = frame.copy()
            pipe.draw(disp, last, fps=fps_ema, show_mask=show_mask, help_lines=RUN_HELP, frame_bgr=frame,
                      extra=("PAUSED " if paused else "") + ("REC" if rec else ""))
            cv2.imshow(WINDOW, disp)
            delay = 1 if src.is_camera else max(1, int(1000 / src.fps - (time.monotonic() - t_prev) * 1000))
            key = cv2.waitKey(1 if paused else delay) & 0xFF
            if key == ord("q") or key == 27 or _window_closed():
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("m"):
                show_mask = not show_mask
            elif key == ord("z"):
                v = pipe.tare()
                if v is None:
                    print("tare: not enough recent detections - is the marker visible?")
                else:
                    dirty = True
                    print(f"tare: zero set at {pipe.fmt_pos(v)} (in memory; press s to save)")
            elif key == ord("s"):
                calib.save(args.calib)
                dirty = False
                print(f"saved calibration to {args.calib}")
    except KeyboardInterrupt:
        pass
    finally:
        log.close()
        if rec:
            rec.close()
        src.release()
        if not args.headless:
            cv2.destroyAllWindows()
    if status and last is not None:
        status.maybe_print(last, force=True)
    rate = pipe.n_found / max(pipe.n_frames, 1)
    print(f"done: {pipe.n_frames} frames, {'scale + marker' if pipe.mode == 'stickers' else 'marker'} found in "
          f"{rate:.1%}; CSV {csv_path}" + (f"; {rec_name} video {rec.path} ({rec.n} frames)" if rec else ""))
    if dirty:
        print("note: tare changed but calibration NOT saved")
    return 0


# --------------------------------------------------------------------------------------------- evaluate

def read_truth(path: str | Path) -> tuple[Axis, dict[int, float]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"error: empty truth file {path}")
    r0 = rows[0]
    axis = Axis((float(r0["p0x"]), float(r0["p0y"])), (float(r0["p1x"]), float(r0["p1y"])))
    return axis, {int(r["frame"]): float(r["grams"]) for r in rows}


WRONG_READING_G = 10.0      # sticker mode: a found frame this far off the truth is a "wrong reading"


def truth_is_stickers(path: str | Path) -> bool:
    """A sticker-mode truth CSV (gauge_reader.synthetic.moving) has u and s1x columns."""
    with open(path, newline="") as f:
        header = next(csv.reader(f), [])
    return "u" in header and "s1x" in header


def _open_eval(video, truth):
    video = Path(video)
    truth = Path(truth) if truth else video.with_suffix(".truth.csv")
    if not truth.is_file():
        raise SystemExit(f"error: truth CSV not found: {truth}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"error: cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return truth, cap, (fps if fps > 0 else 30.0)


def _stats(e):
    return (float(np.mean(e)), float(np.max(e))) if e else (float("nan"), float("nan"))


def evaluate(video: str | Path, truth: str | Path | None = None, g0: float = 0.0, g1: float = 600.0,
             max_frames: int | None = None) -> dict:
    """Run the full pipeline over a synthetic video and compare with its truth CSV. Returns metrics.
    Sticker-mode truth (u / s1x columns) is detected automatically -> evaluate_stickers."""
    truth_p = Path(truth) if truth else Path(video).with_suffix(".truth.csv")
    if truth_p.is_file() and truth_is_stickers(truth_p):
        return evaluate_stickers(video, truth_p, g0, g1, max_frames)
    from gauge_reader.calibration import Calibration
    truth, cap, fps = _open_eval(video, truth)
    axis, truth_g = read_truth(truth)
    calib = Calibration.from_clicks(axis.p0, axis.p1, g0=g0, g1=g1)
    pipe = Pipeline.from_calibration(calib)
    raw_err, filt_err, stable_err = [], [], []
    n = 0
    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = pipe.process(frame, n / fps, n)
        g_true = truth_g.get(n)
        if g_true is not None:
            if r.raw_g is not None:
                raw_err.append(abs(r.raw_g - g_true))
            if r.grams is not None:
                filt_err.append(abs(r.grams - g_true))
                if r.stable:
                    stable_err.append(abs(r.grams - g_true))
        n += 1
        if max_frames and n >= max_frames:
            break
    elapsed = time.perf_counter() - t_start
    cap.release()

    m = {"mode": "fixed", "frames": n, "found": pipe.n_found, "detection_rate": pipe.n_found / max(n, 1),
         "raw_mean_g": _stats(raw_err)[0], "raw_max_g": _stats(raw_err)[1],
         "filt_mean_g": _stats(filt_err)[0], "filt_max_g": _stats(filt_err)[1],
         "stable_mean_g": _stats(stable_err)[0], "stable_max_g": _stats(stable_err)[1], "stable_frames": len(stable_err),
         "fps": n / max(elapsed, 1e-9), "truth": str(truth)}
    m["passed"] = bool(m["detection_rate"] >= 0.95 and m["raw_mean_g"] <= 3.0)
    return m


def evaluate_stickers(video: str | Path, truth: str | Path | None = None, g0: float = 0.0, g1: float = 600.0,
                      max_frames: int | None = None) -> dict:
    """Sticker mode: calibration from the first truth row's s1, s2, p0, p1 (as if the user had clicked them),
    then the full StickerPipeline over the video. Pass = found rate >= 95 % on frames with both stickers
    visible, mean raw error <= 3 g, and no wrong reading (found but |error| > 10 g) anywhere."""
    from gauge_reader.calibration import StickerCalibration
    truth, cap, fps = _open_eval(video, truth)
    with open(truth, newline="") as f:
        rows = {int(r["frame"]): r for r in csv.DictReader(f)}
    if not rows:
        raise SystemExit(f"error: empty truth file {truth}")
    r0 = rows[min(rows)]
    xy = lambda r, k: (float(r[k + "x"]), float(r[k + "y"]))  # noqa: E731
    calib = StickerCalibration.from_clicks(xy(r0, "s1"), xy(r0, "s2"), xy(r0, "p0"), xy(r0, "p1"), g0=g0, g1=g1)
    pipe = StickerPipeline.from_calibration(calib)

    n = n_vis = n_vis_found = n_hidden = n_hidden_found = 0
    raw_err, filt_err, stable_err, wrong, wrong_hidden, stable_wrong = [], [], [], [], 0, 0
    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = pipe.process(frame, n / fps, n)
        tr = rows.get(n)
        if tr is not None:
            g_true = float(tr["grams"])
            hidden = int(float(tr.get("hidden") or 0)) != 0
            if hidden:
                n_hidden += 1
                n_hidden_found += int(r.found)
            else:
                n_vis += 1
                n_vis_found += int(r.found)
            if r.raw_g is not None:
                e = abs(r.raw_g - g_true)
                raw_err.append(e)
                if e > WRONG_READING_G:
                    wrong.append(n)
                    wrong_hidden += int(hidden)
            if r.grams is not None:     # found frames only (see StickerFrameResult.grams)
                e = abs(r.grams - g_true)
                filt_err.append(e)
                if r.stable:
                    stable_err.append(e)
                if e > WRONG_READING_G and r.stable:
                    stable_wrong += 1
        n += 1
        if max_frames and n >= max_frames:
            break
    elapsed = time.perf_counter() - t_start
    cap.release()

    m = {"mode": "stickers", "frames": n, "found": pipe.n_found, "detection_rate": pipe.n_found / max(n, 1),
         "visible_frames": n_vis, "visible_found": n_vis_found, "found_rate_visible": n_vis_found / max(n_vis, 1),
         "hidden_frames": n_hidden, "hidden_found": n_hidden_found,
         "raw_mean_g": _stats(raw_err)[0], "raw_max_g": _stats(raw_err)[1],
         "filt_mean_g": _stats(filt_err)[0], "filt_max_g": _stats(filt_err)[1],
         "stable_mean_g": _stats(stable_err)[0], "stable_max_g": _stats(stable_err)[1], "stable_frames": len(stable_err),
         "wrong_readings": len(wrong), "wrong_hidden": wrong_hidden, "wrong_frames": wrong[:20],
         "stable_wrong": stable_wrong,
         "fps": n / max(elapsed, 1e-9), "truth": str(truth)}
    m["passed"] = bool(m["found_rate_visible"] >= 0.95 and m["raw_mean_g"] <= 3.0 and m["wrong_readings"] == 0
                       and m["stable_wrong"] == 0)
    return m


def cmd_evaluate(args) -> int:
    m = evaluate(args.video, args.truth, args.g0, args.g1, args.max_frames)
    print(f"video          {args.video}  (truth {m['truth']}, {m['mode']} mode)")
    print(f"frames         {m['frames']}")
    if m["mode"] == "stickers":
        print(f"found rate     {m['found_rate_visible']:.1%} of frames with both stickers visible "
              f"({m['visible_found']}/{m['visible_frames']})")
        print(f"hidden sticker {m['hidden_frames']} frames, reading given in {m['hidden_found']}")
    else:
        print(f"detection rate {m['detection_rate']:.1%}  ({m['found']}/{m['frames']})")
    print(f"raw error      mean {m['raw_mean_g']:.2f} g   max {m['raw_max_g']:.2f} g")
    print(f"filtered error mean {m['filt_mean_g']:.2f} g   max {m['filt_max_g']:.2f} g   (lags during ramps)")
    print(f"stable frames  {m['stable_frames']}  mean {m['stable_mean_g']:.2f} g   max {m['stable_max_g']:.2f} g")
    if m["mode"] == "stickers":
        frames = f"  frames {m['wrong_frames']}" if m["wrong_readings"] else ""
        print(f"wrong readings {m['wrong_readings']} (found but > {WRONG_READING_G:.0f} g off; "
              f"{m['wrong_hidden']} on hidden-sticker frames){frames}; stable but > {WRONG_READING_G:.0f} g off: "
              f"{m['stable_wrong']}")
    print(f"processing     {m['fps']:.1f} fps")
    if m["passed"]:
        print("PASS")
    elif m["mode"] == "stickers":
        print("FAIL (need found rate >= 95% on visible frames, mean raw error <= 3 g and no wrong readings)")
    else:
        print("FAIL (need detection rate >= 95% and mean raw error <= 3 g)")
    return 0 if m["passed"] else 1


# --------------------------------------------------------------------------------------------- calibrate

AIM_HELP = ["Aim the camera square-on at the scale. Magenta = suggested red marker.",
            "SPACE freeze frame to click the axis    q quit"]
AIM_HELP_STICKERS = ["Hold the scale with both red stickers visible (not under your fingers). Magenta = red spots.",
                     "SPACE freeze frame to click the 0 g and 600 g lines    q quit"]
CLICK_HELP = ["Click 1: on the 0 g line, where the marker travels.  Click 2: on the 600 g (full-scale) line.",
              "r reset clicks    ENTER/SPACE accept (after 2 clicks)    ESC back to live"]
LINES_HELP_STICKERS = ["Click 1: on the 0 g line, where the marker travels.  Click 2: on the 600 g line.",
                       "r reset clicks    ESC back to live"]
FOUND_HELP_STICKERS = ["Stickers found (S1 / S2). Check they sit on the two red dots.",
                       "ENTER/SPACE accept    c click the stickers yourself    r redo all clicks    ESC back"]
STICKER_CLICK_HELP = ["Stickers not found automatically: click sticker 1 (0 g side), then sticker 2 (600 g side).",
                      "ENTER/SPACE accept (after 2 clicks)    r redo all clicks    ESC back"]
LIVE_HELP = ["z tare (scale unloaded)   w add known weight (type grams in terminal)   s save",
             "r redo clicks   m mask   q quit"]


class _CalibrateUI:
    def __init__(self, args, src: FrameSource):
        self.args, self.src = args, src
        self.mode = getattr(args, "mode", "fixed") or "fixed"
        self.state = "aim"
        self.frozen = None
        self.clicks: list[tuple[float, float]] = []          # 0 g line, 600 g line
        self.sticker_clicks: list[tuple[float, float]] = []  # sticker mode, manual: S1, S2
        self.found_stickers = None                           # sticker mode: (s1, s2) from find_stickers_near_line
        self.searched = False
        self.cursor = None
        self.pipe = None
        self.dirty = False
        self.quit_warned = False
        self.show_mask = False
        self.msg, self.msg_until = "", 0.0

    def flash(self, text, secs=3.0):
        print(text)
        self.msg, self.msg_until = text, time.monotonic() + secs

    # ---- clicking --------------------------------------------------------------------------------
    def reset_clicks(self):
        self.clicks, self.sticker_clicks, self.found_stickers, self.searched = [], [], None, False

    def manual_stickers(self) -> bool:
        """Sticker mode, after the line clicks, when the stickers must be clicked by hand."""
        return self.mode == "stickers" and len(self.clicks) == 2 and self.searched and self.found_stickers is None

    def on_mouse(self, event, x, y, flags, param):
        if self.state != "click":
            return
        self.cursor = (x, y)
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.clicks) < 2:
            self.clicks.append((float(x), float(y)))
            print(f"click {len(self.clicks)} ({'0 g' if len(self.clicks) == 1 else 'full scale'} line): ({x}, {y})")
        elif self.manual_stickers() and len(self.sticker_clicks) < 2:
            self.sticker_clicks.append((float(x), float(y)))
            print(f"sticker {len(self.sticker_clicks)}: ({x}, {y})")

    def search_stickers(self):
        """Sticker mode: once both lines are clicked, look for the two stickers along that line (once)."""
        if self.mode != "stickers" or len(self.clicks) < 2 or self.searched:
            return
        from gauge_reader.stickers import find_stickers_near_line
        self.searched = True
        found = find_stickers_near_line(self.frozen, self.clicks[0], self.clicks[1])
        if found is None:
            self.found_stickers = None
            self.flash("stickers not found - click sticker 1, then sticker 2", 4)
        else:
            self.found_stickers = (tuple(map(float, found[0])), tuple(map(float, found[1])))
            print(f"stickers found: S1={_fmt_pt(found[0])} S2={_fmt_pt(found[1])} - ENTER to accept")

    def stickers(self):
        if self.found_stickers is not None:
            return self.found_stickers
        if len(self.sticker_clicks) == 2:
            return tuple(self.sticker_clicks)
        return None

    def ready(self) -> bool:
        if len(self.clicks) < 2:
            return False
        if self.mode == "fixed":
            return True
        self.search_stickers()
        return self.stickers() is not None

    # ---- states ----------------------------------------------------------------------------------
    def start_live(self, calib):
        self.pipe = make_pipeline(calib)
        self.mode = self.pipe.mode
        self.state = "live"
        print(self.pipe.describe())

    def accept_clicks(self):
        p0, p1 = self.clicks
        if self.mode == "stickers":
            from gauge_reader.calibration import StickerCalibration
            s1, s2 = self.stickers()
            calib = StickerCalibration.from_clicks(s1, s2, p0, p1, g0=self.args.g0, g1=self.args.g1)
        else:
            from gauge_reader.calibration import Calibration
            calib = Calibration.from_clicks(p0, p1, g0=self.args.g0, g1=self.args.g1)
        self.start_live(calib)
        self.dirty = True

    def draw_click(self) -> np.ndarray:
        from gauge_reader.detector import find_red_blobs
        disp = self.frozen.copy()
        if self.mode == "fixed":
            overlay.draw_blobs(disp, find_red_blobs(self.frozen), max_n=1)
        overlay.draw_clicks(disp, self.clicks, self.cursor)
        if len(self.clicks) == 2:
            if self.mode == "fixed":
                overlay.draw_axis(disp, Axis(*self.clicks))
            else:
                self.search_stickers()
        help_lines = CLICK_HELP
        if self.mode == "stickers":
            st = self.stickers()
            if st is not None:
                overlay.draw_stickers(disp, st[0], st[1])
            elif self.sticker_clicks:
                overlay.draw_stickers(disp, self.sticker_clicks[0], None)
            if len(self.clicks) < 2:
                help_lines = LINES_HELP_STICKERS
            elif self.found_stickers is not None:
                help_lines = FOUND_HELP_STICKERS
            else:
                help_lines = STICKER_CLICK_HELP
        overlay.draw_loupe(disp, self.frozen, self.cursor, zoom=4)
        overlay.draw_help(disp, help_lines)
        return disp

    def loop(self) -> int:
        from gauge_reader.detector import find_red_blobs
        _open_window(self.src.width or self.args.width, self.src.height or self.args.height)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        frame = None
        while True:
            if self.state != "click":
                ok, f, ts, idx = self.src.read()
                if not ok:
                    if self.src.is_camera or frame is None:
                        print("error: no more frames")
                        break
                    # video ended: rewind so calibration can continue on a file
                    self.src.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame = f

            if self.state == "aim":
                disp = frame.copy()
                if self.mode == "stickers":
                    overlay.draw_candidates(disp, find_red_blobs(frame)[:8])     # marker + both stickers
                else:
                    overlay.draw_blobs(disp, find_red_blobs(frame))
                overlay.draw_help(disp, AIM_HELP_STICKERS if self.mode == "stickers" else AIM_HELP)
            elif self.state == "click":
                disp = self.draw_click()
            else:
                r = self.pipe.process(frame, ts, idx)
                disp = frame.copy()
                pos = r.u if self.pipe.mode == "stickers" else r.t_px
                extra = f"{self.pipe.fmt_pos(pos)} raw={r.raw_g:.1f}g" if pos is not None and r.raw_g is not None else ""
                self.pipe.draw(disp, r, fps=self.src.fps, show_mask=self.show_mask, help_lines=LIVE_HELP,
                               extra=extra, axis=True, frame_bgr=frame)
                if self.dirty:
                    overlay.put_text(disp, "unsaved", (disp.shape[1] - 140, 60), 0.6, overlay.AMBER)
            if self.msg and time.monotonic() < self.msg_until:
                overlay.draw_banner(disp, self.msg)
            cv2.imshow(WINDOW, disp)

            delay = 1 if (self.src.is_camera or self.state == "click") else max(1, int(1000 / self.src.fps))
            key = cv2.waitKey(delay) & 0xFF
            if _window_closed():
                key = ord("q")
            if key == 255:
                continue
            if key == ord("q"):
                if self.dirty and not self.quit_warned:
                    self.quit_warned = True
                    self.flash("Calibration NOT saved - press s to save, q again to quit", 5)
                    continue
                break
            self.quit_warned = False
            if self.state == "aim" and key == ord(" "):
                self.frozen, self.state = frame.copy(), "click"
                self.reset_clicks()
            elif self.state == "click":
                if key == ord("r"):
                    self.reset_clicks()
                elif key == ord("c") and self.found_stickers is not None:
                    self.found_stickers, self.sticker_clicks = None, []
                    print("click sticker 1, then sticker 2")
                elif key == 27:
                    self.state = "aim" if self.pipe is None else "live"
                elif key in (13, 10, ord(" ")) and self.ready():
                    self.accept_clicks()
            elif self.state == "live":
                self.handle_live_key(key, disp)
        cv2.destroyAllWindows()
        return 0

    def handle_live_key(self, key, disp):
        if key == ord("r"):
            self.state = "aim"
            self.reset_clicks()
        elif key == ord("m"):
            self.show_mask = not self.show_mask
        elif key == ord("z"):
            v = self.pipe.tare()
            if v is None:
                self.flash("tare failed: marker not seen in the last frames")
            else:
                self.dirty = True
                self.flash(f"tare: zero at {self.pipe.fmt_pos(v)}")
        elif key == ord("w"):
            if self.pipe.median_pos() is None:
                self.flash("cannot add weight: marker not seen in the last frames")
                return
            overlay.draw_banner(disp, "Type the known weight in the TERMINAL")
            cv2.imshow(WINDOW, disp)
            cv2.waitKey(1)
            try:
                s = input("known weight hanging on the scale, in grams (empty = cancel): ").strip()
            except EOFError:
                s = ""
            if not s:
                print("cancelled")
                return
            try:
                g = float(s.replace(",", "."))
            except ValueError:
                print(f"not a number: {s!r}")
                return
            out = self.pipe.add_sample(g)
            if out is None:
                self.flash("marker lost - sample not added")
                return
            v, report = out
            self.dirty = True
            print(f"sample: {self.pipe.fmt_pos(v)} = {g:.1f} g")
            _print_fit(report, "g/u" if self.pipe.mode == "stickers" else "g/px")
            self.flash(f"sample {g:.0f} g added - max residual {report.max_abs_residual_g:.2f} g")
        elif key == ord("s"):
            self.pipe.calib.save(self.args.calib)
            self.dirty = False
            self.flash(f"saved {self.args.calib}")


def _first_frame(src: FrameSource) -> np.ndarray:
    ok, frame, _, _ = src.read()
    if not ok:
        raise SystemExit("error: no frame to look for the stickers in")
    if not src.is_camera:
        src.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        src.index = 0
    return frame


def calibration_from_args(args, src: FrameSource | None = None):
    """--p0/--p1 (and in sticker mode --s1/--s2, or stickers searched near that line in the first frame)."""
    if args.p0 is None or args.p1 is None:
        raise SystemExit("error: give both --p0 and --p1")
    if args.mode == "fixed":
        from gauge_reader.calibration import Calibration
        return Calibration.from_clicks(args.p0, args.p1, g0=args.g0, g1=args.g1)
    from gauge_reader.calibration import StickerCalibration
    s1, s2 = args.s1, args.s2
    if (s1 is None) != (s2 is None):
        raise SystemExit("error: give both --s1 and --s2 (or neither, to search for the stickers)")
    if s1 is None:
        from gauge_reader.stickers import find_stickers_near_line
        found = find_stickers_near_line(_first_frame(src), args.p0, args.p1)
        if found is None:
            raise SystemExit("error: stickers not found near the 0 g - 600 g line; give --s1 X,Y --s2 X,Y "
                             "or calibrate by clicking (omit --p0/--p1)")
        s1, s2 = found
        print(f"stickers found: S1={_fmt_pt(s1)} S2={_fmt_pt(s2)}")
    return StickerCalibration.from_clicks(s1, s2, args.p0, args.p1, g0=args.g0, g1=args.g1)


def cmd_calibrate(args) -> int:
    src = open_source(args)
    ui = _CalibrateUI(args, src)
    try:
        if args.p0 is not None or args.p1 is not None:
            ui.start_live(calibration_from_args(args, src))
            ui.dirty = True
        elif args.resume:
            ui.start_live(_load_calibration(args.calib))    # the mode comes from the file
        return ui.loop()
    finally:
        src.release()


# --------------------------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    src = argparse.ArgumentParser(add_help=False)
    g = src.add_argument_group("source")
    g.add_argument("--camera", type=lambda s: int(s) if s.lstrip("-").isdigit() else s, default=None,
                   help="camera index or name, e.g. 1 or iphone (default: $CAMERA_INDEX or 0)")
    g.add_argument("--video", help="read frames from a video file instead of the camera")
    g.add_argument("--width", type=int, default=1920, help="requested camera width (default 1920)")
    g.add_argument("--height", type=int, default=1080, help="requested camera height (default 1080)")

    ap = argparse.ArgumentParser(prog="python -m gauge_reader",
                                 description="Read the Pesola 600 g spring scale through a camera.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", parents=[src], help="click the 0 g / 600 g points, tare, add known weights, save")
    c.add_argument("--mode", choices=MODES, default="stickers",
                   help="stickers (default): hand-held scale with two red stickers; fixed: scale clamped still")
    c.add_argument("--calib", default=DEFAULT_CALIB, help=f"calibration file (default {DEFAULT_CALIB})")
    c.add_argument("--p0", type=_parse_xy, help="X,Y of the 0 g point (skips clicking)")
    c.add_argument("--p1", type=_parse_xy, help="X,Y of the full-scale point (skips clicking)")
    c.add_argument("--s1", type=_parse_xy, help="stickers mode with --p0/--p1: X,Y of sticker 1 (default: search)")
    c.add_argument("--s2", type=_parse_xy, help="stickers mode with --p0/--p1: X,Y of sticker 2 (default: search)")
    c.add_argument("--g0", type=float, default=0.0, help="grams at p0 (default 0)")
    c.add_argument("--g1", type=float, default=600.0, help="grams at p1 (default 600)")
    c.add_argument("--resume", action="store_true",
                   help="load --calib and go straight to live tare/weights (mode from the file; --mode ignored)")
    c.set_defaults(func=cmd_calibrate)

    r = sub.add_parser("run", parents=[src], help="live reading with overlay, CSV log, optional strip recording")
    r.add_argument("--calib", default=DEFAULT_CALIB, help=f"calibration file (default {DEFAULT_CALIB})")
    r.add_argument("--csv", help="CSV output (default recordings/session-YYYYmmdd-HHMMSS.csv)")
    r.add_argument("--record", action="store_true",
                   help="record a crop around the scale only (marker strip / sticker region; never full frames)")
    r.add_argument("--headless", action="store_true", help="no windows; print a status line about once a second")
    r.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    r.set_defaults(func=cmd_run)

    k = sub.add_parser("cameras", help="list the cameras and their --camera numbers")
    k.set_defaults(func=cmd_cameras)

    e = sub.add_parser("evaluate", help="end-to-end check on a synthetic video against its truth CSV "
                                        "(sticker-mode truth detected automatically)")
    e.add_argument("--video", required=True)
    e.add_argument("--truth", help="truth CSV (default: <video>.truth.csv)")
    e.add_argument("--g0", type=float, default=0.0)
    e.add_argument("--g1", type=float, default=600.0)
    e.add_argument("--max-frames", type=int, default=0)
    e.set_defaults(func=cmd_evaluate)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
