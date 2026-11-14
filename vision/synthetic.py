"""Synthetic frames of the Pesola 600 g scale with a red marker at a known position, for tests and offline tuning.

    frame, truth = render_scale_frame(312.0)              # one BGR frame + ground truth
    python -m vision.synthetic recordings/synthetic.mp4   # a sweep video + truth CSV beside it

Geometry follows the datasheet: the 0..600 g scale is 100 mm long (6 g per mm), the tube is 12 mm wide.
Drawing is done 4x supersampled with sub-pixel polygon fills, then downscaled, so marker positions are
exact to well under 0.1 px and sub-pixel detector accuracy can be tested honestly.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from vision.geometry import Axis

SCALE_MM = 100.0          # 0 g line to 600 g line
CAPACITY_G = 600.0
TUBE_DIAM_MM = 12.0
SS = 4                    # supersampling factor
SHIFT = 4                 # fixed-point bits for cv2.fillPoly / cv2.line


def _poly(axis: Axis, t0: float, t1: float, n0: float, n1: float) -> np.ndarray:
    pts = [axis.point_at(t0, n0), axis.point_at(t1, n0), axis.point_at(t1, n1), axis.point_at(t0, n1)]
    # INTER_AREA downscaling maps supersampled pixel centre X to (X - (SS-1)/2) / SS, so add (SS-1)/2 here
    return np.round((np.array(pts) * SS + (SS - 1) / 2) * (1 << SHIFT)).astype(np.int32).reshape(-1, 1, 2)


def grams_to_t(grams: float, px_per_mm: float) -> float:
    """Marker distance along the axis (px) for a load, using the datasheet's 6 g per mm."""
    return grams / CAPACITY_G * SCALE_MM * px_per_mm


def render_scale_frame(
    grams: float,
    *,
    size: tuple[int, int] = (1280, 720),
    p0: tuple[float, float] = (640.0, 60.0),
    angle_deg: float = 90.0,
    px_per_mm: float = 6.0,
    marker_len_mm: float = 1.5,
    noise_sigma: float = 2.0,
    blur_sigma: float = 0.7,
    brightness: float = 1.0,
    distractor: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Render one frame. angle_deg is the direction p0 -> p1 in image coordinates (90 = scale hangs downwards).

    Returns (frame_bgr uint8 HxWx3, truth) where truth = {axis, marker_t_px, marker_xy, grams, px_per_mm}.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    a = np.radians(angle_deg)
    length = SCALE_MM * px_per_mm
    axis = Axis(p0, (p0[0] + length * np.cos(a), p0[1] + length * np.sin(a)))

    # background: soft vertical gradient (a wall), supersampled canvas
    W, H = w * SS, h * SS
    grad = np.linspace(150, 110, H, dtype=np.float32)[:, None, None]
    img = np.broadcast_to(grad, (H, W, 3)).astype(np.float32).copy()
    img[..., 0] += 8  # slightly blue-grey wall
    img = img.astype(np.uint8)

    half = TUBE_DIAM_MM / 2 * px_per_mm
    # aluminium tube from 25 mm above zero to 30 mm past full scale, with a lighter highlight stripe
    cv2.fillPoly(img, [_poly(axis, -25 * px_per_mm, (SCALE_MM + 30) * px_per_mm, -half, half)], (190, 188, 185), cv2.LINE_AA, SHIFT)
    cv2.fillPoly(img, [_poly(axis, -25 * px_per_mm, (SCALE_MM + 30) * px_per_mm, -half * 0.55, -half * 0.25)], (225, 224, 222), cv2.LINE_AA, SHIFT)

    # tick marks every 5 g; longer every 50 g; on one side of the tube so the marker stays visible
    for g in range(0, int(CAPACITY_G) + 1, 5):
        t = grams_to_t(g, px_per_mm)
        n_end = half * (0.95 if g % 50 == 0 else 0.6)
        thick = 0.18 * px_per_mm
        cv2.fillPoly(img, [_poly(axis, t - thick / 2, t + thick / 2, half * 0.25, n_end)], (60, 60, 70), cv2.LINE_AA, SHIFT)

    # the red marker: a short band across the tube centred at the true position
    t_m = grams_to_t(grams, px_per_mm)
    ml = marker_len_mm * px_per_mm / 2
    cv2.fillPoly(img, [_poly(axis, t_m - ml, t_m + ml, -half * 0.8, half * 0.8)], (35, 30, 200), cv2.LINE_AA, SHIFT)

    if distractor:  # a red object elsewhere in the frame (e.g. a shirt), away from the tube
        cx, cy = int(w * 0.15) * SS + (SS - 1) // 2, int(h * 0.7) * SS + (SS - 1) // 2
        cv2.circle(img, (cx, cy), 30 * SS, (30, 25, 190), -1, cv2.LINE_AA)

    frame = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    if blur_sigma > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), blur_sigma)
    frame = frame * brightness + rng.normal(0, noise_sigma, frame.shape)
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    truth = {"axis": axis, "marker_t_px": float(t_m), "marker_xy": axis.point_at(t_m), "grams": float(grams), "px_per_mm": px_per_mm}
    return frame, truth


def sweep(n_frames: int = 300, max_g: float = 500.0, wobble_g: float = 2.0, seed: int = 0) -> np.ndarray:
    """A load profile: rest at 0, ramp up, hold, ramp down, with a small spring wobble."""
    rng = np.random.default_rng(seed)
    k = np.arange(n_frames)
    q = n_frames // 6
    base = np.interp(k, [0, q, 2 * q, 4 * q, 5 * q, n_frames - 1], [0, 0, max_g, max_g, 0, 0])
    return np.clip(base + wobble_g * np.sin(k / 3.0) * (base > 0) + rng.normal(0, 0.3, n_frames) * (base > 0), 0, 600)


def make_video(path: str | Path, grams_seq, fps: float = 30.0, **render_kw) -> Path:
    """Write an mp4 of the given load sequence and a truth CSV (frame, grams, marker_t_px) beside it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = render_kw.get("size", (1280, 720))
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    with open(path.with_suffix(".truth.csv"), "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["frame", "grams", "marker_t_px", "p0x", "p0y", "p1x", "p1y"])
        for i, g in enumerate(grams_seq):
            frame, tr = render_scale_frame(float(g), seed=i, **render_kw)
            vw.write(frame)
            ax = tr["axis"]
            out.writerow([i, f"{g:.3f}", f"{tr['marker_t_px']:.3f}", *[f"{v:.3f}" for v in (*ax.p0, *ax.p1)]])
    vw.release()
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a synthetic scale video with ground truth.")
    ap.add_argument("out", nargs="?", default="recordings/synthetic.mp4")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--angle", type=float, default=90.0, help="direction 0 g -> 600 g in degrees (90 = down)")
    ap.add_argument("--px-per-mm", type=float, default=6.0)
    ap.add_argument("--distractor", action="store_true")
    a = ap.parse_args()
    p = make_video(a.out, sweep(a.frames), angle_deg=a.angle, px_per_mm=a.px_per_mm, distractor=a.distractor)
    print(f"wrote {p} and {p.with_suffix('.truth.csv')}")


if __name__ == "__main__":
    main()
