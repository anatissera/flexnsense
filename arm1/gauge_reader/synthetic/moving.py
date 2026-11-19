"""Synthetic frames of a MOVING scale with two red reference stickers, for the sticker-based (relative) reader.

    frame, truth = render_moving_frame(312.0, center=(900, 500), angle_deg=20, px_per_mm=5)
    python -m gauge_reader.synthetic.moving recordings/moving.mp4     # hand-held style video + truth CSV

The scale is drawn like gauge_reader.synthetic.still (0..600 g over 100 mm, 12 mm tube, 4x supersampled, exact sub-pixel
truth) plus two red round stickers on the tube's centre line, one before the 0 g line and one past the 600 g
line. Stickers and marker are the SAME red, as on the real scale; they are told apart by position.

Axis convention (lib.geometry): t = 0 at the 0 g line, increasing towards the 600 g line.
u = marker position between the stickers as a fraction: 0 at sticker 1 (0 g side), 1 at sticker 2.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from arm1.lib.geometry import Axis
from arm1.gauge_reader.synthetic.still import CAPACITY_G, SCALE_MM, SHIFT, SS, TUBE_DIAM_MM, _poly, grams_to_t

STICKER1_MM = -12.0       # sticker centres along the axis, mm from the 0 g line (gap ~1 cm to the lines)
STICKER2_MM = 112.0
STICKER_DIAM_MM = 5.0
RED = (35, 30, 200)       # BGR, same red for marker and stickers
SKIN = (120, 150, 205)    # BGR, a hand


def u_of_grams(grams: float) -> float:
    """Marker position as a fraction between the stickers for a load (datasheet geometry)."""
    t_mm = grams / CAPACITY_G * SCALE_MM
    return (t_mm - STICKER1_MM) / (STICKER2_MM - STICKER1_MM)


U0 = u_of_grams(0.0)        # where the 0 g line sits between the stickers
U1 = u_of_grams(CAPACITY_G)  # where the 600 g line sits


def _fx(p) -> tuple[int, int]:
    """Image point -> supersampled fixed-point centre (same half-pixel correction as _poly)."""
    return tuple(int(round((v * SS + (SS - 1) / 2) * (1 << SHIFT))) for v in p)


def _motion_kernel(length_px: float, angle_deg: float) -> np.ndarray | None:
    if length_px < 1.0:
        return None
    n = int(np.ceil(length_px)) | 1
    k = np.zeros((n, n), np.float32)
    c = n // 2
    a = np.radians(angle_deg)
    for s in np.linspace(-length_px / 2, length_px / 2, 4 * n):
        x, y = c + s * np.cos(a), c + s * np.sin(a)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            if 0 <= x0 + dx < n and 0 <= y0 + dy < n:
                k[y0 + dy, x0 + dx] += w
    return k / k.sum()


def render_moving_frame(
    grams: float,
    *,
    size: tuple[int, int] = (1280, 720),
    center: tuple[float, float] = (640.0, 360.0),
    angle_deg: float = 0.0,
    px_per_mm: float = 5.0,
    blur_px: float = 0.0,
    blur_angle_deg: float = 0.0,
    hands: bool = True,
    hide_sticker: int | None = None,
    distractor: bool = False,
    noise_sigma: float = 2.0,
    brightness: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """One frame. center = image position of the middle of the 0..600 g scale; angle_deg = direction 0 g -> 600 g.

    blur_px: length of linear motion blur; hide_sticker: 1 or 2 to cover that sticker (e.g. by a thumb).
    Returns (frame_bgr, truth) with truth keys: grams, u, u0, u1, s1, s2, marker, p0, p1, axis, px_per_mm.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    a = np.radians(angle_deg)
    d = np.array([np.cos(a), np.sin(a)])
    half_len = SCALE_MM / 2 * px_per_mm
    p0 = tuple(np.array(center) - d * half_len)
    p1 = tuple(np.array(center) + d * half_len)
    axis = Axis(p0, p1)

    W, H = w * SS, h * SS
    grad = np.linspace(150, 105, H, dtype=np.float32)[:, None, None]
    img = np.broadcast_to(grad, (H, W, 3)).astype(np.float32).copy()
    img[..., 0] += 10
    img = img.astype(np.uint8)
    if distractor:  # a red object away from the scale
        cv2.circle(img, _fx((w * 0.12, h * 0.8)), 28 * SS * (1 << SHIFT), RED, -1, cv2.LINE_AA, SHIFT)

    half = TUBE_DIAM_MM / 2 * px_per_mm
    cv2.fillPoly(img, [_poly(axis, -30 * px_per_mm, (SCALE_MM + 30) * px_per_mm, -half, half)], (190, 188, 185), cv2.LINE_AA, SHIFT)
    cv2.fillPoly(img, [_poly(axis, -30 * px_per_mm, (SCALE_MM + 30) * px_per_mm, -half * 0.55, -half * 0.25)], (225, 224, 222), cv2.LINE_AA, SHIFT)
    for g in range(0, int(CAPACITY_G) + 1, 5):
        t = grams_to_t(g, px_per_mm)
        n_end = half * (0.95 if g % 50 == 0 else 0.6)
        th = 0.18 * px_per_mm
        cv2.fillPoly(img, [_poly(axis, t - th / 2, t + th / 2, half * 0.25, n_end)], (60, 60, 70), cv2.LINE_AA, SHIFT)

    t_m = grams_to_t(grams, px_per_mm)
    ml = 1.5 * px_per_mm / 2
    cv2.fillPoly(img, [_poly(axis, t_m - ml, t_m + ml, -half * 0.8, half * 0.8)], RED, cv2.LINE_AA, SHIFT)

    s1 = axis.point_at(STICKER1_MM * px_per_mm)
    s2 = axis.point_at(STICKER2_MM * px_per_mm)
    r_fx = int(round(STICKER_DIAM_MM / 2 * px_per_mm * SS * (1 << SHIFT)))
    for k, s in ((1, s1), (2, s2)):
        cv2.circle(img, _fx(s), r_fx, RED, -1, cv2.LINE_AA, SHIFT)

    if hands:  # fingers holding both ends, beyond the stickers
        for t_mm in (-27.0, SCALE_MM + 27.0):
            c = axis.point_at(t_mm * px_per_mm)
            ax_len = int(round(9 * px_per_mm * SS))
            cv2.ellipse(img, _fx(c), (ax_len * (1 << SHIFT), int(ax_len * 1.6) * (1 << SHIFT)), angle_deg, 0, 360,
                        SKIN, -1, cv2.LINE_AA, SHIFT)
    if hide_sticker in (1, 2):  # a thumb over one sticker
        c = s1 if hide_sticker == 1 else s2
        rr = int(round(5 * px_per_mm * SS))
        cv2.ellipse(img, _fx(c), (rr * (1 << SHIFT), int(rr * 1.4) * (1 << SHIFT)), angle_deg + 90, 0, 360,
                    SKIN, -1, cv2.LINE_AA, SHIFT)

    frame = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    frame = cv2.GaussianBlur(frame, (0, 0), 0.7)
    k = _motion_kernel(blur_px, blur_angle_deg)
    if k is not None:
        frame = cv2.filter2D(frame, -1, k, borderType=cv2.BORDER_REFLECT)
    frame = np.clip(frame * brightness + rng.normal(0, noise_sigma, frame.shape), 0, 255).astype(np.uint8)

    u = (t_m / px_per_mm - STICKER1_MM) / (STICKER2_MM - STICKER1_MM)
    truth = {"grams": float(grams), "u": float(u), "u0": U0, "u1": U1, "s1": s1, "s2": s2,
             "marker": axis.point_at(t_m), "p0": p0, "p1": p1, "axis": axis, "px_per_mm": px_per_mm}
    return frame, truth


def hand_held_poses(n: int, size=(1280, 720), seed: int = 0, speed: float = 1.0) -> list[dict]:
    """A smooth random walk of position, angle and distance, like a scale held in the hands."""
    rng = np.random.default_rng(seed)
    w, h = size
    k = np.arange(n) / 30.0
    def wander(amp, f1, f2, ph):
        return amp * (0.6 * np.sin(2 * np.pi * f1 * speed * k + ph) + 0.4 * np.sin(2 * np.pi * f2 * speed * k + 2 * ph))
    cx = w / 2 + wander(w * 0.12, 0.23, 0.61, rng.uniform(0, 6))
    cy = h / 2 + wander(h * 0.12, 0.31, 0.53, rng.uniform(0, 6))
    ang = rng.uniform(-10, 10) + wander(18, 0.17, 0.41, rng.uniform(0, 6))
    ppm = 5.0 + wander(1.2, 0.11, 0.29, rng.uniform(0, 6))
    poses = []
    for i in range(n):
        j = max(i - 1, 0)
        vx, vy = cx[i] - cx[j], cy[i] - cy[j]
        poses.append({"center": (float(cx[i]), float(cy[i])), "angle_deg": float(ang[i]), "px_per_mm": float(ppm[i]),
                      "blur_px": float(np.hypot(vx, vy) * 0.5), "blur_angle_deg": float(np.degrees(np.arctan2(vy, vx)))})
    return poses


def pull_profile(n: int, max_g: float = 450.0, seed: int = 0) -> np.ndarray:
    """Rest, a few pulls of different strength, rest: like a rehab exercise."""
    rng = np.random.default_rng(seed)
    g = np.zeros(n)
    k = n // 8
    for c, peak in zip(range(k, n - k, max(1, (n - 2 * k) // 3)), rng.uniform(0.4, 1.0, 3) * max_g):
        width = k * 0.9
        g += peak * np.exp(-0.5 * ((np.arange(n) - c - width / 2) / (width / 2.5)) ** 2)
    return np.clip(g, 0, CAPACITY_G)


def make_moving_video(path: str | Path, n_frames: int = 300, fps: float = 30.0, seed: int = 0, speed: float = 1.0,
                      size=(1280, 720), occlude_every: int = 0) -> Path:
    """Write a hand-held style mp4 and <name>.truth.csv (frame, grams, u, u0, u1, s1, s2, marker, p0, p1, hidden)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    poses, loads = hand_held_poses(n_frames, size, seed, speed), pull_profile(n_frames, seed=seed)
    with open(path.with_suffix(".truth.csv"), "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["frame", "grams", "u", "u0", "u1", "s1x", "s1y", "s2x", "s2y", "mx", "my", "p0x", "p0y", "p1x", "p1y", "hidden"])
        for i, (pose, g) in enumerate(zip(poses, loads)):
            hidden = 1 + (i // occlude_every) % 2 if occlude_every and (i // occlude_every) % 3 == 2 else None
            frame, tr = render_moving_frame(float(g), size=size, seed=i, hide_sticker=hidden, **pose)
            vw.write(frame)
            pts = [*tr["s1"], *tr["s2"], *tr["marker"], *tr["p0"], *tr["p1"]]
            out.writerow([i, f"{g:.3f}", f"{tr['u']:.6f}", f"{U0:.6f}", f"{U1:.6f}", *[f"{v:.3f}" for v in pts], hidden or 0])
    vw.release()
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a synthetic hand-held scale video (red stickers) with truth.")
    ap.add_argument("out", nargs="?", default="recordings/moving.mp4")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--speed", type=float, default=1.0, help="how fast the hands move (1 = calm, 3 = shaky)")
    ap.add_argument("--occlude-every", type=int, default=0, help="hide a sticker in some blocks of N frames")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p = make_moving_video(a.out, a.frames, seed=a.seed, speed=a.speed, occlude_every=a.occlude_every)
    print(f"wrote {p} and {p.with_suffix('.truth.csv')}")


if __name__ == "__main__":
    main()
