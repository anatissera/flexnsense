"""Pure drawing helpers for the scale reader windows.

Every function draws in place on a BGR uint8 image (and returns it for chaining). No window, camera or
detector calls here, so all of it can be tested on a blank frame.
"""
from __future__ import annotations

import cv2
import numpy as np

from arm1.lib.geometry import Axis

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR
GREEN = (80, 200, 60)
AMBER = (0, 180, 255)
RED = (40, 40, 230)
CYAN = (230, 210, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
MAGENTA = (220, 60, 220)
GREY = (160, 160, 160)


def _ui_scale(img: np.ndarray) -> float:
    """Text/line scale relative to a 1280-wide frame, so overlays read the same at 720p and 1080p."""
    return max(0.5, img.shape[1] / 1280.0)


def _pt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def put_text(img, text, org, scale=0.7, color=WHITE, thickness=2, bg=True):
    """Text with a dark outline so it is readable on any background."""
    org = _pt(org)
    if bg:
        cv2.putText(img, text, org, FONT, scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)
    return img


def status_color(found: bool, stable: bool) -> tuple[int, int, int]:
    """Green = stable reading, amber = moving / settling, red = marker not found."""
    if not found:
        return RED
    return GREEN if stable else AMBER


def draw_strip_outline(img, corners, color=CYAN, thickness=None):
    """Outline of the detector's search strip; corners = (4, 2) image coordinates."""
    if corners is None:
        return img
    th = thickness or max(1, int(round(_ui_scale(img))))
    pts = np.round(np.asarray(corners, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, th, cv2.LINE_AA)
    return img


def draw_axis(img, axis: Axis, color=CYAN, label=True):
    """The calibrated axis p0 -> p1 with end ticks and 0 / full-scale labels."""
    s = _ui_scale(img)
    th = max(1, int(round(s)))
    cv2.line(img, _pt(axis.p0), _pt(axis.p1), color, th, cv2.LINE_AA)
    nrm = axis.normal
    for p in (axis.p0, axis.p1):
        a = (p[0] - 12 * s * nrm[0], p[1] - 12 * s * nrm[1])
        b = (p[0] + 12 * s * nrm[0], p[1] + 12 * s * nrm[1])
        cv2.line(img, _pt(a), _pt(b), color, th + 1, cv2.LINE_AA)
    if label:
        off = 18 * s
        put_text(img, "0", (axis.p0[0] + off * nrm[0], axis.p0[1] + off * nrm[1]), 0.6 * s, color)
        put_text(img, "FS", (axis.p1[0] + off * nrm[0], axis.p1[1] + off * nrm[1]), 0.6 * s, color)
    return img


def draw_crosshair(img, point, color=GREEN, size=None, thickness=None):
    """Crosshair with an open centre, so the marker itself stays visible."""
    if point is None:
        return img
    s = _ui_scale(img)
    size = size or int(22 * s)
    th = thickness or max(1, int(round(s)))
    x, y = float(point[0]), float(point[1])
    gap = size * 0.35
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(img, _pt((x + dx * gap, y + dy * gap)), _pt((x + dx * size, y + dy * size)), color, th, cv2.LINE_AA)
    cv2.circle(img, _pt((x, y)), max(2, int(size * 0.6)), color, th, cv2.LINE_AA)
    return img


def format_grams(grams, std_g=None) -> str:
    if grams is None:
        return "-- g"
    txt = f"{grams:6.1f} g"
    if std_g is not None and np.isfinite(std_g):
        txt += f"  +/-{std_g:.1f}"
    return txt


def draw_reading(img, grams, std_g=None, stable=False, found=True, origin=(20, 20), message=None):
    """Big weight readout in a panel at the top-left: green stable, amber moving, red 'no marker' (or `message`)."""
    s = _ui_scale(img)
    color = status_color(found and grams is not None, stable)
    big = 1.9 * s
    th = max(2, int(round(3 * s)))
    ok = found and grams is not None
    main = format_grams(grams, None) if ok else (message or "no marker")
    sub = ""
    if ok:
        sub = ("STABLE" if stable else "moving") + (f"   +/- {std_g:.1f} g" if std_g is not None and np.isfinite(std_g) else "")
    elif message and grams is not None:
        sub = f"last {grams:.1f} g"
    (tw, tht), _ = cv2.getTextSize(main, FONT, big, th)
    x0, y0 = int(origin[0]), int(origin[1])
    pad = int(12 * s)
    h = tht + pad * 3 + int(28 * s)
    w = max(tw, int(300 * s)) + pad * 2
    roi = img[y0:y0 + h, x0:x0 + w]
    if roi.size:
        roi[:] = (roi.astype(np.float32) * 0.35).astype(np.uint8)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), color, max(2, int(round(2 * s))), cv2.LINE_AA)
    put_text(img, main, (x0 + pad, y0 + pad + tht), big, color, th)
    if sub:
        # stable dot
        cy = y0 + pad * 2 + tht + int(14 * s)
        cv2.circle(img, (x0 + pad + int(8 * s), cy), int(8 * s), color, -1, cv2.LINE_AA)
        put_text(img, sub, (x0 + pad + int(24 * s), cy + int(8 * s)), 0.7 * s, color, max(1, int(round(2 * s))))
    return img


def draw_fps(img, fps, extra: str = ""):
    s = _ui_scale(img)
    txt = f"{fps:5.1f} fps" + (f"  {extra}" if extra else "")
    (tw, _), _ = cv2.getTextSize(txt, FONT, 0.6 * s, 2)
    put_text(img, txt, (img.shape[1] - tw - int(16 * s), int(30 * s)), 0.6 * s, WHITE, max(1, int(round(1.5 * s))))
    return img


def draw_help(img, lines, color=WHITE):
    """Key help / instructions at the bottom-left, one line each."""
    s = _ui_scale(img)
    lh = int(28 * s)
    y = img.shape[0] - int(14 * s) - lh * (len(lines) - 1)
    for line in lines:
        put_text(img, line, (int(16 * s), y), 0.62 * s, color, max(1, int(round(1.5 * s))))
        y += lh
    return img


def draw_banner(img, text, color=AMBER):
    """A centred warning line in the upper third (e.g. 'unsaved calibration - press q again')."""
    s = _ui_scale(img)
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.9 * s, 2)
    x = max(0, (img.shape[1] - tw) // 2)
    put_text(img, text, (x, int(img.shape[0] * 0.3)), 0.9 * s, color, max(2, int(round(2 * s))))
    return img


def _fit(src, max_w, max_h):
    h, w = src.shape[:2]
    f = min(max_w / max(w, 1), max_h / max(h, 1))
    f = max(f, 1e-3)
    return cv2.resize(src, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_NEAREST if f > 1 else cv2.INTER_AREA)


def draw_mask_inset(img, mask, strip=None, max_frac=(0.6, 0.3), margin=None):
    """Small inset at the bottom-right: the rectified strip (if given) above its red mask."""
    if mask is None:
        return img
    s = _ui_scale(img)
    margin = margin if margin is not None else int(12 * s)
    m = mask if mask.ndim == 3 else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    tiles = [strip, m] if strip is not None and strip.shape[:2] == mask.shape[:2] else [m]
    stack = np.vstack([t if t.ndim == 3 else cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) for t in tiles])
    H, W = img.shape[:2]
    inset = _fit(stack, int(W * max_frac[0]), int(H * max_frac[1]))
    ih, iw = inset.shape[:2]
    y0, x0 = H - ih - margin, W - iw - margin
    if y0 < 0 or x0 < 0:
        return img
    img[y0:y0 + ih, x0:x0 + iw] = inset
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + iw, y0 + ih), WHITE, 1)
    put_text(img, "strip / mask", (x0, y0 - 6), 0.5 * s, WHITE, 1)
    return img


def draw_blobs(img, blobs, max_n=3, color=MAGENTA):
    """Mark suggested red blobs [(x, y, area), ...] (largest first) - helps finding the marker before clicking."""
    s = _ui_scale(img)
    for i, (x, y, area) in enumerate(list(blobs)[:max_n]):
        r = int(max(10 * s, np.sqrt(area) * 1.2))
        cv2.circle(img, _pt((x, y)), r, color, max(1, int(round(2 * s))) if i == 0 else 1, cv2.LINE_AA)
        if i == 0:
            put_text(img, "marker?", (x + r + 6, y + 6), 0.6 * s, color, max(1, int(round(1.5 * s))))
    return img


def draw_clicks(img, points, cursor=None, color=CYAN):
    """Calibration clicks so far (1 = 0 g, 2 = full scale) and a rubber-band line to the cursor."""
    s = _ui_scale(img)
    labels = ("0 g", "FS")
    for i, p in enumerate(points):
        cv2.drawMarker(img, _pt(p), color, cv2.MARKER_CROSS, int(24 * s), max(1, int(round(s))), cv2.LINE_AA)
        put_text(img, labels[i] if i < 2 else str(i), (p[0] + 10 * s, p[1] - 10 * s), 0.6 * s, color, 1)
    if len(points) == 1 and cursor is not None:
        cv2.line(img, _pt(points[0]), _pt(cursor), color, 1, cv2.LINE_AA)
    return img


def draw_loupe(img, src, cursor, zoom=4, size=None):
    """Magnifier: a (size/zoom)-px patch of `src` around the cursor, scaled `zoom`x with nearest-neighbour,
    drawn in the corner of `img` farthest from the cursor, with a crosshair on the exact cursor pixel."""
    if cursor is None:
        return img
    s = _ui_scale(img)
    size = size or int(240 * s)
    half = size // (2 * zoom)
    cx, cy = int(cursor[0]), int(cursor[1])
    H, W = src.shape[:2]
    padded = cv2.copyMakeBorder(src, half, half + 1, half, half + 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    patch = padded[cy:cy + 2 * half + 1, cx:cx + 2 * half + 1]
    if patch.size == 0:
        return img
    big = cv2.resize(patch, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    bh, bw = big.shape[:2]
    m = int(16 * s)
    x0 = m if cx > W / 2 else img.shape[1] - bw - m
    y0 = m + int(110 * s) if cy > H / 2 else img.shape[0] - bh - m
    x0 = max(0, min(x0, img.shape[1] - bw))
    y0 = max(0, min(y0, img.shape[0] - bh))
    img[y0:y0 + bh, x0:x0 + bw] = big
    c = (x0 + half * zoom + zoom // 2, y0 + half * zoom + zoom // 2)
    cv2.line(img, (c[0] - bw // 2, c[1]), (c[0] - zoom, c[1]), CYAN, 1)
    cv2.line(img, (c[0] + zoom, c[1]), (c[0] + bw // 2, c[1]), CYAN, 1)
    cv2.line(img, (c[0], c[1] - bh // 2), (c[0], c[1] - zoom), CYAN, 1)
    cv2.line(img, (c[0], c[1] + zoom), (c[0], c[1] + bh // 2), CYAN, 1)
    cv2.rectangle(img, (x0, y0), (x0 + bw - 1, y0 + bh - 1), WHITE, 1)
    put_text(img, f"{zoom}x  ({cx},{cy})", (x0 + 6, y0 + bh - 8), 0.5 * s, WHITE, 1)
    return img


def draw_run_overlay(img, *, corners=None, point=None, grams=None, std_g=None, stable=False, found=False,
                     fps=None, mask=None, strip=None, show_mask=False, help_lines=None, extra=""):
    """Everything the `run` window shows, in one call."""
    color = status_color(found, stable)
    draw_strip_outline(img, corners)
    if found:
        draw_crosshair(img, point, color)
    draw_reading(img, grams, std_g, stable, found)
    if fps is not None:
        draw_fps(img, fps, extra)
    if show_mask:
        draw_mask_inset(img, mask, strip)
    if help_lines:
        draw_help(img, help_lines)
    return img


# --------------------------------------------------------------------------------------------- sticker mode

REASON_TEXT = {
    "no-scale": "scale not found",
    "sticker-hidden": "sticker hidden",
    "no-marker": "marker not found",
    "ambiguous": "too many red spots",
}


def reason_text(reason: str | None) -> str:
    """Short user message for a StickerDetection.reason ('' / None -> 'not found')."""
    if not reason:
        return "not found"
    return REASON_TEXT.get(reason, reason.replace("-", " "))


def draw_stickers(img, s1=None, s2=None, marker=None, color=CYAN, marker_color=GREEN, u_marks=None):
    """Sticker 1 and 2 (circled, labelled S1 / S2), the line between them, the marker crosshair and, optionally,
    ticks at fractions `u_marks` (e.g. the 0 g and 600 g positions u0, u1) along the S1 -> S2 line."""
    s = _ui_scale(img)
    th = max(1, int(round(s)))
    r = int(14 * s)
    if s1 is not None and s2 is not None:
        cv2.line(img, _pt(s1), _pt(s2), color, th, cv2.LINE_AA)
        d = np.array(s2, float) - np.array(s1, float)
        L = float(np.hypot(*d))
        if u_marks and L > 0:
            nrm = np.array([-d[1], d[0]]) / L
            for u in u_marks:
                c = np.array(s1, float) + float(u) * d
                cv2.line(img, _pt(c - 10 * s * nrm), _pt(c + 10 * s * nrm), color, th + 1, cv2.LINE_AA)
    for label, p in (("S1", s1), ("S2", s2)):
        if p is None:
            continue
        cv2.circle(img, _pt(p), r, color, th + 1, cv2.LINE_AA)
        put_text(img, label, (p[0] + r + 4, p[1] - r), 0.6 * s, color, max(1, int(round(1.5 * s))))
    if marker is not None:
        draw_crosshair(img, marker, marker_color)
    return img


def draw_candidates(img, blobs, color=MAGENTA):
    """All red candidates [(x, y, area), ...] as thin circles (debug view of the sticker tracker)."""
    s = _ui_scale(img)
    for x, y, area in list(blobs or []):
        r = int(max(6 * s, np.sqrt(max(area, 1)) * 1.2))
        cv2.circle(img, _pt((x, y)), r, color, 1, cv2.LINE_AA)
    return img


def draw_sticker_overlay(img, *, s1=None, s2=None, marker=None, grams=None, std_g=None, stable=False, found=False,
                         reason="", fps=None, blobs=None, mask=None, strip=None, show_mask=False, help_lines=None,
                         extra="", u_marks=None):
    """Everything the sticker-mode window shows, in one call. When not found, the reading panel says why."""
    color = status_color(found, stable)
    draw_stickers(img, s1, s2, marker if found else None, CYAN if found else GREY, color, u_marks=u_marks)
    draw_reading(img, grams, std_g, stable, found, message=None if found else reason_text(reason))
    if fps is not None:
        draw_fps(img, fps, extra)
    if show_mask:
        draw_candidates(img, blobs)
        draw_mask_inset(img, mask, strip)
    if help_lines:
        draw_help(img, help_lines)
    return img
