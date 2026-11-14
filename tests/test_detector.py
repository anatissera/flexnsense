"""Tests for vision.detector on synthetic frames (no camera).

Note on ground truth: vision.synthetic draws at supersampled coordinate SS*x and downsamples with INTER_AREA,
so output pixel j's centre is at supersampled (SS*j + (SS-1)/2), i.e. every rendered feature sits
(SS-1)/(2*SS) = 0.375 px up-left of `truth` in OpenCV pixel-centre coordinates. `RENDER_OFFSET` measures that
offset once (detector-independent, on an axis-aligned frame) and snaps it to {0, (SS-1)/(2*SS)}, so these
tests stay correct whether or not the renderer gets fixed.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from vision import synthetic
from vision.detector import DetectorConfig, Detection, RedMarkerDetector, find_red_blobs, red_mask
from vision.geometry import Axis
from vision.synthetic import grams_to_t, render_scale_frame


def _measure_render_offset() -> float:
    """Marker centre offset of the renderer along +x on a horizontal, noise-free frame (plain column centroid)."""
    p0 = (200.0, 360.0)
    frame, truth = render_scale_frame(300.0, p0=p0, angle_deg=0.0, px_per_mm=6.0, noise_sigma=0.0, seed=0)
    f = frame.astype(np.float64)
    rows = slice(int(p0[1]) - 20, int(p0[1]) + 21)
    red = np.clip(f[rows, :, 2] - np.maximum(f[rows, :, 0], f[rows, :, 1]) - 20, 0, None)
    x = (red.sum(axis=0) * np.arange(f.shape[1])).sum() / red.sum()
    measured = x - truth["marker_xy"][0]
    ss = getattr(synthetic, "SS", 4)
    cands = [0.0, -(ss - 1) / (2 * ss)]
    return min(cands, key=lambda c: abs(c - measured))


RENDER_OFFSET = _measure_render_offset()


def rendered_t(truth: dict) -> float:
    """Where the marker actually is in the frame, along the axis (pixel-centre convention)."""
    ax: Axis = truth["axis"]
    mx, my = truth["marker_xy"]
    return ax.project((mx + RENDER_OFFSET, my + RENDER_OFFSET))[0]


def frame_with_marker_in_view(grams, angle_deg, px_per_mm, seed, marker_y, **kw):
    """Place p0 so the marker lands at image y=marker_y, x=640 (the full 100 mm axis may not fit in 720 px)."""
    t = grams_to_t(grams, px_per_mm)
    a = np.radians(angle_deg)
    p0 = (640.0 - t * np.cos(a), marker_y - t * np.sin(a))
    return render_scale_frame(grams, p0=p0, angle_deg=angle_deg, px_per_mm=px_per_mm, seed=seed, **kw)


def random_cases(n, seed=123):
    rng = np.random.default_rng(seed)
    for i in range(n):
        yield dict(grams=float(rng.uniform(0, 600)), angle_deg=float(rng.uniform(60, 120)),
                   px_per_mm=float(rng.uniform(4, 9)), seed=1000 + i, marker_y=float(rng.uniform(100, 620)))


def test_render_offset_is_known():
    ss = getattr(synthetic, "SS", 4)
    assert RENDER_OFFSET in (0.0, -(ss - 1) / (2 * ss))


def test_ac1_subpixel_accuracy_random_cases():
    errs = []
    for c in random_cases(80):
        frame, truth = frame_with_marker_in_view(**c)
        d = RedMarkerDetector(truth["axis"]).detect(frame)
        assert d.found, c
        errs.append(abs(d.t_px - rendered_t(truth)))
    errs = np.array(errs)
    print(f"\nAC1: n={len(errs)} max={errs.max():.3f} px mean={errs.mean():.3f} px (render offset {RENDER_OFFSET})")
    assert errs.max() < 0.2, errs.max()
    assert errs.mean() < 0.1


def test_default_geometry_point_and_n():
    frame, truth = render_scale_frame(250.0, seed=7)
    d = RedMarkerDetector(truth["axis"]).detect(frame)
    assert d.found and 0.5 < d.confidence <= 1.0
    assert abs(d.t_px - rendered_t(truth)) < 0.2
    assert abs(d.n_px) < 1.0
    ex = (truth["marker_xy"][0] + RENDER_OFFSET, truth["marker_xy"][1] + RENDER_OFFSET)
    assert np.hypot(d.point[0] - ex[0], d.point[1] - ex[1]) < 1.0


@pytest.mark.parametrize("grams,angle", [(0, 90), (120, 75), (300, 90), (450, 105), (520, 90)])
def test_distractor_does_not_change_result(grams, angle):
    kw = dict(angle_deg=angle, px_per_mm=6.0, seed=5)
    f0, truth = render_scale_frame(grams, **kw)
    f1, _ = render_scale_frame(grams, distractor=True, **kw)
    det = RedMarkerDetector(truth["axis"])
    d0, d1 = det.detect(f0), det.detect(f1)
    assert d0.found and d1.found
    assert abs(d1.t_px - d0.t_px) < 1e-6
    assert abs(d1.t_px - rendered_t(truth)) < 0.2
    # the distractor really is in the frame and red
    blobs = find_red_blobs(f1)
    assert len(blobs) >= 2


def test_not_found_when_marker_outside_strip():
    frame, truth = render_scale_frame(300.0, seed=3)
    ax = truth["axis"]
    shift = 200 * ax.normal
    moved = Axis(tuple(np.array(ax.p0) + shift), tuple(np.array(ax.p1) + shift))
    d = RedMarkerDetector(moved).detect(frame)
    assert isinstance(d, Detection)
    assert not d.found and d.t_px is None and d.point is None and d.confidence == 0.0


@pytest.mark.parametrize("brightness", [0.8, 1.15])
def test_brightness_change(brightness):
    errs = []
    for c in random_cases(15, seed=77):
        frame, truth = frame_with_marker_in_view(**c, brightness=brightness)
        d = RedMarkerDetector(truth["axis"]).detect(frame)
        assert d.found
        errs.append(abs(d.t_px - rendered_t(truth)))
    assert max(errs) < 0.3, max(errs)


def test_red_mask_and_blobs_sanity():
    img = np.full((100, 200, 3), (180, 180, 180), np.uint8)
    img[40:60, 20:50] = (35, 30, 200)     # red
    img[40:60, 120:150] = (200, 60, 40)   # blue
    m = red_mask(img)
    assert m.shape == img.shape[:2] and m.dtype == np.uint8
    assert set(np.unique(m)) <= {0, 255}
    assert m[50, 35] == 255 and m[50, 135] == 0 and m[5, 5] == 0
    blobs = find_red_blobs(img)
    assert len(blobs) == 1
    x, y, area = blobs[0]
    assert abs(x - 34.5) < 0.6 and abs(y - 49.5) < 0.6 and 500 <= area <= 650
    img[70:95, 160:195] = (35, 30, 200)
    blobs = find_red_blobs(img)
    assert len(blobs) == 2 and blobs[0][2] >= blobs[1][2]
    assert find_red_blobs(img, min_area=10_000) == []


def test_strip_corners_and_last_buffers():
    frame, truth = render_scale_frame(100.0, angle_deg=80.0, seed=1)
    cfg = DetectorConfig()
    det = RedMarkerDetector(truth["axis"], cfg)
    corners = det.strip_corners()
    assert corners.shape == (4, 2)
    ax = truth["axis"]
    t, n = ax.project(tuple(corners[0]))
    assert abs(t + cfg.margin_px) < 1e-6 and abs(n + cfg.strip_half_width_px) < 1e-6
    assert det.last_strip is None and det.last_mask is None
    d = det.detect(frame)
    assert d.found
    assert det.last_strip.shape[:2] == det.last_mask.shape
    assert det.last_strip.shape[0] >= 2 * cfg.strip_half_width_px
    assert det.last_strip.shape[1] >= ax.length + 2 * cfg.margin_px
    # strip convention: x = t + margin, y = n + half_width lands on the mask
    xs, ys = int(round(d.t_px + cfg.margin_px)), int(round(d.n_px + cfg.strip_half_width_px))
    assert det.last_mask[ys, xs] == 255


def test_detect_timing_1280x720():
    frame, truth = render_scale_frame(300.0, seed=2)
    assert frame.shape[:2] == (720, 1280)
    det = RedMarkerDetector(truth["axis"])
    det.detect(frame)
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        det.detect(frame)
    ms = (time.perf_counter() - t0) / n * 1e3
    print(f"\nmean detect(): {ms:.2f} ms")
    assert ms < 10.0
