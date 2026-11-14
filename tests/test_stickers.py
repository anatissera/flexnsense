"""Tests for vision.stickers: relative reading between two red reference stickers on a moving scale."""
from __future__ import annotations

import time

import numpy as np
import pytest

from vision.stickers import StickerConfig, StickerTracker, find_stickers_near_line
from vision.synthetic_moving import U0, U1, hand_held_poses, pull_profile, render_moving_frame

G_PER_U = 600.0 / (U1 - U0)
APP_CFG = dict(u_range=(U0 - 0.05, U1 + 0.05))   # what the app sets from calibration


def grams_err(det, truth) -> float:
    return abs(det.u - truth["u"]) * G_PER_U


def in_frame(truth, size=(1280, 720), margin=30) -> bool:
    return all(margin < p[0] < size[0] - margin and margin < p[1] < size[1] - margin for p in (truth["s1"], truth["s2"]))


def static_cases(n: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        g, ang, ppm = rng.uniform(0, 600), rng.uniform(-60, 60), rng.uniform(3.5, 7)
        c = (rng.uniform(300, 980), rng.uniform(150, 570))
        f, t = render_moving_frame(g, center=c, angle_deg=ang, px_per_mm=ppm, seed=len(out))
        if in_frame(t):
            out.append((f, t))
    return out


def test_u_of():
    assert StickerTracker.u_of((5, 0), (0, 0), (10, 0)) == pytest.approx(0.5)
    assert StickerTracker.u_of((2, 7), (0, 0), (10, 0)) == pytest.approx(0.2)
    assert StickerTracker.u_of((10, 10), (0, 0), (20, 20)) == pytest.approx(0.5)


def test_static_accuracy():
    errs = []
    for f, t in static_cases(60):
        d = StickerTracker(StickerConfig(**APP_CFG), s1_hint=t["s1"], s2_hint=t["s2"]).detect(f)
        assert d.found, d.reason
        assert d.reason == ""
        assert np.hypot(*np.subtract(d.s1, t["s1"])) < 0.5
        assert np.hypot(*np.subtract(d.s2, t["s2"])) < 0.5
        errs.append(grams_err(d, t))
    assert np.mean(errs) < 1.0 and np.max(errs) < 3.0, (np.mean(errs), np.max(errs))


def test_static_without_hints_uses_u_range():
    # low load: only one orientation puts u inside the calibrated range -> s1 is the 0 g end without hints
    f, t = render_moving_frame(20.0, angle_deg=170, seed=4)   # 0 g end is on the right
    cfg = StickerConfig(u_range=(U0 - 0.05, 0.5))
    d = StickerTracker(cfg).detect(f)
    assert d.found
    assert np.hypot(*np.subtract(d.s1, t["s1"])) < 1.0
    assert grams_err(d, t) < 1.0


def run_sequence(n=300, speed=2.0, seed=0):
    poses, loads = hand_held_poses(n, speed=speed, seed=seed), pull_profile(n, seed=seed)
    tr, errs, found, times = None, [], 0, []
    for i, (pose, g) in enumerate(zip(poses, loads)):
        f, t = render_moving_frame(float(g), seed=i, **pose)
        if tr is None:
            tr = StickerTracker(StickerConfig(**APP_CFG), s1_hint=t["s1"], s2_hint=t["s2"])
        t0 = time.perf_counter()
        d = tr.detect(f)
        times.append(time.perf_counter() - t0)
        if d.found:
            found += 1
            errs.append(grams_err(d, t))
    return found / n, np.array(errs), np.array(times)


def test_hand_held_sequence():
    rate, errs, times = run_sequence()
    assert rate >= 0.98, rate
    assert errs.max() < 4.0, errs.max()
    assert times.mean() < 0.015, times.mean()


@pytest.mark.parametrize("blur_angle", [0, 45, 90, 160])
@pytest.mark.parametrize("ppm", [3.5, 5.0])
def test_heavy_motion_blur(blur_angle, ppm):
    for g in (0.0, 300.0, 600.0):
        f, t = render_moving_frame(g, angle_deg=15, px_per_mm=ppm, blur_px=12, blur_angle_deg=blur_angle, seed=2)
        d = StickerTracker(StickerConfig(**APP_CFG), s1_hint=t["s1"], s2_hint=t["s2"]).detect(f)
        assert d.found, d.reason
        assert grams_err(d, t) < 3.0


def test_decoy_has_no_effect():
    for g, ang in ((50.0, 0.0), (310.0, 35.0), (590.0, -40.0)):
        f0, t = render_moving_frame(g, angle_deg=ang, seed=7)
        f1, _ = render_moving_frame(g, angle_deg=ang, distractor=True, seed=7)
        d0 = StickerTracker(s1_hint=t["s1"], s2_hint=t["s2"]).detect(f0)
        d1 = StickerTracker(s1_hint=t["s1"], s2_hint=t["s2"]).detect(f1)
        assert d0.found and d1.found
        assert abs(d0.u - d1.u) * G_PER_U < 0.05
        assert len(d1.blobs) == len(d0.blobs) + 1   # the decoy is seen, but not used


@pytest.mark.parametrize("hide", [1, 2])
@pytest.mark.parametrize("grams", [0.0, 150.0, 300.0, 450.0, 600.0])
@pytest.mark.parametrize("distractor", [False, True])
def test_hidden_sticker_never_reads(hide, grams, distractor):
    for ang, ppm in ((0.0, 5.0), (25.0, 3.5), (-50.0, 7.0)):
        f, t = render_moving_frame(grams, angle_deg=ang, px_per_mm=ppm, hide_sticker=hide, distractor=distractor, seed=3)
        # fresh tracker (lost) ...
        d = StickerTracker(StickerConfig(**APP_CFG), s1_hint=t["s1"], s2_hint=t["s2"]).detect(f)
        assert not d.found and d.u is None and d.reason == "sticker-hidden", (ang, d)
        # ... and a tracker that was locked on the previous frame
        tr = StickerTracker(StickerConfig(**APP_CFG), s1_hint=t["s1"], s2_hint=t["s2"])
        f_ok, _ = render_moving_frame(grams, angle_deg=ang, px_per_mm=ppm, seed=3)
        assert tr.detect(f_ok).found
        d = tr.detect(f)
        assert not d.found and d.u is None and d.reason == "sticker-hidden", (ang, d)


def test_recovers_after_hidden_stretch():
    poses = hand_held_poses(90, speed=1.5, seed=3)
    tr = None
    for i, pose in enumerate(poses):
        hidden = 1 if 30 <= i < 45 else (2 if 45 <= i < 60 else None)
        f, t = render_moving_frame(200.0 + i, hide_sticker=hidden, seed=i, **pose)
        if tr is None:
            tr = StickerTracker(StickerConfig(**APP_CFG), s1_hint=t["s1"], s2_hint=t["s2"])
        d = tr.detect(f)
        if hidden:
            assert not d.found and d.reason == "sticker-hidden"
        else:
            assert d.found, (i, d.reason)
            assert grams_err(d, t) < 3.0
            assert np.hypot(*np.subtract(d.s1, t["s1"])) < 1.0


def test_find_stickers_near_line():
    for g, ang, ppm, c in ((321.0, -33.0, 4.2, (600, 380)), (0.0, 10.0, 6.0, (640, 360)), (600.0, 55.0, 3.5, (700, 330))):
        f, t = render_moving_frame(g, angle_deg=ang, px_per_mm=ppm, center=c, distractor=True, seed=5)
        res = find_stickers_near_line(f, t["p0"], t["p1"])
        assert res is not None
        s1, s2 = res
        assert np.hypot(*np.subtract(s1, t["s1"])) < 0.3
        assert np.hypot(*np.subtract(s2, t["s2"])) < 0.3
        # clicking in the other order swaps the result
        r2 = find_stickers_near_line(f, t["p1"], t["p0"])
        assert np.allclose(r2[0], s2) and np.allclose(r2[1], s1)


def test_find_stickers_near_line_hidden_returns_none():
    f, t = render_moving_frame(100.0, angle_deg=20, hide_sticker=2, seed=6)
    assert find_stickers_near_line(f, t["p0"], t["p1"]) is None


def test_orientation_follows_rotation_with_hints():
    """Rotate the scale by 150 degrees over a sequence: s1 must stay on the 0 g side throughout."""
    n = 76
    _, t0 = render_moving_frame(120.0, angle_deg=0.0, seed=0)
    tr = StickerTracker(StickerConfig(**APP_CFG), s1_hint=t0["s1"], s2_hint=t0["s2"])
    for i in range(n):
        ang = 150.0 * i / (n - 1)
        f, t = render_moving_frame(120.0, angle_deg=ang, center=(640 + 20 * np.sin(i / 7), 360), seed=i)
        d = tr.detect(f)
        assert d.found, (i, d.reason)
        assert np.hypot(*np.subtract(d.s1, t["s1"])) < 1.0, (i, d.s1, t["s1"])
        assert grams_err(d, t) < 1.0


def test_no_scale():
    f = np.full((720, 1280, 3), 128, np.uint8)
    d = StickerTracker().detect(f)
    assert not d.found and d.reason == "no-scale"


def test_speed_lost_and_tracking():
    frames = [render_moving_frame(g, angle_deg=a, seed=i)[0] for i, (g, a) in enumerate(((100, 0), (300, 20), (500, -30)))]
    tr = StickerTracker()
    t0 = time.perf_counter()
    for _ in range(10):
        for f in frames:
            tr.reset()          # whole-frame search every time
            assert tr.detect(f).found
    assert (time.perf_counter() - t0) / 30 < 0.015
