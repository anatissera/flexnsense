import json

import numpy as np
import pytest

from vision.calibration import (
    Calibration,
    FitReport,
    MIN_FIT_SPREAD_U,
    StickerCalibration,
    load_calibration,
    project_u,
)
from vision.synthetic_moving import U0, U1, render_moving_frame, u_of_grams


def _truth(grams=0.0, **pose):
    _, tr = render_moving_frame(grams, size=(320, 240), hands=False, noise_sigma=0.0, seed=0, **pose)
    return tr


def _cal_from_truth(tr, **kw):
    return StickerCalibration.from_clicks(tr["s1"], tr["s2"], tr["p0"], tr["p1"], **kw)


def _u_in_frame(tr):
    """u of the marker measured in the image (what the tracker reports), not the closed-form truth."""
    return project_u(tr["s1"], tr["s2"], tr["marker"])


# ---- construction -------------------------------------------------------------------------------

def test_from_clicks_recovers_u0_u1_from_synthetic_geometry():
    tr = _truth(center=(160, 120), angle_deg=20.0, px_per_mm=1.5)
    cal = _cal_from_truth(tr)
    assert cal.u0 == pytest.approx(U0, abs=1e-9)
    assert cal.u1 == pytest.approx(U1, abs=1e-9)
    assert cal.slope_g_per_u == pytest.approx(600.0 / (U1 - U0))
    assert cal.to_grams(cal.u0) == pytest.approx(0.0, abs=1e-9)
    assert cal.to_grams(cal.u1) == pytest.approx(600.0, abs=1e-9)
    assert cal.offset_g == pytest.approx(cal.to_grams(0.0))
    assert cal.s1_ref == pytest.approx(tr["s1"]) and cal.s2_ref == pytest.approx(tr["s2"])
    assert cal.mode == "stickers" and cal.created_at


def test_from_clicks_off_line_clicks_project_onto_sticker_line():
    tr = _truth(center=(160, 120), angle_deg=-35.0, px_per_mm=1.5)
    n = np.array(tr["axis"].normal)
    p0 = tuple(np.array(tr["p0"]) + 7.0 * n)      # user clicked beside the tube, on the printed line
    p1 = tuple(np.array(tr["p1"]) - 4.0 * n)
    cal = StickerCalibration.from_clicks(tr["s1"], tr["s2"], p0, p1)
    assert cal.u0 == pytest.approx(U0, abs=1e-9)
    assert cal.u1 == pytest.approx(U1, abs=1e-9)


def test_from_clicks_custom_range_and_to_u_inverse():
    tr = _truth()
    cal = _cal_from_truth(tr, g0=100.0, g1=500.0)
    assert cal.to_grams(cal.u0) == pytest.approx(100.0)
    assert cal.to_grams(cal.u1) == pytest.approx(500.0)
    assert cal.to_u(cal.to_grams(0.4321)) == pytest.approx(0.4321)


def test_from_clicks_degenerate_raises():
    with pytest.raises(ValueError):
        StickerCalibration.from_clicks((10, 10), (10, 10), (0, 0), (5, 5))
    with pytest.raises(ValueError):
        StickerCalibration.from_clicks((0, 0), (100, 0), (20, 5), (20, -5))


def test_pose_invariance_one_calibration_many_poses():
    """Calibrate once, then read the scale at random positions / angles / distances: within 0.5 g."""
    cal = _cal_from_truth(_truth(center=(640, 360), angle_deg=5.0, px_per_mm=5.0))
    rng = np.random.default_rng(42)
    for _ in range(40):
        grams = float(rng.uniform(0, 600))
        tr = _truth(grams, center=(float(rng.uniform(100, 220)), float(rng.uniform(80, 160))),
                    angle_deg=float(rng.uniform(-180, 180)), px_per_mm=float(rng.uniform(0.6, 8.0)))
        assert _u_in_frame(tr) == pytest.approx(tr["u"], abs=1e-9)
        assert abs(cal.to_grams(_u_in_frame(tr)) - grams) < 0.5
        assert abs(cal.to_grams(u_of_grams(grams)) - grams) < 0.5


# ---- adjustment ---------------------------------------------------------------------------------

def test_tare_shifts_offset_keeps_slope():
    cal = _cal_from_truth(_truth())
    slope = cal.slope_g_per_u
    cal.tare(U0 + 0.01)
    assert cal.tare_u == pytest.approx(U0 + 0.01)
    assert cal.slope_g_per_u == slope
    assert cal.to_grams(U0 + 0.01) == pytest.approx(0.0, abs=1e-9)
    assert cal.to_grams(U0 + 0.11) == pytest.approx(0.1 * slope)


def test_fit_no_samples_changes_nothing():
    cal = _cal_from_truth(_truth())
    before = (cal.slope_g_per_u, cal.offset_g)
    rep = cal.fit()
    assert isinstance(rep, FitReport)
    assert rep.n == 0 and rep.residuals_g == [] and rep.rms_g == 0
    assert (cal.slope_g_per_u, cal.offset_g) == before


def test_fit_one_sample_shifts_offset_only():
    cal = _cal_from_truth(_truth())
    slope = cal.slope_g_per_u
    rng = np.random.default_rng(1)
    u = u_of_grams(250.0) + rng.normal(0, 0.002)
    cal.add_sample(u, 250.0)
    rep = cal.fit()
    assert rep.n == 1 and cal.slope_g_per_u == slope
    assert cal.to_grams(u) == pytest.approx(250.0)
    assert rep.max_abs_residual_g == pytest.approx(0.0, abs=1e-9)


def test_fit_least_squares_recovers_line_from_noisy_samples():
    cal = _cal_from_truth(_truth())               # clicked: 600 g over (U1 - U0)
    true_slope = 0.97 * cal.slope_g_per_u         # real spring 3 % softer, zero a bit off
    true_offset = -true_slope * (U0 + 0.004)
    rng = np.random.default_rng(0)
    us = np.linspace(U0, U1, 7)
    gs = true_offset + true_slope * us + rng.normal(0.0, 0.5, us.size)
    for u, g in zip(us, gs):
        cal.add_sample(u, g)
    rep = cal.fit()
    assert rep.n == 7
    assert cal.slope_g_per_u == pytest.approx(true_slope, rel=0.005)
    assert cal.offset_g == pytest.approx(true_offset, abs=2.0)
    assert rep.slope_g_per_px == cal.slope_g_per_u and rep.offset_g == cal.offset_g
    assert rep.rms_g < 1.0 and rep.max_abs_residual_g < 1.5
    assert rep.residuals_g[0] == pytest.approx(cal.to_grams(us[0]) - gs[0])


def test_fit_small_spread_acts_like_one_sample_at_the_mean():
    cal = _cal_from_truth(_truth())
    slope = cal.slope_g_per_u
    u_a, u_b = 0.40, 0.40 + MIN_FIT_SPREAD_U / 2
    cal.add_sample(u_a, 200.0)
    cal.add_sample(u_b, 230.0)
    rep = cal.fit()
    assert rep.n == 2 and cal.slope_g_per_u == slope
    assert cal.to_grams((u_a + u_b) / 2) == pytest.approx(215.0)


def test_check_sign():
    cal = _cal_from_truth(_truth())
    u = cal.to_u(300.0)
    assert cal.check(u, 300.0) == pytest.approx(0.0, abs=1e-9)
    assert cal.check(u, 295.0) == pytest.approx(5.0)
    assert cal.check(u, 310.0) == pytest.approx(-10.0)


def test_u_range():
    cal = _cal_from_truth(_truth())
    assert cal.u_range() == pytest.approx((U0 - 0.05, U1 + 0.05))
    assert cal.u_range(0.0) == pytest.approx((U0, U1))
    cal.tare(U0 - 0.02)
    cal.add_sample(U1 + 0.03, 610.0)
    assert cal.u_range(0.01) == pytest.approx((U0 - 0.03, U1 + 0.04))
    # Reversed order (u0 > u1) still gives (low, high).
    rev = StickerCalibration(u0=0.9, u1=0.1)
    assert rev.u_range(0.0) == pytest.approx((0.1, 0.9))


# ---- persistence --------------------------------------------------------------------------------

def _sample_cal():
    cal = _cal_from_truth(_truth(center=(150, 110), angle_deg=12.5, px_per_mm=1.7))
    cal.tare(U0 + 0.003)
    cal.add_sample(0.3, 180.0)
    cal.add_sample(0.6, 420.5)
    cal.fit()
    cal.notes = "hand-held, 720p"
    return cal


def test_save_load_round_trip(tmp_path):
    cal = _sample_cal()
    path = tmp_path / "nested" / "dir" / "calib.json"
    cal.save(path)
    data = json.loads(path.read_text())
    assert data["schema"] == "scale-calib/2" and data["mode"] == "stickers"
    assert not path.with_name(path.name + ".tmp").exists()

    back = StickerCalibration.load(path)
    assert back == cal
    assert isinstance(back.samples[0], tuple) and isinstance(back.s1_ref, tuple)
    assert back.to_grams(0.5) == cal.to_grams(0.5)


def test_round_trip_without_refs_or_tare(tmp_path):
    cal = StickerCalibration(u0=U0, u1=U1, slope_g_per_u=600 / (U1 - U0), offset_g=-600 / (U1 - U0) * U0)
    path = tmp_path / "c.json"
    cal.save(path)
    back = StickerCalibration.load(path)
    assert back == cal and back.s1_ref is None and back.tare_u is None


def test_load_calibration_dispatches_both_schemas(tmp_path):
    sp = tmp_path / "stickers.json"
    _sample_cal().save(sp)
    pp = tmp_path / "pixels.json"
    Calibration.from_clicks((100.0, 400.0), (130.0, 40.0)).save(pp)
    assert isinstance(load_calibration(sp), StickerCalibration)
    assert load_calibration(sp) == StickerCalibration.load(sp)
    assert isinstance(load_calibration(pp), Calibration)
    assert load_calibration(pp) == Calibration.load(pp)


@pytest.mark.parametrize("payload", [
    {"schema": "something-else"},
    {"schema": "scale-calib/2", "mode": "pixels"},
    {},
])
def test_load_calibration_rejects_unknown(tmp_path, payload):
    path = tmp_path / "c.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_calibration(path)


def test_cross_load_errors(tmp_path):
    sp = tmp_path / "stickers.json"
    _sample_cal().save(sp)
    pp = tmp_path / "pixels.json"
    Calibration.from_clicks((100.0, 400.0), (130.0, 40.0)).save(pp)
    with pytest.raises(ValueError, match="load_calibration"):
        Calibration.load(sp)
    with pytest.raises(ValueError, match="load_calibration"):
        StickerCalibration.load(pp)
