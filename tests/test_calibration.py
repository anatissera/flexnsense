import json
import math

import numpy as np
import pytest

from vision.calibration import DEFAULT_CALIB_PATH, Calibration, FitReport
from vision.geometry import Axis

P0 = (100.0, 400.0)
P1 = (130.0, 40.0)   # slightly tilted axis, length = hypot(30, 360)


def test_from_clicks_maps_endpoints_exactly():
    cal = Calibration.from_clicks(P0, P1)
    L = math.hypot(30.0, 360.0)
    assert cal.axis == Axis(P0, P1)
    assert cal.slope_g_per_px == pytest.approx(600.0 / L)
    assert cal.offset_g == 0.0
    assert cal.to_grams(0.0) == pytest.approx(0.0, abs=1e-9)
    assert cal.to_grams(cal.axis.length) == pytest.approx(600.0, abs=1e-9)
    # Through the axis projection, too.
    assert cal.to_grams(cal.axis.project(P1)[0]) == pytest.approx(600.0, abs=1e-9)
    assert cal.to_t(600.0) == pytest.approx(L)
    assert cal.created_at


def test_from_clicks_custom_range():
    cal = Calibration.from_clicks(P0, P1, g0=100.0, g1=500.0)
    assert cal.to_grams(0.0) == pytest.approx(100.0)
    assert cal.to_grams(cal.axis.length) == pytest.approx(500.0)
    assert cal.to_t(cal.to_grams(123.4)) == pytest.approx(123.4)


def test_tare_shifts_offset_keeps_slope():
    cal = Calibration.from_clicks(P0, P1)
    slope = cal.slope_g_per_px
    cal.tare(4.5)
    assert cal.tare_t_px == 4.5
    assert cal.slope_g_per_px == slope
    assert cal.to_grams(4.5) == pytest.approx(0.0, abs=1e-12)
    assert cal.to_grams(4.5 + 100.0) == pytest.approx(100.0 * slope)


def test_fit_no_samples_changes_nothing():
    cal = Calibration.from_clicks(P0, P1)
    before = (cal.slope_g_per_px, cal.offset_g)
    rep = cal.fit()
    assert isinstance(rep, FitReport)
    assert rep.n == 0 and rep.residuals_g == [] and rep.max_abs_residual_g == 0 and rep.rms_g == 0
    assert (cal.slope_g_per_px, cal.offset_g) == before


def test_fit_one_sample_shifts_offset_only():
    cal = Calibration.from_clicks(P0, P1)
    slope = cal.slope_g_per_px
    cal.add_sample(150.0, 255.0)
    rep = cal.fit()
    assert rep.n == 1
    assert cal.slope_g_per_px == slope
    assert cal.to_grams(150.0) == pytest.approx(255.0)
    assert rep.max_abs_residual_g == pytest.approx(0.0, abs=1e-9)


def test_fit_least_squares_recovers_line_from_noisy_samples():
    cal = Calibration.from_clicks(P0, P1)          # clicked slope ~1.66 g/px
    true_slope, true_offset = 1.62, -3.0           # the "real" scale differs a little
    rng = np.random.default_rng(0)
    ts = np.array([10.0, 70.0, 130.0, 190.0, 250.0, 310.0, 350.0])
    gs = true_offset + true_slope * ts + rng.normal(0.0, 0.5, ts.size)
    for t, g in zip(ts, gs):
        cal.add_sample(t, g)
    rep = cal.fit()
    assert rep.n == len(ts)
    assert cal.slope_g_per_px == pytest.approx(true_slope, abs=0.01)
    assert cal.offset_g == pytest.approx(true_offset, abs=1.0)
    assert rep.slope_g_per_px == cal.slope_g_per_px and rep.offset_g == cal.offset_g
    assert len(rep.residuals_g) == len(ts)
    assert rep.rms_g < 1.0 and rep.max_abs_residual_g < 1.5
    assert rep.max_abs_residual_g == pytest.approx(max(abs(r) for r in rep.residuals_g))
    # Residual sign: reading - truth.
    assert rep.residuals_g[0] == pytest.approx(cal.to_grams(ts[0]) - gs[0])


def test_fit_small_spread_acts_like_one_sample_at_the_mean():
    cal = Calibration.from_clicks(P0, P1)
    slope = cal.slope_g_per_px
    cal.add_sample(100.0, 170.0)
    cal.add_sample(110.0, 200.0)                   # spread 10 px < 20 px: slope would be junk
    rep = cal.fit()
    assert rep.n == 2
    assert cal.slope_g_per_px == slope
    assert cal.to_grams(105.0) == pytest.approx(185.0)


def test_check_sign():
    cal = Calibration.from_clicks(P0, P1)
    t = cal.to_t(300.0)
    assert cal.check(t, 300.0) == pytest.approx(0.0, abs=1e-9)
    assert cal.check(t, 295.0) == pytest.approx(5.0)    # reads high -> positive
    assert cal.check(t, 310.0) == pytest.approx(-10.0)  # reads low -> negative


def test_save_load_round_trip(tmp_path):
    cal = Calibration.from_clicks((12.5, 300.25), (14.0, 20.75), g0=0.0, g1=600.0)
    cal.tare(2.0)
    cal.add_sample(100.0, 210.0)
    cal.add_sample(200.0, 425.5)
    cal.notes = "kitchen table, 720p"
    path = tmp_path / "nested" / "dir" / "calib.json"
    cal.save(path)

    data = json.loads(path.read_text())
    assert data["schema"] == "scale-calib/1"

    back = Calibration.load(path)
    assert back == cal
    assert back.samples == [(100.0, 210.0), (200.0, 425.5)]
    assert isinstance(back.samples[0], tuple)
    assert back.to_grams(123.0) == cal.to_grams(123.0)


def test_load_rejects_unknown_schema(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"schema": "something-else"}))
    with pytest.raises(ValueError):
        Calibration.load(path)


def test_default_path():
    assert DEFAULT_CALIB_PATH == "config/scale_calib.json"


def test_load_rejects_sticker_schema_with_pointer(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"schema": "scale-calib/2", "mode": "stickers"}))
    with pytest.raises(ValueError, match="load_calibration"):
        Calibration.load(path)
