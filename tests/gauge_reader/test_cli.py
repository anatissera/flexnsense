"""Tests for the scale reader CLI (gauge_reader.cli) and overlays (gauge_reader.overlay). No camera, no windows.

The end-to-end tests need gauge_reader.detector / calibration / smoothing (fixed mode) and gauge_reader.stickers +
gauge_reader.calibration.StickerCalibration (sticker mode); they are skipped until those exist.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from gauge_reader import overlay
from lib.geometry import Axis
from gauge_reader.synthetic.still import make_video, sweep

REPO = Path(__file__).resolve().parents[2]   # tests/gauge_reader/test_cli.py -> repo root
N_FRAMES = 72


def _need_modules():
    pytest.importorskip("gauge_reader.detector")
    pytest.importorskip("gauge_reader.calibration")
    pytest.importorskip("lib.filters")


def _need_sticker_modules():
    _need_modules()
    pytest.importorskip("gauge_reader.stickers")
    vc = pytest.importorskip("gauge_reader.calibration")
    if not (hasattr(vc, "StickerCalibration") and hasattr(vc, "load_calibration")):
        pytest.skip("gauge_reader.calibration.StickerCalibration / load_calibration not there yet")


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("vid")
    return make_video(d / "syn.mp4", sweep(N_FRAMES), fps=30.0, size=(960, 720), angle_deg=80.0)


@pytest.fixture()
def calib_file(video, tmp_path) -> Path:
    _need_modules()
    from gauge_reader.calibration import Calibration
    from gauge_reader.cli import read_truth
    axis, _ = read_truth(video.with_suffix(".truth.csv"))
    p = tmp_path / "calib.json"
    Calibration.from_clicks(axis.p0, axis.p1).save(p)
    return p


# ------------------------------------------------------------------------------------------- overlay

def test_overlay_functions_draw_on_blank_frame():
    img = np.zeros((720, 1280, 3), np.uint8)
    axis = Axis((640.0, 60.0), (640.0, 660.0))
    corners = np.array([[610, 30], [610, 690], [670, 690], [670, 30]], float)
    mask = np.zeros((51, 661), np.uint8)
    mask[:, 300:310] = 255
    strip = np.full((51, 661, 3), 128, np.uint8)
    overlay.draw_run_overlay(img, corners=corners, point=(640.0, 300.0), grams=123.4, std_g=0.8, stable=True,
                             found=True, fps=29.9, mask=mask, strip=strip, show_mask=True, help_lines=["q quit"])
    overlay.draw_run_overlay(img, grams=None, found=False, fps=10.0)          # "no marker" path
    overlay.draw_reading(img, 50.0, float("nan"), stable=False, found=True)   # nan std is tolerated
    overlay.draw_axis(img, axis)
    overlay.draw_blobs(img, [(100.0, 100.0, 400), (300.0, 300.0, 50)])
    overlay.draw_clicks(img, [(640.0, 60.0)], cursor=(700, 500))
    overlay.draw_loupe(img, img.copy(), (5, 5), zoom=4)                       # near the border
    overlay.draw_loupe(img, img.copy(), (1275, 715), zoom=4)
    overlay.draw_banner(img, "unsaved")
    overlay.draw_help(img, ["a", "b"])
    assert img.any()


def test_status_colors():
    assert overlay.status_color(True, True) == overlay.GREEN
    assert overlay.status_color(True, False) == overlay.AMBER
    assert overlay.status_color(False, True) == overlay.RED
    assert overlay.format_grams(None) == "-- g"


def test_parser_has_commands():
    from gauge_reader.cli import build_parser
    ap = build_parser()
    a = ap.parse_args(["calibrate", "--p0", "10,20", "--p1", "10.5,620", "--camera", "1"])
    assert a.p0 == (10.0, 20.0) and a.p1 == (10.5, 620.0) and a.camera == 1
    a = ap.parse_args(["run", "--headless", "--video", "x.mp4", "--max-frames", "5"])
    assert a.headless and a.max_frames == 5 and a.width == 1920 and a.height == 1080
    a = ap.parse_args(["evaluate", "--video", "x.mp4"])
    assert a.truth is None


def test_parser_mode():
    from gauge_reader.cli import build_parser
    ap = build_parser()
    assert ap.parse_args(["calibrate"]).mode == "stickers"                      # default
    assert ap.parse_args(["calibrate", "--mode", "fixed"]).mode == "fixed"
    a = ap.parse_args(["calibrate", "--p0", "1,2", "--p1", "3,4", "--s1", "0,0", "--s2", "5,5"])
    assert a.s1 == (0.0, 0.0) and a.s2 == (5.0, 5.0)
    with pytest.raises(SystemExit):
        ap.parse_args(["calibrate", "--mode", "wobbly"])


# ------------------------------------------------------------------------------------------- end to end

def test_evaluate_passes_on_synthetic_video(video):
    _need_modules()
    from gauge_reader.cli import evaluate
    m = evaluate(video)
    assert m["frames"] == N_FRAMES
    assert m["detection_rate"] >= 0.95
    assert m["raw_mean_g"] <= 3.0
    assert m["passed"]


def test_evaluate_cli_exit_codes(video, tmp_path):
    _need_modules()
    cmd = [sys.executable, "-m", "gauge_reader", "evaluate", "--video", str(video)]
    ok = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=120)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "detection rate" in ok.stdout and "PASS" in ok.stdout

    # same video, truth shifted by +20 g -> mean raw error > 3 g -> exit code 1
    rows = list(csv.DictReader(open(video.with_suffix(".truth.csv"))))
    bad = tmp_path / "bad.truth.csv"
    with open(bad, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "grams": f"{float(r['grams']) + 20:.3f}"})
    fail = subprocess.run(cmd + ["--truth", str(bad)], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert fail.returncode == 1, fail.stdout + fail.stderr
    assert "FAIL" in fail.stdout


def test_run_headless_writes_csv_and_strip_only(video, calib_file, tmp_path, monkeypatch, capsys):
    _need_modules()
    from gauge_reader.cli import CSV_COLUMNS, main
    monkeypatch.chdir(tmp_path)     # recordings/ goes to tmp
    out = tmp_path / "session.csv"
    rc = main(["run", "--video", str(video), "--calib", str(calib_file), "--csv", str(out), "--headless", "--record"])
    assert rc == 0
    assert "fps" in capsys.readouterr().out          # at least one status line

    rows = list(csv.DictReader(open(out)))
    assert list(rows[0].keys()) == CSV_COLUMNS
    assert len(rows) == N_FRAMES
    assert sum(int(r["found"]) for r in rows) >= 0.95 * N_FRAMES
    truth = {int(r["frame"]): float(r["grams"]) for r in csv.DictReader(open(video.with_suffix(".truth.csv")))}
    errs = [abs(float(r["raw_g"]) - truth[int(r["frame"])]) for r in rows if r["raw_g"]]
    assert np.mean(errs) <= 3.0
    assert float(rows[1]["time_s"]) == pytest.approx(1 / 30.0, abs=1e-3)   # video timestamps = frame / fps

    vids = list((tmp_path / "recordings").glob("strip-*.mp4"))
    assert len(vids) == 1
    cap = cv2.VideoCapture(str(vids[0]))
    ok, f = cap.read()
    cap.release()
    assert ok
    assert f.shape[:2] != (720, 960)          # never a full frame: the strip is long and thin
    assert f.shape[0] < f.shape[1] / 4


def test_run_max_frames_and_default_csv(video, calib_file, tmp_path, monkeypatch):
    _need_modules()
    from gauge_reader.cli import main
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--video", str(video), "--calib", str(calib_file), "--headless", "--max-frames", "10"]) == 0
    csvs = list((tmp_path / "recordings").glob("session-*.csv"))
    assert len(csvs) == 1
    assert len(list(csv.DictReader(open(csvs[0])))) == 10


def test_run_without_calibration_explains(tmp_path, video):
    _need_modules()
    from gauge_reader.cli import main
    with pytest.raises(SystemExit, match="calibrate"):
        main(["run", "--video", str(video), "--calib", str(tmp_path / "missing.json"), "--headless"])


def test_pipeline_tare_and_known_weight(video):
    _need_modules()
    from gauge_reader.calibration import Calibration
    from gauge_reader.cli import Pipeline, read_truth
    from gauge_reader.synthetic.still import render_scale_frame
    axis, _ = read_truth(video.with_suffix(".truth.csv"))
    # deliberately bad clicks: p0 4 px too far down the axis
    calib = Calibration.from_clicks(axis.point_at(4.0), axis.p1)
    pipe = Pipeline.from_calibration(calib)
    kw = dict(size=(960, 720), angle_deg=80.0)
    for i in range(15):
        pipe.process(render_scale_frame(0.0, seed=i, **kw)[0], i / 30)
    assert pipe.tare() is not None
    for i in range(15):
        pipe.process(render_scale_frame(400.0, seed=100 + i, **kw)[0], 1 + i / 30)
    t, report = pipe.add_sample(400.0)
    assert report.n == 1 and report.max_abs_residual_g < 0.5
    r = pipe.process(render_scale_frame(200.0, seed=999, **kw)[0], 3.0)
    assert r.raw_g == pytest.approx(200.0, abs=3.0)


# ------------------------------------------------------------------------------------------- GUI paths (faked)

class FakeGUI:
    """Replaces cv2's window functions: feeds a key script to waitKey and can fire mouse clicks."""

    def __init__(self, monkeypatch, script):
        self.script = list(script)       # items: key char, None (no key), or ("click", x, y)
        self.shown = 0
        self.mouse = None
        for name in ("namedWindow", "resizeWindow", "destroyAllWindows"):
            monkeypatch.setattr(cv2, name, lambda *a, **k: None)
        monkeypatch.setattr(cv2, "imshow", self.imshow)
        monkeypatch.setattr(cv2, "waitKey", self.wait_key)
        monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1.0)
        monkeypatch.setattr(cv2, "setMouseCallback", lambda name, cb: setattr(self, "mouse", cb))

    def imshow(self, name, img):
        assert img.ndim == 3
        self.shown += 1

    def wait_key(self, delay=0):
        while self.script:
            item = self.script.pop(0)
            if isinstance(item, tuple):
                _, x, y = item
                self.mouse(cv2.EVENT_MOUSEMOVE, x, y, 0, None)
                self.mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
                continue
            return 255 if item is None else ord(item)
        return ord("q")


def test_run_window_keys(video, calib_file, tmp_path, monkeypatch):
    _need_modules()
    from gauge_reader.calibration import Calibration
    from gauge_reader.cli import main
    monkeypatch.chdir(tmp_path)
    gui = FakeGUI(monkeypatch, [None] * 5 + ["m", None, " ", None, " ", "z", "s", "q"])
    assert main(["run", "--video", str(video), "--calib", str(calib_file), "--csv", str(tmp_path / "s.csv")]) == 0
    assert gui.shown >= 10
    assert Calibration.load(calib_file).tare_t_px is not None      # z then s saved the tare


def test_calibrate_clicks_tare_weight_save(video, tmp_path, monkeypatch):
    _need_modules()
    from gauge_reader.calibration import Calibration
    from gauge_reader.cli import main, read_truth
    axis, _ = read_truth(video.with_suffix(".truth.csv"))
    out = tmp_path / "cal.json"
    monkeypatch.setattr("builtins.input", lambda prompt="": "0")
    script = [None, " ", ("click", *map(round, axis.p0)), ("click", *map(round, axis.p1)), "\r"]
    script += [None] * 8 + ["z", "w", None, "s", "q"]   # None: the w handler pumps the window once
    FakeGUI(monkeypatch, script)
    assert main(["calibrate", "--mode", "fixed", "--video", str(video), "--calib", str(out)]) == 0
    c = Calibration.load(out)
    assert c.axis.length == pytest.approx(axis.length, abs=2.0)
    assert c.tare_t_px is not None and len(c.samples) == 1
    assert abs(c.to_grams(c.to_t(300.0)) - 300.0) < 1e-6


def test_calibrate_warns_before_quitting_unsaved(video, tmp_path, monkeypatch, capsys):
    _need_modules()
    from gauge_reader.cli import main
    from gauge_reader.cli import read_truth
    axis, _ = read_truth(video.with_suffix(".truth.csv"))
    p0, p1 = (f"{v[0]},{v[1]}" for v in (axis.p0, axis.p1))
    FakeGUI(monkeypatch, [None, None, "q", None, "q"])
    out = tmp_path / "never.json"
    assert main(["calibrate", "--mode", "fixed", "--video", str(video), "--calib", str(out), "--p0", p0, "--p1", p1]) == 0
    assert "NOT saved" in capsys.readouterr().out
    assert not out.exists()


def test_run_headless_fixed_csv_has_sticker_columns_empty(video, calib_file, tmp_path):
    _need_modules()
    from gauge_reader.cli import main
    out = tmp_path / "s.csv"
    assert main(["run", "--video", str(video), "--calib", str(calib_file), "--csv", str(out), "--headless",
                 "--max-frames", "5"]) == 0
    rows = list(csv.DictReader(open(out)))
    assert all(r["u"] == "" and r["sticker_dist_px"] == "" for r in rows)
    assert all(r["reason"] == ("" if r["found"] == "1" else "no-marker") for r in rows)


# =========================================================================================== sticker mode

MOVING_FRAMES = 120


@pytest.fixture(scope="module")
def moving_video(tmp_path_factory) -> Path:
    from gauge_reader.synthetic.moving import make_moving_video
    d = tmp_path_factory.mktemp("mov")
    return make_moving_video(d / "moving.mp4", MOVING_FRAMES, seed=1, speed=2.0, occlude_every=30)


def _truth_rows(video: Path) -> list[dict]:
    return list(csv.DictReader(open(video.with_suffix(".truth.csv"))))


def _xy(row, k):
    return float(row[k + "x"]), float(row[k + "y"])


@pytest.fixture()
def sticker_calib_file(moving_video, tmp_path) -> Path:
    _need_sticker_modules()
    from gauge_reader.calibration import StickerCalibration
    r0 = _truth_rows(moving_video)[0]
    p = tmp_path / "stickers.json"
    StickerCalibration.from_clicks(_xy(r0, "s1"), _xy(r0, "s2"), _xy(r0, "p0"), _xy(r0, "p1")).save(p)
    return p


def test_sticker_overlay_functions_draw_on_blank_frame():
    img = np.zeros((720, 1280, 3), np.uint8)
    mask = np.zeros((160, 960), np.uint8)
    strip = np.full((160, 960, 3), 128, np.uint8)
    blobs = [(300.0, 300.0, 120), (700.0, 320.0, 80), (500.0, 310.0, 60)]
    overlay.draw_sticker_overlay(img, s1=(300.0, 300.0), s2=(700.0, 320.0), marker=(500.0, 310.0), grams=212.3,
                                 std_g=0.6, stable=True, found=True, fps=30.0, blobs=blobs, mask=mask, strip=strip,
                                 show_mask=True, help_lines=["q quit"], extra="REC", u_marks=(0.1, 0.9))
    for reason in ("no-scale", "sticker-hidden", "no-marker", "ambiguous", "", "something-new"):
        overlay.draw_sticker_overlay(img, s1=None, s2=None, marker=None, grams=None, found=False, reason=reason)
        overlay.draw_sticker_overlay(img, s1=(300.0, 300.0), s2=None, grams=150.0, found=False, reason=reason,
                                     show_mask=True, blobs=[], mask=None)     # last value greyed, no inset
    overlay.draw_stickers(img, (10.0, 10.0), (12.0, 10.0), None, u_marks=(0.5,))
    overlay.draw_candidates(img, None)
    overlay.draw_reading(img, 99.0, found=False, message="sticker hidden")
    assert img.any()
    assert overlay.reason_text("sticker-hidden") == "sticker hidden"
    assert overlay.reason_text("no-scale") == "scale not found"
    assert overlay.reason_text(None) == "not found"


def test_sticker_crop_is_only_the_scale():
    from gauge_reader.cli import CROP_SIZE, sticker_crop
    frame = np.zeros((720, 1280, 3), np.uint8)
    cv2.circle(frame, (400, 360), 10, (0, 0, 255), -1)
    crop = sticker_crop(frame, (400.0, 360.0), (900.0, 300.0))
    assert crop.shape == (CROP_SIZE[1], CROP_SIZE[0], 3)
    assert crop.shape[0] < crop.shape[1] / 4
    assert crop[:, :, 2].max() > 200                         # sticker 1 is inside the crop
    assert sticker_crop(frame, None, (1.0, 1.0)) is None
    assert sticker_crop(frame, (1.0, 1.0), (1.0, 1.0)) is None


def test_evaluate_stickers_passes_on_moving_video(moving_video):
    _need_sticker_modules()
    from gauge_reader.cli import evaluate, truth_is_stickers
    assert truth_is_stickers(moving_video.with_suffix(".truth.csv"))
    m = evaluate(moving_video)
    assert m["mode"] == "stickers" and m["frames"] == MOVING_FRAMES
    assert m["hidden_frames"] > 0 and m["visible_frames"] + m["hidden_frames"] == MOVING_FRAMES
    assert m["found_rate_visible"] >= 0.95
    assert m["raw_mean_g"] <= 3.0
    assert m["wrong_readings"] == 0 and m["wrong_hidden"] == 0
    assert m["passed"]


def test_evaluate_stickers_cli_fails_on_wrong_truth(moving_video, tmp_path):
    _need_sticker_modules()
    cmd = [sys.executable, "-m", "gauge_reader", "evaluate", "--video", str(moving_video)]
    ok = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=180)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "stickers mode" in ok.stdout and "wrong readings 0" in ok.stdout and "PASS" in ok.stdout

    rows = _truth_rows(moving_video)
    bad = tmp_path / "bad.truth.csv"
    with open(bad, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "grams": f"{float(r['grams']) + 20:.3f}"})     # every reading now 20 g "wrong"
    fail = subprocess.run(cmd + ["--truth", str(bad)], cwd=REPO, capture_output=True, text=True, timeout=180)
    assert fail.returncode == 1, fail.stdout + fail.stderr
    assert "FAIL" in fail.stdout


def test_run_headless_stickers_writes_new_columns_and_crop(moving_video, sticker_calib_file, tmp_path, monkeypatch,
                                                           capsys):
    _need_sticker_modules()
    from gauge_reader.cli import CROP_SIZE, CSV_COLUMNS, main
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "session.csv"
    assert main(["run", "--video", str(moving_video), "--calib", str(sticker_calib_file), "--csv", str(out),
                 "--headless", "--record"]) == 0
    assert "stickers mode" in capsys.readouterr().out

    rows = list(csv.DictReader(open(out)))
    assert list(rows[0].keys()) == CSV_COLUMNS
    assert {"u", "sticker_dist_px", "reason"} <= set(CSV_COLUMNS)
    assert len(rows) == MOVING_FRAMES
    truth = {int(r["frame"]): r for r in _truth_rows(moving_video)}
    found = [r for r in rows if r["found"] == "1"]
    lost = [r for r in rows if r["found"] == "0"]
    assert found and lost                                         # occlusions in the video
    for r in found:
        assert r["u"] and r["sticker_dist_px"] and r["reason"] == ""
        assert abs(float(r["u"]) - float(truth[int(r["frame"])]["u"])) < 0.01
        assert abs(float(r["raw_g"]) - float(truth[int(r["frame"])]["grams"])) < 10
    for r in lost:
        assert r["reason"] and r["u"] == "" and r["grams"] == "" and r["raw_g"] == ""   # never a reading
    hidden = [r for r in rows if truth[int(r["frame"])]["hidden"] != "0"]
    assert all(r["found"] == "0" for r in hidden)

    vids = list((tmp_path / "recordings").glob("crop-*.mp4"))
    assert len(vids) == 1
    cap = cv2.VideoCapture(str(vids[0]))
    ok, f = cap.read()
    cap.release()
    assert ok and f.shape[:2] != (720, 1280)                      # never the full frame
    assert f.shape[1] / f.shape[0] == pytest.approx(CROP_SIZE[0] / CROP_SIZE[1], rel=0.05)


def test_run_window_keys_stickers(moving_video, sticker_calib_file, tmp_path, monkeypatch):
    _need_sticker_modules()
    from gauge_reader.calibration import load_calibration
    from gauge_reader.cli import main
    monkeypatch.chdir(tmp_path)
    gui = FakeGUI(monkeypatch, [None] * 5 + ["m", None, " ", None, " ", "z", "s"] + [None] * 70 + ["q"])
    assert main(["run", "--video", str(moving_video), "--calib", str(sticker_calib_file),
                 "--csv", str(tmp_path / "s.csv")]) == 0
    assert gui.shown >= 70                                        # also through the hidden-sticker frames
    c = load_calibration(sticker_calib_file)
    assert c.mode == "stickers" and c.tare_u is not None


def test_calibrate_stickers_found_automatically(moving_video, tmp_path, monkeypatch):
    _need_sticker_modules()
    from gauge_reader.calibration import StickerCalibration, load_calibration
    from gauge_reader.cli import main
    r1 = _truth_rows(moving_video)[1]           # SPACE freezes frame 1 (frame 0 is shown with no key)
    out = tmp_path / "cal.json"
    monkeypatch.setattr("builtins.input", lambda prompt="": "0")
    script = [None, " ", ("click", *map(round, _xy(r1, "p0"))), ("click", *map(round, _xy(r1, "p1"))), None, "\r"]
    script += [None] * 8 + ["z", "w", None, "s", "q"]
    FakeGUI(monkeypatch, script)
    assert main(["calibrate", "--video", str(moving_video), "--calib", str(out)]) == 0     # default mode: stickers
    c = load_calibration(out)
    assert isinstance(c, StickerCalibration)
    assert np.hypot(*np.subtract(c.s1_ref, _xy(r1, "s1"))) < 2.0
    assert np.hypot(*np.subtract(c.s2_ref, _xy(r1, "s2"))) < 2.0
    assert c.u0 == pytest.approx(float(r1["u0"]), abs=0.01) and c.u1 == pytest.approx(float(r1["u1"]), abs=0.01)
    assert c.tare_u is not None and len(c.samples) == 1


def test_calibrate_stickers_clicked_by_hand(moving_video, tmp_path, monkeypatch):
    _need_sticker_modules()
    import gauge_reader.stickers
    from gauge_reader.calibration import load_calibration
    from gauge_reader.cli import main
    monkeypatch.setattr(gauge_reader.stickers, "find_stickers_near_line", lambda *a, **k: None)
    r1 = _truth_rows(moving_video)[1]
    s1, s2 = (tuple(map(round, _xy(r1, k))) for k in ("s1", "s2"))
    out = tmp_path / "cal.json"
    script = [None, " ", ("click", *map(round, _xy(r1, "p0"))), ("click", *map(round, _xy(r1, "p1"))), None,
              "\r",                                        # ENTER before the sticker clicks: ignored
              ("click", *s1), ("click", *s2), "\r", None, None, "s", "q"]
    FakeGUI(monkeypatch, script)
    assert main(["calibrate", "--video", str(moving_video), "--calib", str(out)]) == 0
    c = load_calibration(out)
    assert c.mode == "stickers" and c.s1_ref == pytest.approx(s1) and c.s2_ref == pytest.approx(s2)


def test_calibrate_stickers_from_args(moving_video, tmp_path, monkeypatch, capsys):
    _need_sticker_modules()
    from gauge_reader.calibration import load_calibration
    from gauge_reader.cli import main
    r0 = _truth_rows(moving_video)[0]
    p0, p1 = (f"{x},{y}" for x, y in (_xy(r0, "p0"), _xy(r0, "p1")))
    out = tmp_path / "cal.json"
    FakeGUI(monkeypatch, [None, "s", "q"])
    assert main(["calibrate", "--video", str(moving_video), "--calib", str(out), "--p0", p0, "--p1", p1]) == 0
    assert "stickers found" in capsys.readouterr().out
    c = load_calibration(out)
    assert np.hypot(*np.subtract(c.s1_ref, _xy(r0, "s1"))) < 2.0

    # --resume takes the mode from the file
    FakeGUI(monkeypatch, [None, None, "q"])
    assert main(["calibrate", "--mode", "fixed", "--resume", "--video", str(moving_video), "--calib", str(out)]) == 0
    assert "stickers mode" in capsys.readouterr().out
