from arm1.gauge_reader.launcher import MODES, build_cmd


def test_build_cmd():
    assert build_cmd("calibrate", 1)[3:] == ["gauge_reader", "calibrate", "--camera", "1", "--mode", "stickers"]
    assert build_cmd("recalibrate", 0)[4:6] == ["calibrate", "--resume"]
    assert build_cmd("run", 1, record=True)[-1] == "--record"
    assert "--record" not in build_cmd("calibrate", 1, record=True)


def test_build_cmd_mode():
    assert build_cmd("calibrate", 0, mode="fixed")[-2:] == ["--mode", "fixed"]
    assert "--mode" not in build_cmd("run", 0, mode="fixed")            # run: mode comes from the calibration file
    assert "--mode" not in build_cmd("recalibrate", 0, mode="fixed")    # --resume: same
    assert MODES[0][1] == "stickers"                                    # default selection


def test_build_cmd_parses():
    from arm1.gauge_reader.cli import build_parser
    for mode in ("stickers", "fixed"):
        a = build_parser().parse_args(build_cmd("calibrate", 2, mode=mode)[4:])
        assert a.mode == mode and a.camera == 2


def test_calib_summary_both_modes(tmp_path, monkeypatch):
    import pytest
    import arm1.gauge_reader.launcher as L
    vc = pytest.importorskip("arm1.gauge_reader.calibration")
    monkeypatch.setattr(L, "CALIB", tmp_path / "c.json")
    assert "Not calibrated" in L.calib_summary()
    vc.Calibration.from_clicks((0, 0), (0, 600)).save(L.CALIB)
    assert "fixed mode" in L.calib_summary()
    if hasattr(vc, "StickerCalibration"):
        vc.StickerCalibration.from_clicks((0, 0), (0, 124), (0, 12), (0, 112)).save(L.CALIB)
        assert "stickers mode" in L.calib_summary()


def test_camera_choices_on_linux(tmp_path, monkeypatch):
    # the panel's camera list on a faked Linux: V4L2 numbers and names, no iPhone needed for a default
    import arm1.gauge_reader.launcher as L
    from arm1.lib import camera
    (tmp_path / "video2").mkdir()
    (tmp_path / "video2" / "name").write_text("ROG EYE S: ROG EYE S\n")
    (tmp_path / "video2" / "index").write_text("0\n")
    monkeypatch.setattr(camera.sys, "platform", "linux")
    monkeypatch.setattr(camera, "V4L_SYSFS", tmp_path)
    assert L.camera_choices() == [(2, "ROG EYE S")]
    monkeypatch.setattr(camera, "V4L_SYSFS", tmp_path / "missing")
    monkeypatch.setattr(camera, "V4L_DEV", tmp_path / "missing")
    assert L.camera_choices() == [(0, "Camera 0"), (1, "Camera 1"), (2, "Camera 2")]
