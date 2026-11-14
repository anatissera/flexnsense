from vision.launcher import MODES, build_cmd


def test_build_cmd():
    assert build_cmd("calibrate", 1)[3:] == ["vision.scale_reader", "calibrate", "--camera", "1", "--mode", "stickers"]
    assert build_cmd("recalibrate", 0)[4:6] == ["calibrate", "--resume"]
    assert build_cmd("run", 1, record=True)[-1] == "--record"
    assert "--record" not in build_cmd("calibrate", 1, record=True)


def test_build_cmd_mode():
    assert build_cmd("calibrate", 0, mode="fixed")[-2:] == ["--mode", "fixed"]
    assert "--mode" not in build_cmd("run", 0, mode="fixed")            # run: mode comes from the calibration file
    assert "--mode" not in build_cmd("recalibrate", 0, mode="fixed")    # --resume: same
    assert MODES[0][1] == "stickers"                                    # default selection


def test_build_cmd_parses():
    from vision.scale_reader import build_parser
    for mode in ("stickers", "fixed"):
        a = build_parser().parse_args(build_cmd("calibrate", 2, mode=mode)[4:])
        assert a.mode == mode and a.camera == 2


def test_calib_summary_both_modes(tmp_path, monkeypatch):
    import pytest
    import vision.launcher as L
    vc = pytest.importorskip("vision.calibration")
    monkeypatch.setattr(L, "CALIB", tmp_path / "c.json")
    assert "Not calibrated" in L.calib_summary()
    vc.Calibration.from_clicks((0, 0), (0, 600)).save(L.CALIB)
    assert "fixed mode" in L.calib_summary()
    if hasattr(vc, "StickerCalibration"):
        vc.StickerCalibration.from_clicks((0, 0), (0, 124), (0, 12), (0, 112)).save(L.CALIB)
        assert "stickers mode" in L.calib_summary()
