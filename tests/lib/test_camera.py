"""lib.camera without a real camera: name resolution, defaults and the video-file source."""
import cv2
import numpy as np
import pytest

from lib import camera


def test_resolve_camera_passthrough():
    assert camera.resolve_camera(1) == 1
    assert camera.resolve_camera("2") == 2
    assert camera.resolve_camera("rtsp://host/stream") == "rtsp://host/stream"


def test_resolve_camera_by_name(monkeypatch):
    monkeypatch.setattr(camera, "list_cameras", lambda: [(0, "MacBook Air Camera"), (1, "Someone's iPhone Camera"),
                                                         (2, "Someone's iPhone Desk View Camera")])
    assert camera.resolve_camera("iphone") == 1          # desk view is skipped
    assert camera.resolve_camera("MacBook") == 0
    with pytest.raises(SystemExit):
        camera.resolve_camera("webcam-that-does-not-exist")


def test_default_camera_from_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CAMERA_INDEX", raising=False)
    assert camera.default_camera() == 0
    (tmp_path / ".env").write_text("CAMERA_INDEX=iphone\n")
    assert camera.default_camera() == "iphone"
    monkeypatch.setenv("CAMERA_INDEX", "3")
    assert camera.default_camera() == 3


def test_open_source_video_file(tmp_path):
    path = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 48))
    for i in range(5):
        vw.write(np.full((48, 64, 3), 40 * i, np.uint8))
    vw.release()
    src = camera.open_source(video=path)
    assert not src.is_camera and (src.width, src.height) == (64, 48) and src.fps == pytest.approx(20.0)
    stamps = []
    while True:
        ok, frame, ts, idx = src.read()
        if not ok:
            break
        stamps.append((idx, ts))
    src.release()
    assert stamps == [(i, pytest.approx(i / 20.0)) for i in range(5)]


def test_open_video_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        camera.open_source(video=tmp_path / "nope.mp4")
