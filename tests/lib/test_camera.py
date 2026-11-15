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


def test_list_cameras_uses_opencv_order(monkeypatch):
    # OpenCV numbers cameras sorted by unique ID, not in the order the system lists them
    import json
    import subprocess
    import types
    listing = {"SPCameraDataType": [
        {"_name": "MacBook Air Camera", "spcamera_unique-id": "6C707041-05AC-0010-000C-000000000001"},
        {"_name": "ROG EYE S", "spcamera_unique-id": "0x1200000b0519c0"},
        {"_name": "Someone's iPhone Camera", "spcamera_unique-id": "E45F980F-8AB2-4CAD-A46A-A71E00000001"}]}
    monkeypatch.setattr(camera.sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=json.dumps(listing), stderr=""))
    assert camera.list_cameras() == [(0, "ROG EYE S"), (1, "MacBook Air Camera"), (2, "Someone's iPhone Camera")]


# ------------------------------------------------------------------ operating system: macOS and Linux
# The platform is faked (sys.platform) and so are sysfs, /dev and cv2.VideoCapture: no camera is opened.

def _fake_v4l(tmp_path, monkeypatch, nodes, dev=()):
    """A fake /sys/class/video4linux (name -> (device name, index)) and a fake /dev."""
    sysfs, devdir = tmp_path / "sys", tmp_path / "dev"
    sysfs.mkdir()
    devdir.mkdir()
    for node, (name, index) in nodes.items():
        (sysfs / node).mkdir()
        if name is not None:
            (sysfs / node / "name").write_text(name + "\n")
        if index is not None:
            (sysfs / node / "index").write_text(f"{index}\n")
    for d in dev:
        (devdir / d).write_text("")
    monkeypatch.setattr(camera.sys, "platform", "linux")
    monkeypatch.setattr(camera, "V4L_SYSFS", sysfs)
    monkeypatch.setattr(camera, "V4L_DEV", devdir)


@pytest.mark.parametrize("platform, expected", [("darwin", "macos"), ("linux", "linux"), ("linux2", "linux"),
                                                ("win32", "other"), ("freebsd14", "other")])
def test_host_os(monkeypatch, platform, expected):
    monkeypatch.setattr(camera.sys, "platform", platform)
    assert camera.host_os() == expected


def test_list_cameras_linux_from_sysfs(tmp_path, monkeypatch):
    # a UVC webcam makes two nodes (picture = index 0, metadata = index 1); the number is the /dev/videoN number
    _fake_v4l(tmp_path, monkeypatch, {
        "video0": ("Integrated Camera: Integrated C", 0), "video1": ("Integrated Camera: Integrated C", 1),
        "video10": ("OBS Virtual Camera", 0),
        "video2": ("ROG EYE S: ROG EYE S", 0), "video3": ("ROG EYE S: ROG EYE S", 1),
        "v4l-subdev0": ("not a camera", 0)})
    assert camera.list_cameras() == [(0, "Integrated Camera: Integrated C"), (2, "ROG EYE S"),
                                     (10, "OBS Virtual Camera")]


def test_list_cameras_linux_does_not_run_any_tool(tmp_path, monkeypatch):
    import subprocess

    def boom(*a, **k):
        raise AssertionError("the Linux listing must not depend on an external tool")
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(camera.shutil, "which", boom)
    _fake_v4l(tmp_path, monkeypatch, {"video0": ("USB Camera", 0)})
    assert camera.list_cameras() == [(0, "USB Camera")]


def test_list_cameras_linux_incomplete_sysfs(tmp_path, monkeypatch):
    # no index file: keep the node; no name file: fall back to the device path
    _fake_v4l(tmp_path, monkeypatch, {"video0": ("USB Camera", None), "video4": (None, 0)})
    assert camera.list_cameras() == [(0, "USB Camera"), (4, "/dev/video4")]


def test_list_cameras_linux_falls_back_to_dev(tmp_path, monkeypatch):
    _fake_v4l(tmp_path, monkeypatch, {}, dev=["video2", "video0", "null", "videox"])
    cams = camera.list_cameras()
    assert [i for i, _ in cams] == [0, 2] and cams[0][1].endswith("video0")


def test_list_cameras_linux_nothing(tmp_path, monkeypatch, capsys):
    _fake_v4l(tmp_path, monkeypatch, {})
    monkeypatch.setattr(camera, "V4L_SYSFS", tmp_path / "missing")
    assert camera.list_cameras() == []
    assert camera.main() == 1
    out = capsys.readouterr().out
    assert "macOS + ffmpeg" not in out and "plugged in" in out


def test_resolve_camera_by_name_linux(tmp_path, monkeypatch):
    # issue #44: on Linux a name fragment never matched, because the listing was empty
    _fake_v4l(tmp_path, monkeypatch, {"video0": ("Integrated Camera", 0), "video2": ("ROG EYE S: ROG EYE S", 0)})
    assert camera.resolve_camera("rog") == 2
    assert camera.resolve_camera("/dev/video2") == "/dev/video2"      # a device path goes through untouched
    with pytest.raises(SystemExit):
        camera.resolve_camera("iphone")


def test_main_lists_linux_cameras(tmp_path, monkeypatch, capsys):
    _fake_v4l(tmp_path, monkeypatch, {"video0": ("USB Camera", 0)})
    assert camera.main() == 0
    assert "0   USB Camera" in capsys.readouterr().out


def test_list_cameras_other_os(monkeypatch):
    monkeypatch.setattr(camera.sys, "platform", "win32")
    assert camera.list_cameras() == []


def test_backends_and_hints_by_os(monkeypatch):
    monkeypatch.setattr(camera.sys, "platform", "darwin")
    assert camera.camera_backends() == [cv2.CAP_AVFOUNDATION] and camera.camera_hint() == camera.MAC_CAMERA_HINT
    monkeypatch.setattr(camera.sys, "platform", "linux")
    assert camera.camera_backends() == [cv2.CAP_V4L2, cv2.CAP_ANY] and "video" in camera.camera_hint()
    monkeypatch.setattr(camera.sys, "platform", "win32")
    assert camera.camera_backends() == [cv2.CAP_ANY] and camera.camera_hint() is None


class _FakeCapture:
    """Stands in for cv2.VideoCapture: records how it was opened, opens only with the backends in `works`."""
    works: set = set()
    calls: list = []

    def __init__(self, *args):
        type(self).calls.append(args)
        self.opened = len(args) < 2 or args[1] in type(self).works

    def isOpened(self):
        return self.opened

    def release(self):
        self.opened = False

    def set(self, prop, value):
        return True

    def get(self, prop):
        return 30.0

    def read(self):
        return True, np.full((4, 4, 3), 128, np.uint8)


def _fake_capture(monkeypatch, platform, works):
    _FakeCapture.works, _FakeCapture.calls = set(works), []
    monkeypatch.setattr(camera.sys, "platform", platform)
    monkeypatch.setattr(camera.cv2, "VideoCapture", _FakeCapture)
    return _FakeCapture.calls


def test_open_camera_backend_macos(monkeypatch):
    calls = _fake_capture(monkeypatch, "darwin", {cv2.CAP_AVFOUNDATION})
    assert camera.open_camera(1).is_camera
    assert calls == [(1, cv2.CAP_AVFOUNDATION)]


def test_open_camera_backend_linux(monkeypatch):
    calls = _fake_capture(monkeypatch, "linux", {cv2.CAP_V4L2})
    assert camera.open_camera(2).is_camera
    assert calls == [(2, cv2.CAP_V4L2)]


def test_open_camera_linux_falls_back_when_v4l2_is_missing(monkeypatch):
    calls = _fake_capture(monkeypatch, "linux", {cv2.CAP_ANY})
    assert camera.open_camera(0).is_camera
    assert calls == [(0, cv2.CAP_V4L2), (0, cv2.CAP_ANY)]


def test_open_camera_linux_device_path(monkeypatch):
    calls = _fake_capture(monkeypatch, "linux", set())
    assert camera.open_camera("/dev/video2").is_camera
    assert calls == [("/dev/video2",)]


@pytest.mark.parametrize("platform, hint", [("linux", "usermod -aG video"), ("darwin", "Privacy & Security")])
def test_open_camera_failure_prints_the_hint_for_this_os(monkeypatch, capsys, platform, hint):
    _fake_capture(monkeypatch, platform, set())
    with pytest.raises(SystemExit):
        camera.open_camera(0)
    assert hint in capsys.readouterr().err
