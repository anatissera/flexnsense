"""Cameras and video files as one frame source, for any script in the repo.

    from lib.camera import open_source
    src = open_source(camera="iphone")            # or 0, 1, a URL ... ; default: CAMERA_INDEX from env/.env, else 0
    src = open_source(video="clip.mp4")           # same interface for a file
    ok, frame, ts, idx = src.read()               # BGR frame, timestamp in seconds, frame index
    src.release()

    python -m lib.camera                          # list cameras and their numbers (macOS)

macOS details handled here: AVFoundation backend, cameras chosen by name (an iPhone via Continuity Camera shows
up as "<name>'s iPhone Camera"), a warm-up that waits for real (non-black) frames, tolerance of short frame gaps
while an iPhone reconnects, and hints when permission or the iPhone connection is missing.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

from lib.env import env_get

WARMUP_S = 20.0
MAC_CAMERA_HINT = (
    "macOS: allow camera access for the app running Python - System Settings -> Privacy & Security -> Camera -> "
    "enable your terminal / IDE (Terminal, iTerm, VS Code, ...), then restart it. Also try another camera number."
)
IPHONE_HINT = ("  iPhone (Continuity Camera): lock the phone, keep it still in landscape near the Mac, Wi-Fi + Bluetooth on,\n"
               "  same Apple ID; the phone should show 'Connected to <Mac>'. Opening Photo Booth and picking the iPhone\n"
               "  in its Camera menu forces the connection. List cameras: python -m lib.camera")


def list_cameras() -> list[tuple[int, str]]:
    """(index, name) of the video devices, with the indices OpenCV uses (macOS). Empty if unknown.

    OpenCV's AVFoundation backend numbers the cameras sorted by their unique ID, which is NOT the order ffmpeg
    lists them in: with a USB webcam plugged in, ffmpeg's index 1 can be OpenCV's index 0. So the names come
    from system_profiler (name + unique ID), sorted the way OpenCV sorts; ffmpeg is only the fallback.
    """
    if sys.platform != "darwin":
        return []
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"],
                             capture_output=True, text=True, timeout=15).stdout
        cams = [(c["spcamera_unique-id"], c["_name"]) for c in json.loads(out)["SPCameraDataType"]]
        if cams:
            return [(i, name) for i, (_, name) in enumerate(sorted(cams))]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        pass
    if not shutil.which("ffmpeg"):
        return []
    out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                         capture_output=True, text=True).stderr
    video = out.split("audio devices")[0]
    return [(int(i), n.strip()) for i, n in re.findall(r"\[(\d+)\] (.+)", video) if "Capture screen" not in n]


def resolve_camera(cam: int | str) -> int | str:
    """A name fragment such as 'iphone' -> its index; a number, path or URL is returned as is."""
    if isinstance(cam, int) or "/" in cam or ":" in cam:
        return cam
    if cam.lstrip("-").isdigit():
        return int(cam)
    want = cam.lower()
    hits = [(i, n) for i, n in list_cameras() if want in n.lower() and "desk view" not in n.lower()]
    if not hits:
        raise SystemExit(f"error: no camera matching {cam!r}; list them with: python -m lib.camera")
    print(f"camera {cam!r} -> {hits[0][0]} ({hits[0][1]})")
    return hits[0][0]


def default_camera() -> int | str:
    """CAMERA_INDEX from the environment or .env (a number or a name such as 'iphone'), else 0."""
    val = env_get("CAMERA_INDEX")
    if not val:
        return 0
    return int(val) if val.lstrip("-").isdigit() else val


class FrameSource:
    """A camera or a video file. read() -> (ok, frame, timestamp_s, frame_index)."""

    CAMERA_STALL_S = 10.0   # a live camera may drop frames (iPhone Continuity Camera reconnecting): wait this long

    def __init__(self, cap: cv2.VideoCapture, is_camera: bool, fps: float, name: str):
        self.cap, self.is_camera, self.fps, self.name = cap, is_camera, fps, name
        self.index = 0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else 0

    def read(self):
        ok, frame = self.cap.read()
        if (not ok or frame is None) and self.is_camera:
            t_end = time.monotonic() + self.CAMERA_STALL_S
            while (not ok or frame is None) and time.monotonic() < t_end:
                time.sleep(0.02)
                ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None, 0.0, self.index
        i = self.index
        self.index += 1
        ts = time.monotonic() if self.is_camera else i / self.fps
        return True, frame, ts, i

    def release(self):
        self.cap.release()


def configure_camera(cap: cv2.VideoCapture, width: int, height: int, fps: float = 30.0) -> list[str]:
    """Request the resolution, frame rate and best-effort turn off auto exposure / auto white
    balance. Returns a report."""
    report = []
    # Before the resolution: a V4L2 webcam's uncompressed format (YUYV) is often capped far below
    # its compressed one at the same size - on the ROG EYE S, 1920x1080 is 5 fps over YUYV but 30
    # (up to 60) over MJPG, and OpenCV does not pick MJPG on its own. Harmless to ask everywhere:
    # a backend that does not support changing FOURCC (macOS AVFoundation, a video file) just
    # ignores it, same as the exposure/white-balance sets below.
    fourcc_ok = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    got_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    got_fourcc_s = "".join(chr((got_fourcc >> (8 * i)) & 0xFF) for i in range(4)) if got_fourcc else ""
    report.append(f"pixel format MJPG: {'accepted' if fourcc_ok else 'NOT supported by backend'}"
                  f" [read back {got_fourcc_s!r}]")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Asked explicitly rather than left to whatever the driver defaults to for this format/size:
    # without this, re-opening the same camera at the same resolution has negotiated a different
    # frame interval run to run on at least one V4L2 webcam, which is a frame rate that visibly
    # varies session to session, not within one - `FrameSource.fps` is read once at open.
    fps_ok = cap.set(cv2.CAP_PROP_FPS, fps)
    report.append(f"frame rate {fps:g}: {'accepted' if fps_ok else 'NOT supported by backend'}"
                  f" [read back {cap.get(cv2.CAP_PROP_FPS):g}]")
    # Backends disagree on the value meaning "manual": V4L2 uses 0.25 (or 1), others 0. Try them and read back.
    before = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    took = None
    for v in (0.25, 0.0, 1.0):
        if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v):
            took = v
            break
    after = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    report.append(f"auto exposure off: {'accepted (value %s)' % took if took is not None else 'NOT supported by backend'}"
                  f" [read back {before} -> {after}]")
    wb_ok = cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    report.append(f"auto white balance off: {'accepted' if wb_ok else 'NOT supported by backend'}"
                  f" [read back {cap.get(cv2.CAP_PROP_AUTO_WB)}]")
    if took is None or not wb_ok:
        report.append("  -> lock exposure/white balance another way (steady diffuse light; avoid moving shadows), "
                      "or a camera utility app.")
    return report


def apply_v4l2_controls(cam: int | str) -> list[str]:
    """Best-effort, opt-in manual exposure / white balance via v4l2-ctl, right after the camera opens.

    Why this exists and why it is not automatic for everyone: `configure_camera()` above turns
    auto exposure and auto white balance *off* through OpenCV, but several UVC webcams (the ROG
    EYE S used here is one) reset their *absolute* exposure and white-balance-automatic controls
    to the device's power-on defaults every time a process opens them fresh - not just on
    unplug/reboot, the way `data_collection/camera_setup.sh` notes for other cameras. The result
    without this: every new run of a script that opens the camera comes up dark and colour-cast
    (exposure stuck at a near-zero default, white balance off with no real value set), and has to
    be fixed by hand with `v4l2-ctl` each time, from outside the running process.

    `CAMERA_V4L2_FOCUS` and `CAMERA_V4L2_ROI_AUTO` are the same idea for two more things the ROG
    EYE S hunts continuously on its own: `focus_automatic_continuous` re-focusing as anything in
    frame moves (a blur pulse every time it does, which a fast-moving scale triggers constantly),
    and `region_of_interest_auto_ctrls`, a UVC auto-ROI control that shifts what the camera treats
    as the interesting region of the frame and visibly perturbs the picture doing it - this is
    most of what reads as the camera "stabilising" on its own. Setting `CAMERA_V4L2_FOCUS` locks
    focus at that value (`focus_absolute`) and turns continuous autofocus off; `CAMERA_V4L2_ROI_AUTO=0`
    turns the auto-ROI bitmask off. Find a working focus value by trial with `v4l2-ctl -d
    /dev/video<N> --set-ctrl=focus_absolute=<value>` while watching the live picture, scale at its
    working distance, then read `focus_absolute` back as `value=` once it looks sharp.

    So this is per-machine, per-camera config, the same way `CAMERA_INDEX` is: nothing happens
    unless at least one `CAMERA_V4L2_*` variable is set in `.env`. Linux (v4l2) and an integer
    camera index only, since that is what maps to `/dev/video<N>`; a no-op everywhere else, same
    spirit as the rest of this module's macOS/Linux split.
    """
    report: list[str] = []
    if not sys.platform.startswith("linux") or not isinstance(cam, int):
        return report
    exposure = env_get("CAMERA_V4L2_EXPOSURE")
    auto_wb = env_get("CAMERA_V4L2_AUTO_WB")
    focus = env_get("CAMERA_V4L2_FOCUS")
    roi_auto = env_get("CAMERA_V4L2_ROI_AUTO")
    if exposure is None and auto_wb is None and focus is None and roi_auto is None:
        return report
    if shutil.which("v4l2-ctl") is None:
        return [f"v4l2-ctl not installed; CAMERA_V4L2_* settings skipped"]

    device = f"/dev/video{cam}"
    controls = ["-c", "auto_exposure=1"] if exposure is not None else []
    if exposure is not None:
        controls += ["-c", f"exposure_time_absolute={exposure}"]
    if auto_wb is not None:
        controls += ["-c", f"white_balance_automatic={1 if auto_wb not in ('0', '', None) else 0}"]
    if focus is not None:
        controls += ["-c", "focus_automatic_continuous=0", "-c", f"focus_absolute={focus}"]
    if roi_auto is not None:
        controls += ["-c", f"region_of_interest_auto_ctrls={1 if roi_auto not in ('0', '', None) else 0}"]
    try:
        result = subprocess.run(["v4l2-ctl", "-d", device, *controls],
                                capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"v4l2-ctl on {device} failed to run: {exc}"]
    if result.returncode != 0:
        report.append(f"v4l2-ctl on {device} reported an error (control may be unsupported): "
                      f"{result.stderr.strip() or result.returncode}")
    else:
        parts = []
        if exposure is not None:
            parts.append(f"exposure={exposure}")
        if auto_wb is not None:
            parts.append(f"auto_wb={'on' if auto_wb not in ('0', '', None) else 'off'}")
        if focus is not None:
            parts.append(f"focus={focus} (autofocus off)")
        if roi_auto is not None:
            parts.append(f"roi_auto={'on' if roi_auto not in ('0', '', None) else 'off'}")
        report.append(f"{device}: {', '.join(parts)} (from CAMERA_V4L2_* in .env)")
    return report


def open_video(path: str | Path) -> FrameSource:
    path = str(path)
    if not Path(path).is_file():
        raise SystemExit(f"error: video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"error: cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src = FrameSource(cap, False, fps if fps > 0 else 30.0, path)
    print(f"video {path}: {src.width}x{src.height} @ {src.fps:.1f} fps, {src.n_frames} frames")
    return src


def open_camera(camera: int | str | None = None, width: int = 1920, height: int = 1080,
                fps: float = 30.0) -> FrameSource:
    """Open a live camera (number, name fragment such as 'iphone', or URL); waits for a real picture."""
    cam = resolve_camera(camera if camera is not None else default_camera())
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam, backend) if isinstance(cam, int) else cv2.VideoCapture(cam)
    if not cap.isOpened():
        print(f"error: cannot open camera {cam!r}", file=sys.stderr)
        if sys.platform == "darwin":
            print(MAC_CAMERA_HINT, file=sys.stderr)
        raise SystemExit(2)
    for line in configure_camera(cap, width, height, fps):
        print("camera: " + line)
    for line in apply_v4l2_controls(cam):
        print("camera: " + line)
    # The first frames after opening are often empty or black while the sensor starts; an iPhone (Continuity
    # Camera) can take several seconds to connect. Wait up to WARMUP_S for a real (non-black) frame.
    ok, lit, t0, said = False, False, time.monotonic(), False
    while time.monotonic() - t0 < WARMUP_S:
        ok, f = cap.read()
        if ok and f is not None and float(f.mean()) > 3.0:
            lit = True
            break
        if not said and time.monotonic() - t0 > 2.0:
            print(f"camera {cam!r}: waiting for a picture (up to {WARMUP_S:.0f} s)...", flush=True)
            said = True
        time.sleep(0.03)
    if not ok:
        print(f"error: camera {cam!r} opened but delivers no frames", file=sys.stderr)
        if sys.platform == "darwin":
            print(MAC_CAMERA_HINT, file=sys.stderr)
        raise SystemExit(2)
    if not lit:
        print(f"warning: camera {cam!r} sends only black frames.", file=sys.stderr)
        if sys.platform == "darwin":
            print(IPHONE_HINT, file=sys.stderr)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src = FrameSource(cap, True, fps if fps > 0 else 30.0, f"camera {cam}")
    print(f"camera {cam}: requested {width}x{height}, got {src.width}x{src.height} @ {src.fps:.1f} fps")
    return src


def open_source(camera: int | str | None = None, video: str | Path | None = None,
                width: int = 1920, height: int = 1080, fps: float = 30.0) -> FrameSource:
    """A video file if `video` is given, else a live camera."""
    return open_video(video) if video else open_camera(camera, width, height, fps)


def main() -> int:
    cams = list_cameras()
    if not cams:
        print("could not list cameras (needs macOS + ffmpeg); try camera 0, 1, 2 ...")
        return 1
    for i, n in cams:
        print(f"  {i}   {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
