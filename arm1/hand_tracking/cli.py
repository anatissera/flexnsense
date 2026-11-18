"""Camera reader for finger flexion, from either tracker, emitting the same HandFrame.

    python -m hand_tracking run                                   # live window, MediaPipe
    python -m hand_tracking run --csv out.csv                      # + one row per frame
    python -m hand_tracking run --jsonl out.jsonl                  # + one HandFrame per line
    python -m hand_tracking run --headless --video clip.mp4 --csv out.csv

    python -m hand_tracking run --source wilor --jsonl out.jsonl   # MANO rotations, needs a GPU

MediaPipe is the default: CPU, video rate, model auto-downloaded to MODEL_PATH on first run.
WiLoR is the precision path and needs its own checkout, checkpoint and CUDA environment, so it
is opt-in; point `--wilor-root` or WILOR_ROOT at it. See docs/hand-tracking.md.

Both sources produce the same 21 landmarks in the same layout, so the CSV columns, the overlay
and the JSONL mean the same thing whichever one ran.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2

from arm1.hand_tracking.frame import FINGERTIPS, HandFrame
from arm1.hand_tracking.landmarks import FINGERS, all_flexion_deg
from arm1.lib import camera
from arm1.lib.env import env_get

MODEL_PATH = Path("models/hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
CSV_COLUMNS = ["time_s", "frame", "found", *FINGERS.keys()]
WINDOW = "hand tracking"
TIP_COLOR = (77, 193, 255)    # BGR amber, matching the dashboard overlay
JOINT_COLOR = (232, 245, 77)  # BGR cyan


def ensure_model(path: Path = MODEL_PATH) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {MODEL_URL} -> {path} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"  done ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def draw_overlay(image, frames: list[HandFrame]) -> None:
    """The landmarks of every tracked hand, fingertips picked out, straight off the HandFrames."""
    h, w = image.shape[:2]
    if not frames:
        cv2.putText(image, "no hand", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return
    for frame in frames:
        for i, (x, y, _z) in enumerate(frame.landmarks):
            tip = i in FINGERTIPS
            cv2.circle(image, (int(x * w), int(y * h)), 6 if tip else 3,
                       TIP_COLOR if tip else JOINT_COLOR, -1)
    y = 24
    for name, deg in all_flexion_deg(frames[0].world).items():
        cv2.putText(image, f"{name:6s} {deg:5.1f} deg", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y += 22


def make_tracker(args):
    """The chosen backend, plus a callable that turns one image into a list of HandFrames."""
    if args.source == "wilor":
        from arm1.hand_tracking.wilor_source import WilorTracker

        root = args.wilor_root or env_get("WILOR_ROOT", default="WiLoR")
        print(f"loading WiLoR from {root} (this takes a moment) ...")
        tracker = WilorTracker(root, fast=not args.no_fast)
        return tracker, (lambda image, t_ms, ts_ms: tracker.track(image, t_ms)), (lambda: None)

    from arm1.hand_tracking.mediapipe_source import MediaPipeTracker

    tracker = MediaPipeTracker(ensure_model(Path(args.model)), args.max_hands)
    return tracker, tracker.track, tracker.close


def run(args) -> int:
    _tracker, track, close = make_tracker(args)
    src = camera.open_source(camera=args.camera, video=args.video)

    csv_file = jsonl_file = None
    writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(CSV_COLUMNS)
    if args.jsonl:
        jsonl_file = open(args.jsonl, "w")

    last_ts_ms = -1
    t0 = time.monotonic()
    try:
        while True:
            ok, image, ts, idx = src.read()
            if not ok:
                break
            # detect_for_video insists on a strictly increasing timestamp; a video's own can repeat.
            ts_ms = max(int(ts * 1000), last_ts_ms + 1)
            last_ts_ms = ts_ms
            t_ms = (time.monotonic() - t0) * 1000.0

            frames = track(image, t_ms, ts_ms)

            if writer is not None:
                row = [f"{t_ms / 1000:.4f}", idx, int(bool(frames))]
                if frames:
                    flexion = all_flexion_deg(frames[0].world)
                    row += [f"{flexion[name]:.2f}" for name in FINGERS]
                else:
                    row += [""] * len(FINGERS)
                writer.writerow(row)
            if jsonl_file is not None:
                for frame in frames:
                    jsonl_file.write(json.dumps(frame.to_dict()) + "\n")

            if not args.headless:
                draw_overlay(image, frames)
                cv2.imshow(WINDOW, image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        src.release()
        close()
        if csv_file is not None:
            csv_file.close()
        if jsonl_file is not None:
            jsonl_file.close()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run"])
    p.add_argument("--source", choices=["mediapipe", "wilor"], default="mediapipe",
                   help="which tracker to run (default: mediapipe, CPU; wilor needs a CUDA GPU)")
    p.add_argument("--camera", help="camera number or name fragment, e.g. iphone (default: CAMERA_INDEX from .env, else 0)")
    p.add_argument("--video", help="read from a video file instead of a live camera")
    p.add_argument("--csv", help="log one row per frame (time_s, frame, found, one flexion column per finger)")
    p.add_argument("--jsonl", help="log one HandFrame as JSON per line, per hand found")
    p.add_argument("--headless", action="store_true", help="no window (needed for a video file with no display)")
    p.add_argument("--max-hands", type=int, default=1, help="mediapipe only; wilor reports every hand it detects")
    p.add_argument("--model", default=str(MODEL_PATH), help="path to hand_landmarker.task (downloaded here if missing)")
    p.add_argument("--wilor-root", help="path to a WiLoR checkout (default: WILOR_ROOT from .env, else ./WiLoR)")
    p.add_argument("--no-fast", action="store_true", help="wilor only: skip fp16 and layer dropping")
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
