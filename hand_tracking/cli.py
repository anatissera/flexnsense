"""Camera reader for finger flexion: MediaPipe Hand Landmarker -> one angle per finger.

    python -m hand_tracking run                               # live window
    python -m hand_tracking run --csv out.csv                  # + CSV log
    python -m hand_tracking run --headless --video clip.mp4 --csv out.csv

The model (hand_landmarker.task, ~a few MB) is downloaded once to MODEL_PATH on first run.
Angles come from `hand_world_landmarks` (metres, hand-centred), not the pixel landmarks used
only to draw the overlay - see hand_tracking/landmarks.py for why.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from hand_tracking.landmarks import FINGERS, all_flexion_deg
from lib import camera

MODEL_PATH = Path("models/hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
CSV_COLUMNS = ["time_s", "frame", "found", *FINGERS.keys()]
WINDOW = "hand tracking"


def ensure_model(path: Path = MODEL_PATH) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {MODEL_URL} -> {path} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"  done ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def make_landmarker(model_path: Path, max_hands: int):
    # Imported here, not at module load: mediapipe is a heavy, optional dependency and importing
    # it should not be the reason `from hand_tracking.landmarks import ...` fails in a pure test.
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max_hands,
    )
    return mp, vision.HandLandmarker.create_from_options(options)


def process_frame(mp, landmarker, frame_bgr: np.ndarray, ts_ms: int):
    """-> (found: bool, flexion: dict[str, float] | None, image_landmarks | None), for the first detected hand."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, ts_ms)
    if not result.hand_world_landmarks:
        return False, None, None
    world = [(p.x, p.y, p.z) for p in result.hand_world_landmarks[0]]
    image = result.hand_landmarks[0] if result.hand_landmarks else None
    return True, all_flexion_deg(world), image


def draw_overlay(frame: np.ndarray, found: bool, flexion: dict[str, float] | None, image_landmarks) -> None:
    h, w = frame.shape[:2]
    if image_landmarks is not None:
        for lm in image_landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
    y = 24
    if not found:
        cv2.putText(frame, "no hand", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return
    for name, deg in flexion.items():
        cv2.putText(frame, f"{name:6s} {deg:5.1f} deg", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y += 22


def run(args) -> int:
    model_path = ensure_model(Path(args.model))
    mp, landmarker = make_landmarker(model_path, args.max_hands)
    src = camera.open_source(camera=args.camera, video=args.video)

    writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(CSV_COLUMNS)

    last_ts_ms = -1
    t0 = time.monotonic()
    try:
        while True:
            ok, frame, ts, idx = src.read()
            if not ok:
                break
            ts_ms = max(int(ts * 1000), last_ts_ms + 1)
            last_ts_ms = ts_ms

            found, flexion, image_landmarks = process_frame(mp, landmarker, frame, ts_ms)

            if writer is not None:
                row = [f"{time.monotonic() - t0:.4f}", idx, int(found)]
                row += [f"{flexion[name]:.2f}" for name in FINGERS] if found else [""] * len(FINGERS)
                writer.writerow(row)

            if not args.headless:
                draw_overlay(frame, found, flexion, image_landmarks)
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        src.release()
        landmarker.close()
        if csv_file is not None:
            csv_file.close()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run"])
    p.add_argument("--camera", help="camera number or name fragment, e.g. iphone (default: CAMERA_INDEX from .env, else 0)")
    p.add_argument("--video", help="read from a video file instead of a live camera")
    p.add_argument("--csv", help="log one row per frame (time_s, frame, found, one flexion column per finger)")
    p.add_argument("--headless", action="store_true", help="no window (needed for a video file with no display)")
    p.add_argument("--max-hands", type=int, default=1)
    p.add_argument("--model", default=str(MODEL_PATH), help=f"path to hand_landmarker.task (downloaded here if missing)")
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
