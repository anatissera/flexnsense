"""Hand keypoints for the dataset: WiLoR on the overhead camera, one result per frame.

Used by `record_dataset.py` (live, while recording) and `add_hand_keypoints.py` (episodes that
are already on the Hub). Both go through the same code and the same model settings, so old and
new episodes carry comparable values.

The tracker itself is `hand_tracking.wilor_source.WilorTracker`; this module only adds what the
dataset needs: decoding the frame, choosing one hand, filling frames with no detection, and the
Parquet columns.

Columns added to every episode (one value per frame):

    hand_camera         uint8          index of the overhead camera in `images` (it is not the same in
                                       every recording session: Station orders cameras as it finds them)
    hand_detected       bool           WiLoR found a hand in this frame
    hand_is_right       bool           handedness reported by WiLoR's detector
    hand_keypoints_2d   21 x 2 float32 x, y as fractions of the overhead image (0..1, origin top-left)
    hand_fingertips_2d  5 x 2 float32  thumb, index, middle, ring, pinky: rows 4, 8, 12, 16, 20 of the above
    hand_keypoints_3d   21 x 3 float32 metres, hand centred, camera axes (hand shape, not hand position)

The 21 points use the MediaPipe / OpenPose layout (see hand_tracking/pose.py).

Frames with no detection repeat the nearest detection of the same episode (previous one first,
else the next one) with `hand_detected = False`, so the values never jump to zero in the middle
of an episode. An episode with no detection at all gets zeros.
"""
from __future__ import annotations

import inspect
import io
import os
import site
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WILOR_ROOT = REPO_ROOT / "WiLoR"
FINGERTIPS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky

HAND_FIELDS = [
    pa.field("hand_camera", pa.uint8()),
    pa.field("hand_detected", pa.bool_()),
    pa.field("hand_is_right", pa.bool_()),
    pa.field("hand_keypoints_2d", pa.list_(pa.float32(), 42)),
    pa.field("hand_fingertips_2d", pa.list_(pa.float32(), 10)),
    pa.field("hand_keypoints_3d", pa.list_(pa.float32(), 63)),
]
HAND_COLUMNS = [f.name for f in HAND_FIELDS]


def rerun_without_user_site() -> None:
    """Restart the interpreter with `-s` if ~/.local site-packages is active.

    WiLoR imports pytorch-lightning, which imports torchmetrics, which imports `transformers` if
    it finds one. A stale `transformers` in ~/.local then breaks the import. Everything this
    pipeline needs is installed in the environment itself, so ~/.local is simply left out.
    """
    if site.ENABLE_USER_SITE and not sys.flags.no_user_site:
        os.execv(sys.executable, [sys.executable, "-s", *sys.argv])


def find_overhead_camera(jpegs: list[bytes]) -> int:
    """Which of the images of one frame is the overhead camera.

    Specific to our setup: the overhead camera looks down through an opening, so its left and
    right edges are black (mean grey level about 2 of 255, against 38 or more for the other two
    cameras in all 43 episodes checked). Raises if no camera stands out that clearly; pass the
    index by hand in that case.
    """
    from PIL import Image

    def edge_level(jpeg: bytes) -> float:
        grey = np.asarray(Image.open(io.BytesIO(jpeg)).convert("L"), dtype=np.float32)
        k = max(1, grey.shape[1] // 12)
        return float((grey[:, :k].mean() + grey[:, -k:].mean()) / 2)

    levels = [edge_level(j) for j in jpegs]
    order = np.argsort(levels)
    if levels[order[0]] > 10 or (len(levels) > 1 and levels[order[1]] < 20):
        raise RuntimeError(f"Cannot tell which camera is the overhead one (edge levels {np.round(levels, 1)}); "
                           "pass --hand-camera <index>")
    return int(order[0])


class HandKeypointTracker:
    """WiLoR on single frames. Needs a CUDA GPU and a WiLoR checkout (see README)."""

    def __init__(self, wilor_root: str | Path | None = None, fast: bool = True):
        wilor_root = str(Path(wilor_root or os.environ.get("WILOR_ROOT") or DEFAULT_WILOR_ROOT).resolve())
        if not Path(wilor_root, "pretrained_models", "wilor_final.ckpt").exists():
            raise FileNotFoundError(f"No WiLoR checkpoint under {wilor_root}/pretrained_models (see README)")

        # chumpy (needed to unpickle MANO_RIGHT.pkl) predates numpy 1.24 and Python 3.11.
        for name, value in dict(bool=bool, int=int, float=float, complex=complex, object=object,
                                unicode=str, str=str).items():
            if name not in np.__dict__:
                setattr(np, name, value)
        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        for p in (wilor_root, str(REPO_ROOT)):
            if p not in sys.path:
                sys.path.insert(0, p)
        # WiLoR builds an offscreen OpenGL renderer in its constructor. Inference does not use
        # it, and it fails on machines with no working EGL, so it is replaced by a no-op.
        import wilor.models.wilor as wilor_module

        wilor_module.MeshRenderer = lambda *a, **k: None

        import cv2
        import torch
        from arm1.hand_tracking.wilor_source import WilorTracker

        if not torch.cuda.is_available():
            raise RuntimeError("WiLoR needs a CUDA GPU (torch.cuda.is_available() is False)")
        self._cv2 = cv2
        self._tracker = WilorTracker(wilor_root, fast=fast)
        self._last_center: np.ndarray | None = None

    def reset(self) -> None:
        """Call at the start of an episode: forget which hand was being followed."""
        self._last_center = None

    def detect(self, jpeg: bytes) -> dict | None:
        """One JPEG frame in; the keypoints of one hand out, or None if no hand was found.

        With several hands in view, the one closest to the hand chosen in the previous frame is
        kept; on the first frame, the largest one.
        """
        image = self._cv2.imdecode(np.frombuffer(jpeg, np.uint8), self._cv2.IMREAD_COLOR)
        frames = self._tracker.track(image, 0.0)
        if not frames:
            return None
        points = [np.asarray(f.landmarks, dtype=np.float32)[:, :2] for f in frames]
        if self._last_center is not None:
            i = int(np.argmin([np.linalg.norm(p.mean(0) - self._last_center) for p in points]))
        else:
            i = int(np.argmax([np.ptp(p, axis=0).prod() for p in points]))
        self._last_center = points[i].mean(0)
        return {
            "is_right": frames[i].handedness == "right",
            "kp2d": points[i],
            "kp3d": np.asarray(frames[i].world, dtype=np.float32),
        }


def hand_columns(detections: list[dict | None], hand_camera: int) -> dict[str, list]:
    """Per-frame detections of one episode to the dataset columns, gaps filled (see module doc)."""
    n = len(detections)
    filled: list[dict | None] = list(detections)
    for i in range(1, n):
        if filled[i] is None:
            filled[i] = filled[i - 1]
    for i in range(n - 2, -1, -1):
        if filled[i] is None:
            filled[i] = filled[i + 1]
    empty = {"is_right": False, "kp2d": np.zeros((21, 2), np.float32), "kp3d": np.zeros((21, 3), np.float32)}
    filled = [d or empty for d in filled]
    return {
        "hand_camera": [hand_camera] * n,
        "hand_detected": [d is not None for d in detections],
        "hand_is_right": [bool(d["is_right"]) for d in filled],
        "hand_keypoints_2d": [d["kp2d"].reshape(-1).tolist() for d in filled],
        "hand_fingertips_2d": [d["kp2d"][list(FINGERTIPS)].reshape(-1).tolist() for d in filled],
        "hand_keypoints_3d": [d["kp3d"].reshape(-1).tolist() for d in filled],
    }


def with_hand_columns(table: pa.Table, detections: list[dict | None], hand_camera: int) -> pa.Table:
    """`table` plus the hand columns (replacing them if they are already there)."""
    columns = hand_columns(detections, hand_camera)
    table = table.drop_columns([c for c in HAND_COLUMNS if c in table.column_names])
    for field in HAND_FIELDS:
        table = table.append_column(field, pa.array(columns[field.name], type=field.type))
    return table
