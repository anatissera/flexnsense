"""MediaPipe Hand Landmarker as a HandFrame source, the counterpart to wilor_source.

This is the default and the one to reach for first: it runs on CPU at video rate, needs no GPU
and no checkpoint. What it cannot give is ``HandFrame.mano``, because it estimates landmark
positions rather than fitting a hand model, so that field stays unset for this source.

mediapipe is imported lazily inside :class:`MediaPipeTracker`, so the module stays importable
without it (the repo's own venv does not always have it).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hand_tracking.frame import HandFrame
from hand_tracking.pose import Point, landmarks_to_pose, tip_wrist_mm


def to_points(landmarks: Any) -> list[Point]:
    """MediaPipe's landmark objects to plain (x, y, z) tuples."""
    return [(float(p.x), float(p.y), float(p.z)) for p in landmarks]


def handedness_of(categories: Any) -> str | None:
    """MediaPipe's handedness categories to "left", "right", or None when it does not say."""
    if not categories:
        return None
    name = getattr(categories[0], "category_name", None)
    name = name.lower() if isinstance(name, str) else None
    return name if name in ("left", "right") else None


def build_frame(t_ms: float, landmarks: Any, world: Any, categories: Any) -> HandFrame:
    """One MediaPipe detection to a HandFrame. Angles come from the metric world landmarks."""
    world_points = to_points(world)
    return HandFrame(
        source="mediapipe",
        t_ms=t_ms,
        handedness=handedness_of(categories),
        landmarks=to_points(landmarks),
        world=world_points,
        angles=landmarks_to_pose(world_points),
        tip_wrist_mm=tip_wrist_mm(world_points),
    )


class MediaPipeTracker:
    """Runs the Hand Landmarker over frames and yields HandFrames."""

    def __init__(self, model_path: Path, max_hands: int = 1) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
            )
        )

    def track(self, frame_bgr: Any, t_ms: float, ts_ms: int) -> list[HandFrame]:
        """One frame in, one HandFrame per hand found out.

        ``ts_ms`` is the monotonically increasing timestamp the VIDEO running mode requires, which
        is not the same thing as ``t_ms``, the time that goes into the frame.
        """
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.hand_world_landmarks:
            return []
        handedness = getattr(result, "handedness", None) or []
        return [
            build_frame(t_ms, result.hand_landmarks[i], world, handedness[i] if i < len(handedness) else None)
            for i, world in enumerate(result.hand_world_landmarks)
        ]

    def close(self) -> None:
        self._landmarker.close()
