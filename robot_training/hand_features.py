"""Yemas de los dedos de la mano del paciente como entrada extra de la policy.

La tarea es llevar la anilla de un dedo a otro: dónde están el índice (origen) y el corazón
(destino) en la imagen es justo lo que la policy tiene que averiguar. Aquí se le da ya
calculado, como 5 números añadidos a `observation.state`:

    hand.index_x, hand.index_y, hand.middle_x, hand.middle_y, hand.detected

x, y = posición de la yema en la imagen de la cámara `side`, como fracción del ancho y del
alto (0..1). `hand.detected` = 1 si hay mano en ese frame.

El MISMO código se usa al convertir el dataset (convert_normacore.py --hand ...) y en inferencia
(inference/policy_runner.py), sobre el mismo JPEG 224x224 que ve la policy. Así los números de
entrenamiento y los de inferencia salen del mismo sitio.

Detector: MediaPipe Hand Landmarker en modo IMAGE (sin memoria entre frames: el resultado de
un frame no depende de los anteriores, y por eso se puede reproducir exactamente). Corre en CPU.

Frames sin mano: se repite la última detección del episodio con detected = 0; si aún no ha
habido ninguna, (0.5, 0.5, 0.5, 0.5, 0). Es causal: solo mira hacia atrás, igual en ambos lados.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

HAND_NAMES = ["hand.index_x", "hand.index_y", "hand.middle_x", "hand.middle_y", "hand.detected"]
HAND_DIM = len(HAND_NAMES)
HAND_VIEW = "side"                 # cámara de la que salen las coordenadas
INDEX_TIP, MIDDLE_TIP = 8, 12      # índices de MediaPipe (21 puntos); en hand_fingertips_2d son el 1 y el 2
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
             "float16/latest/hand_landmarker.task")


def model_path() -> Path:
    """hand_landmarker.task: $HAND_LANDMARKER_TASK, o se descarga una vez a la caché (8 MB)."""
    if os.environ.get("HAND_LANDMARKER_TASK"):
        return Path(os.environ["HAND_LANDMARKER_TASK"])
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache")) / "mediapipe" / "hand_landmarker.task"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"descargando {MODEL_URL} -> {cache}")
        tmp = cache.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(cache)
    return cache


class HandTips:
    """rgb (H, W, 3) uint8 -> [index_x, index_y, middle_x, middle_y] o None si no hay mano."""

    def __init__(self, model: str | Path | None = None):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError as e:
            raise ImportError("hace falta mediapipe para las yemas de los dedos: uv pip install mediapipe") from e
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model or model_path())),
            running_mode=vision.RunningMode.IMAGE, num_hands=1))

    def __call__(self, rgb: np.ndarray) -> np.ndarray | None:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb, dtype=np.uint8))
        result = self._landmarker.detect(image)
        if not result.hand_landmarks:
            return None
        lm = result.hand_landmarks[0]
        return np.array([lm[INDEX_TIP].x, lm[INDEX_TIP].y, lm[MIDDLE_TIP].x, lm[MIDDLE_TIP].y], dtype=np.float32)

    def close(self) -> None:
        self._landmarker.close()


class HandFill:
    """Convierte detecciones (o None) en el vector de 5 números, rellenando los frames sin mano."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._last = np.full(4, 0.5, dtype=np.float32)

    def step(self, tips: np.ndarray | None) -> np.ndarray:
        if tips is not None:
            self._last = np.asarray(tips, dtype=np.float32)
        return np.append(self._last, 0.0 if tips is None else 1.0).astype(np.float32)
