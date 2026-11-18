"""Yemas de los dedos de la mano del paciente como entrada extra de la policy.

La tarea es llevar la anilla de un dedo a otro: dónde están el índice (origen) y el corazón
(destino) en la imagen es justo lo que la policy tiene que averiguar. Aquí se le da ya
calculado, como 5 números añadidos a `observation.state`:

    hand.source_x, hand.source_y, hand.target_x, hand.target_y, hand.detected

source = yema del dedo donde está la anilla, target = yema del dedo al que va. Salen de la
frase de la tarea ("move device from right index to right middle finger" -> índice, corazón;
"... from right middle to right ring finger" -> corazón, anular), ver task_fingers(). Así
todas las tareas dedo -> dedo son la misma para la policy y un solo modelo sirve para todas.
Sin frase (o si no nombra dos dedos) se usa índice -> corazón, que es lo que hacían los
modelos anteriores (sus dos primeras parejas de números eran índice y corazón).

x, y = posición de la yema en la imagen de la cámara `side`, como fracción del ancho y del
alto (0..1). `hand.detected` = 1 si hay mano en ese frame.

El MISMO código se usa al convertir el dataset (convert_normacore.py --hand ...) y en inferencia
(inference/policy_runner.py), sobre el mismo JPEG 224x224 que ve la policy. Así los números de
entrenamiento y los de inferencia salen del mismo sitio.

Detector: MediaPipe Hand Landmarker en modo IMAGE (sin memoria entre frames: el resultado de
un frame no depende de los anteriores, y por eso se puede reproducir exactamente). Corre en CPU.

Marcas en la imagen (opcional, convert_normacore.py --hand-marks): además de los 5 números, se
dibujan dos anillos sobre la imagen `side`, verde en la yema origen y magenta en la destino
(draw_marks). Una red de visión aprovecha mejor una marca en la imagen que unas coordenadas.
El checkpoint lo recuerda con un fichero hand.json ({"marks": true}) y la inferencia las dibuja
igual. Las marcas usan los mismos valores rellenados que el estado; antes de la primera
detección del episodio no se dibuja nada.

Frames sin mano: se repite la última detección del episodio con detected = 0; si aún no ha
habido ninguna, (0.5, 0.5, 0.5, 0.5, 0). Es causal: solo mira hacia atrás, igual en ambos lados.
"""

from __future__ import annotations

import io
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np

HAND_NAMES = ["hand.source_x", "hand.source_y", "hand.target_x", "hand.target_y", "hand.detected"]
HAND_DIM = len(HAND_NAMES)
HAND_VIEW = "side"                 # cámara de la que salen las coordenadas
TIP_LANDMARKS = (4, 8, 12, 16, 20)  # yemas en los 21 puntos de MediaPipe / WiLoR: pulgar ... meñique
FINGERS = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4, "little": 4}   # orden de hand_fingertips_2d
DEFAULT_FINGERS = (FINGERS["index"], FINGERS["middle"])
_TASK_RE = re.compile(r"from\s+(?:(?:right|left)\s+)?(\w+)\s+to\s+(?:(?:right|left)\s+)?(\w+)")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
             "float16/latest/hand_landmarker.task")


def task_fingers(task: str | None) -> tuple[int, int]:
    """(dedo origen, dedo destino) como índices 0..4 de hand_fingertips_2d, según la frase de la tarea."""
    m = _TASK_RE.search((task or "").lower())
    if m and m.group(1) in FINGERS and m.group(2) in FINGERS:
        return FINGERS[m.group(1)], FINGERS[m.group(2)]
    return DEFAULT_FINGERS


def pick_tips(tips10, task: str | None) -> np.ndarray | None:
    """Las 5 yemas (10 números: pulgar x, y ... meñique x, y) -> [source_x, source_y, target_x, target_y]."""
    if tips10 is None:
        return None
    t = np.asarray(tips10, dtype=np.float32).reshape(5, 2)
    src, dst = task_fingers(task)
    return np.concatenate([t[src], t[dst]])


MARKS_FILE = "hand.json"                       # dentro de pretrained_model/: {"marks": true}
MARK_COLORS = ((0, 255, 0), (255, 0, 255))     # origen verde, destino magenta


def draw_marks(rgb: np.ndarray, vec) -> np.ndarray:
    """Copia de rgb (H, W, 3) con un anillo en (source_x, source_y) y otro en (target_x, target_y)."""
    out = np.array(rgb, dtype=np.uint8, copy=True)
    h, w = out.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = 0.03 * w                                # ~7 px en 224: rodea la yema sin taparla
    for (x, y), color in zip(np.asarray(vec, dtype=np.float32)[:4].reshape(2, 2), MARK_COLORS):
        d = np.hypot(xx - x * w, yy - y * h)
        out[np.abs(d - r) <= 2.0] = (0, 0, 0)   # borde negro para que se vea sobre cualquier fondo
        out[np.abs(d - r) <= 1.0] = color
    return out


def mark_jpeg(jpeg: bytes, vec) -> bytes:
    """Lo mismo sobre un JPEG (inferencia: la policy recibe JPEG de Station)."""
    from PIL import Image
    rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
    buf = io.BytesIO()
    Image.fromarray(draw_marks(rgb, vec)).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def checkpoint_uses_marks(checkpoint: str | Path) -> bool:
    f = Path(checkpoint) / MARKS_FILE
    return bool(f.exists() and json.loads(f.read_text()).get("marks"))


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
    """rgb (H, W, 3) uint8 -> las 5 yemas (10 números, como hand_fingertips_2d) o None si no hay mano."""

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
        return np.array([c for i in TIP_LANDMARKS for c in (lm[i].x, lm[i].y)], dtype=np.float32)

    def close(self) -> None:
        self._landmarker.close()


class HandFill:
    """Convierte detecciones (o None) en el vector de 5 números, rellenando los frames sin mano."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._last = np.full(4, 0.5, dtype=np.float32)
        self.seen = False          # ¿ha habido ya alguna detección en este episodio?

    def step(self, tips: np.ndarray | None) -> np.ndarray:
        if tips is not None:
            self.seen = True
            self._last = np.asarray(tips, dtype=np.float32)
        return np.append(self._last, 0.0 if tips is None else 1.0).astype(np.float32)
