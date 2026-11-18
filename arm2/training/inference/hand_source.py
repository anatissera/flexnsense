"""De dónde salen las yemas de los dedos en inferencia (ver ../hand_features.py).

    make_hand_source("wilor", wilor_root=...)   el modelo con el que se anotó el dataset (columnas hand_*)
    make_hand_source("mediapipe")               MediaPipe en CPU, para datasets convertidos con --hand mediapipe

Las dos devuelven una función  rgb (H, W, 3) uint8 -> [index_x, index_y, middle_x, middle_y] | None.
Hay que usar la misma fuente con la que se generaron los datos de entrenamiento: WiLoR y MediaPipe
difieren ~0.01-0.02 (2-4 px de 224) en las mismas imágenes.

Se ejecuta en la máquina del robot (control.py), no en el servidor de la policy: en modo remoto
las 5 cifras viajan dentro del estado.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))        # arm2/training/ -> hand_features
from hand_features import TIP_LANDMARKS, HandTips  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]                        # raíz del repo -> hand_tracking/


class WilorTips:
    """WiLoR a través de hand_tracking.wilor_source.WilorTracker (el mismo tracker del repo).

    Necesita un checkout de WiLoR con sus pesos (WILOR_ROOT) y GPU; ~90 ms por imagen en una
    RTX 3050 y ~1.7 GB de VRAM. Solo se llama cuando se pide un chunk, no en cada tick.
    Misma configuración que el script que anotó el dataset (arm2/data_collection/hand_keypoints.py):
    modo rápido fp16, conf 0.3, rescale_factor 2.0, sobre el JPEG 224x224 de Station. Si hay
    varias manos se queda con la más cercana a la elegida en la llamada anterior y, en la
    primera, con la más grande.
    """

    def __init__(self, wilor_root: str | None = None, fast: bool = True):
        wilor_root = wilor_root or os.environ.get("WILOR_ROOT")
        if not wilor_root or not Path(wilor_root).is_dir():
            raise SystemExit("WiLoR: pasa --wilor-root <checkout de WiLoR> o define WILOR_ROOT "
                             "(o usa --hand-source mediapipe si el dataset se convirtió con --hand mediapipe)")
        wilor_root = str(Path(wilor_root).resolve())
        # chumpy (lo necesita MANO_RIGHT.pkl) es anterior a numpy 1.24 y a Python 3.11.
        import inspect
        for name, value in dict(bool=bool, int=int, float=float, complex=complex, object=object,
                                unicode=str, str=str).items():
            if name not in np.__dict__:
                setattr(np, name, value)
        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        for path in (wilor_root, str(REPO_ROOT)):
            if path not in sys.path:
                sys.path.insert(0, path)
        # WiLoR crea un renderer OpenGL en su constructor. La inferencia no lo usa y falla en
        # máquinas sin EGL (eglInitialize, error 12289), así que se sustituye por uno vacío.
        import wilor.models.wilor as wilor_module
        wilor_module.MeshRenderer = lambda *a, **k: None
        from arm1.hand_tracking.wilor_source import WilorTracker
        self._tracker = WilorTracker(wilor_root, fast=fast)
        self._last_center = None

    def __call__(self, rgb: np.ndarray) -> np.ndarray | None:
        frames = self._tracker.track(np.ascontiguousarray(rgb[..., ::-1]), 0.0)     # WiLoR espera BGR
        if not frames:
            return None
        points = [np.asarray(f.landmarks, dtype=np.float32)[:, :2] for f in frames]
        if self._last_center is not None:
            i = int(np.argmin([np.linalg.norm(p.mean(0) - self._last_center) for p in points]))
        else:
            i = int(np.argmax([np.ptp(p, axis=0).prod() for p in points]))
        self._last_center = points[i].mean(0)
        lm = frames[i].landmarks
        return np.array([c for i in TIP_LANDMARKS for c in (lm[i][0], lm[i][1])], dtype=np.float32)   # 5 yemas


def make_hand_source(kind: str, wilor_root: str | None = None):
    if kind == "wilor":
        return WilorTips(wilor_root)
    if kind == "mediapipe":
        return HandTips()
    raise ValueError(f"fuente de mano desconocida: {kind}")
