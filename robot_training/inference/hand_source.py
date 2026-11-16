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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))        # robot_training/ -> hand_features
from hand_features import TIP_LANDMARKS, HandTips  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]                        # raíz del repo -> hand_tracking/


class WilorTips:
    """WiLoR a través de hand_tracking.wilor_source.WilorTracker (el mismo tracker del repo).

    Necesita un checkout de WiLoR con sus pesos (WILOR_ROOT) y GPU; ~90 ms por imagen en una
    RTX 3050 y ~1.7 GB de VRAM. Solo se llama cuando se pide un chunk, no en cada tick.
    Si hay varias manos se queda con la primera derecha (la del dataset es la derecha).
    """

    def __init__(self, wilor_root: str | None = None, fast: bool = True):
        wilor_root = wilor_root or os.environ.get("WILOR_ROOT")
        if not wilor_root or not Path(wilor_root).is_dir():
            raise SystemExit("WiLoR: pasa --wilor-root <checkout de WiLoR> o define WILOR_ROOT "
                             "(o usa --hand-source mediapipe si el dataset se convirtió con --hand mediapipe)")
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from hand_tracking.wilor_source import WilorTracker
        self._tracker = WilorTracker(wilor_root, fast=fast)

    def __call__(self, rgb: np.ndarray) -> np.ndarray | None:
        frames = self._tracker.track(np.ascontiguousarray(rgb[..., ::-1]), 0.0)     # WiLoR espera BGR
        if not frames:
            return None
        hand = next((f for f in frames if f.handedness == "right"), frames[0])
        lm = hand.landmarks
        return np.array([c for i in TIP_LANDMARKS for c in (lm[i][0], lm[i][1])], dtype=np.float32)   # 5 yemas


def make_hand_source(kind: str, wilor_root: str | None = None):
    if kind == "wilor":
        return WilorTips(wilor_root)
    if kind == "mediapipe":
        return HandTips()
    raise ValueError(f"fuente de mano desconocida: {kind}")
