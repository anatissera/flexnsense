#!/usr/bin/env python
"""Comprueba que la fuente de yemas de los dedos de inferencia da lo mismo que el dataset.

    python check_hand.py --parquet <episode_XXXX.parquet> --hand-source wilor --wilor-root ../../../WiLoR

Pasa cada frame del episodio por la fuente (la misma que usará control.py) y lo compara con las
columnas hand_fingertips_2d / hand_detected grabadas. Si la fuente es la del dataset, la
diferencia mediana debe ser ~0 (< 0.005). Si sale ~0.01-0.02 es otro modelo (p.ej. MediaPipe
contra un dataset anotado con WiLoR) o el mismo con otra configuración: la policy recibiría en
inferencia números distintos de los del entrenamiento.
"""

import argparse

import numpy as np
import pyarrow.parquet as pq

from hand_source import make_hand_source
from station_io import decode_rgb


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True)
    p.add_argument("--hand-source", choices=["wilor", "mediapipe"], default="wilor")
    p.add_argument("--wilor-root", default=None)
    p.add_argument("--step", type=int, default=2, help="mirar 1 de cada N frames")
    a = p.parse_args()

    t = pq.read_table(a.parquet, columns=["images", "hand_camera", "hand_detected", "hand_fingertips_2d"]).to_pydict()
    source = make_hand_source(a.hand_source, a.wilor_root)
    err, n, found, recorded = [], 0, 0, 0
    for i in range(0, len(t["images"]), a.step):
        tips = source(decode_rgb(t["images"][i][t["hand_camera"][i]]["jpeg"]))
        n += 1
        found += tips is not None
        recorded += bool(t["hand_detected"][i])
        if tips is not None and t["hand_detected"][i]:
            err.append(np.abs(tips - np.array(t["hand_fingertips_2d"][i], np.float32)))
    print(f"{n} frames | mano detectada: {a.hand_source} {found}/{n}, dataset {recorded}/{n}")
    if not err:
        raise SystemExit("ningún frame con mano en ambos: no se puede comparar")
    err = np.array(err)
    med = np.median(err, 0)
    print("diferencia |fuente - dataset| (x, y de pulgar, índice, corazón, anular, meñique), en fracción de imagen:")
    print("  mediana", med.round(4).tolist(), "| p95", np.percentile(err, 95, axis=0).round(4).tolist())
    print("OK: es la misma fuente que el dataset" if med.max() < 0.005 else
          "AVISO: NO coincide con el dataset (otra fuente u otra configuración)")


if __name__ == "__main__":
    main()
