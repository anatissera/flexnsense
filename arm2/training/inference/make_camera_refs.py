#!/usr/bin/env python
"""Genera camera_refs.npz: banco de ejemplos por vista para reconocer las cámaras en inferencia.

    python make_camera_refs.py --src /ruta/al/dataset_normacore [--ref-views side,gripper,table]

Station no etiqueta las imágenes de inference/normvla y su orden cambia entre arranques, así
que `CameraMapper` (station_io.py) reconoce cada cámara comparándola con este banco.
Usa el MISMO episodio de referencia y los MISMOS nombres de vista que convert_normacore.py:
cada slot de cada episodio se asigna a una vista igual que en la conversión, y de cada vista
se guardan miniaturas 32x32 en gris normalizadas (1 de cada --every frames).

Hay que regenerarlo si cambias las cámaras de sitio o grabas un dataset con otra escena.
El fichero contiene miniaturas derivadas de tus imágenes: no lo publiques si el dataset es privado.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convert_normacore import DEFAULT_VIEWS, _feat, build_reference, classify_slots, list_episodes  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", required=True)
ap.add_argument("--ref-episode", type=int, default=0)
ap.add_argument("--ref-views", default=",".join(DEFAULT_VIEWS))
ap.add_argument("--every", type=int, default=8)
ap.add_argument("--out", default=str(Path(__file__).with_name("camera_refs.npz")))
a = ap.parse_args()

views = a.ref_views.split(","); files = list_episodes(a.src)
ref = build_reference(pq.read_table(files[a.ref_episode], columns=["images"]).to_pydict()["images"])
bank = {v: [] for v in views}; orders = {}
for f in files:
    im = pq.read_table(f, columns=["images"]).to_pydict()["images"]
    perm, _ = classify_slots(im, ref)
    orders[tuple(views[v] for v in perm)] = orders.get(tuple(views[v] for v in perm), 0) + 1
    for s, v in enumerate(perm):
        bank[views[v]] += [_feat(im[i][s]["jpeg"]) for i in range(0, len(im), a.every)]
np.savez_compressed(a.out, **{v: np.stack(b).astype(np.float16) for v, b in bank.items()})
print("guardado", a.out, {v: len(b) for v, b in bank.items()})
print("órdenes de images[] vistos en el dataset:", {" , ".join(k): n for k, n in orders.items()})
