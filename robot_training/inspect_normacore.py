#!/usr/bin/env python
"""Inspecciona un dataset Parquet de NormaCore ANTES de convertirlo.

    python inspect_normacore.py --src /ruta/al/dataset_normacore [--sheet camaras.jpg]

Muestra: episodios, frames, fps real, huecos de grabación, nº de joints/imágenes, tasks y
calibración; y guarda una lámina con un frame de cada images[i] en varios episodios, para
decidir a ojo --ref-views (qué vista es images[0], images[1], images[2] en el episodio 1)
y detectar si el orden de las cámaras cambia entre episodios.
"""
import argparse, glob, io, os
import numpy as np, pyarrow.parquet as pq
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", required=True)
ap.add_argument("--sheet", default="camera_sheet.jpg")
ap.add_argument("--sheet-episodes", type=int, default=6, help="nº de episodios (repartidos) en la lámina")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.src, "data", "*", "*.parquet")))
assert files, f"no hay parquet en {a.src}/data/*/"
print(pq.read_schema(files[0]), "\n")
tot, dts, gaps, tasks, ranges, shapes, bad = 0, [], [], set(), set(), set(), []
for f in files:
    t = pq.read_table(f, columns=["timestamp_ns_since_episode_start", "joints", "task"]).to_pydict()
    ts = np.array(t["timestamp_ns_since_episode_start"], dtype=np.int64) / 1e9
    dt = np.diff(ts); dts += dt.tolist(); tot += len(ts)
    med = float(np.median(dt)); g = dt[dt > 1.5 * med]
    if len(g): gaps.append((os.path.basename(f), len(g), round(float(g.sum()), 1)))
    tasks |= set(t["task"])
    nj = {len(j) for j in t["joints"]}; shapes |= nj
    for j in t["joints"][:1]: ranges.add(tuple((x["range_min"], x["range_max"]) for x in j))
dts = np.array(dts); med = float(np.median(dts))
print(f"episodios: {len(files)} | frames: {tot} | dt mediana {med:.3f} s -> fps nominal {1/med:.1f} | fps medio {1/dts.mean():.2f}")
print(f"joints por frame: {sorted(shapes)} | tasks: {sorted(tasks)}")
print(f"calibraciones (range_min,range_max) distintas: {len(ranges)}"); print(" ", list(ranges)[0])
print(f"episodios con huecos (dt > 1.5x mediana): {len(gaps)} de {len(files)}")
for g in gaps: print("  ", g[0], f"{g[1]} hueco(s), {g[2]} s")

idx = sorted(set(np.linspace(0, len(files) - 1, min(a.sheet_episodes, len(files))).astype(int).tolist()))
rows = []
for i in idx:
    im = pq.read_table(files[i], columns=["images"]).to_pydict()["images"]; fr = im[len(im) // 2]
    rows.append((os.path.basename(files[i]), [Image.open(io.BytesIO(x["jpeg"])).convert("RGB").resize((224, 224)) for x in fr]))
ncol = max(len(r[1]) for r in rows); sheet = Image.new("RGB", (224 * ncol, 224 * len(rows)), "white")
for r, (name, ims) in enumerate(rows):
    for c, im in enumerate(ims):
        sheet.paste(im, (224 * c, 224 * r)); ImageDraw.Draw(sheet).text((224 * c + 4, 224 * r + 4), f"{name[:12]} images[{c}]", fill=(255, 0, 0))
sheet.save(a.sheet); print(f"\nlámina de cámaras: {os.path.abspath(a.sheet)}  (filas = episodios, columnas = images[i])")
