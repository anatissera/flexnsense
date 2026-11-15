#!/usr/bin/env python
r"""Parquet de NormaCore Station (un fichero por episodio) -> UN Parquet con el mismo esquema
y las cámaras siempre en el mismo orden. Es la entrada del fine-tune de SmolVLA de NormaCore
(norma-core/software/ai/smolvla_py), que lee `images[i]` por índice.

    python normalize_cameras.py \
        --src /ruta/al/dataset_normacore \          # contiene data/<task>/episode_XXXX.parquet
        --output /ruta/flexnsense_skill1_canon.parquet \
        --ref-views side,gripper,table              # vistas de images[0..2] en el episodio 1

Por qué: Station no etiqueta las imágenes y el orden de `images[]` cambia entre sesiones
(ver README, paso 1.2). Aquí cada episodio se reordena por contenido de imagen, con el mismo
clasificador que convert_normacore.py; si un episodio es ambiguo, aborta.

La salida tiene images[] = --out-views (por defecto side,gripper,table) en TODOS los frames.
Los JPEG no se recodifican. Los huecos de grabación no se tocan: el dataset de smolvla_py usa
filas consecutivas, así que quedan colapsados igual que en el LeRobotDataset.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from convert_normacore import DEFAULT_VIEWS, N_JOINTS, build_reference, classify_slots, list_episodes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="raíz del dataset NormaCore (contiene data/<task>/*.parquet)")
    ap.add_argument("--output", required=True, type=Path, help="Parquet de salida")
    ap.add_argument("--ref-episode", type=int, default=0, help="índice (0-based) del episodio de referencia")
    ap.add_argument("--ref-views", default=",".join(DEFAULT_VIEWS),
                    help="vista de images[0],images[1],... en el episodio de referencia")
    ap.add_argument("--out-views", default=",".join(DEFAULT_VIEWS), help="orden de images[] en la salida")
    args = ap.parse_args()

    views = args.ref_views.split(",")
    out_views = args.out_views.split(",")
    if sorted(views) != sorted(out_views):
        raise SystemExit(f"--out-views {out_views} debe tener las mismas vistas que --ref-views {views}")

    files = list_episodes(args.src)
    ref = build_reference(pq.read_table(files[args.ref_episode], columns=["images"]).to_pydict()["images"])
    if len(ref) != len(views):
        raise SystemExit(f"el episodio de referencia tiene {len(ref)} imágenes y --ref-views {len(views)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer, seen_eps, total = None, set(), 0
    try:
        for f in files:
            table = pq.read_table(f)
            images = table.column("images").to_pylist()
            n_joints = {len(r) for r in table.column("joints").to_pylist()}
            if n_joints != {N_JOINTS} or {len(r) for r in images} != {len(views)}:
                raise RuntimeError(f"{f}: joints por frame {n_joints}, se esperaban {N_JOINTS} y {len(views)} imágenes")
            eps = set(table.column("episode_start_ns").to_pylist())
            if len(eps) != 1 or eps & seen_eps:
                raise RuntimeError(f"{f}: episode_start_ns {eps} no es único (smolvla_py separa episodios con él)")
            seen_eps |= eps
            ts = table.column("timestamp_ns_since_episode_start").to_numpy()
            if (np.diff(ts.astype(np.int64)) <= 0).any():
                raise RuntimeError(f"{f}: los frames no están en orden temporal")

            perm, margin = classify_slots(images, ref)               # perm[slot] = índice en `views`
            slot_of = {views[v]: s for s, v in enumerate(perm)}
            order = [slot_of[v] for v in out_views]
            col = pa.array([[row[s] for s in order] for row in images], type=table.schema.field("images").type)
            table = table.set_column(table.schema.get_field_index("images"), table.schema.field("images"), col)

            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema, compression="snappy")
            writer.write_table(table)
            total += table.num_rows
            print(f"{Path(f).stem}: {table.num_rows} frames, images[] era {[views[v] for v in perm]} (margen {margin:.2f})")
    finally:
        if writer is not None:
            writer.close()

    print(f"\nOK -> {args.output}  ({args.output.stat().st_size / 1e6:.0f} MB)")
    print(f"  {len(files)} episodios, {total} frames, images[] = {out_views}")


if __name__ == "__main__":
    main()
