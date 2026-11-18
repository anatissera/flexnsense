#!/usr/bin/env python
r"""Fine-tune de SmolVLA para el ElRobot con el entrenamiento estándar de NormaCore.

Ejecuta tal cual `norma-core/software/ai/smolvla_py/scripts/train.py` (mismo bucle, mismo
optimizador, mismos checkpoints) cambiando solo lo que ese script trae fijo para el SO-101:

  * 6 joints            -> los del dataset (8 en el ElRobot)
  * 2 cámaras           -> --cameras (3: side, gripper, table)
  * suelo de la std     -> --std-floor (0.02). Los motores 5 y 7 casi no se mueven en
    flexnsense_skill1 (std 0.012 y 0.003); sin suelo, la normalización amplifica el ruido
    del encoder en la entrada. Con --std-floor 1e-4 se obtiene el comportamiento original.

Las opciones que no son de este script se pasan a train.py (--steps, --batch-size, --lr,
--parquets, --output, --save-every, --num-workers...):

    cd $NORMA_CORE/software/ai/smolvla_py
    uv run python /ruta/robot_training/train_smolvla.py \
        --steps 30000 --batch-size 64 --save-every 5000 --num-workers 8 \
        --parquets /ruta/flexnsense_skill1_canon.parquet --output /ruta/checkpoints/<RUN>

El Parquet debe venir de normalize_cameras.py (cámaras en orden fijo). En cada checkpoint se
guarda además `elrobot.json` con las vistas y dimensiones, que es lo que lee la inferencia.
"""

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smolvla-py", default=None,
                    help="carpeta smolvla_py de norma-core (por defecto $NORMA_CORE/software/ai/smolvla_py)")
    ap.add_argument("--cameras", default="side,gripper,table",
                    help="vistas de images[0],[1],... en el Parquet (el --out-views de normalize_cameras.py)")
    ap.add_argument("--std-floor", type=float, default=0.02)
    args, rest = ap.parse_known_args()

    root = Path(args.smolvla_py or Path(os.environ.get("NORMA_CORE", "")) / "software/ai/smolvla_py")
    if not (root / "scripts" / "train.py").exists():
        raise SystemExit(f"no encuentro scripts/train.py en {root} (usa --smolvla-py o define NORMA_CORE)")
    sys.path[:0] = [str(root), str(root / "scripts")]
    import train  # el train.py de NormaCore

    sys.argv = [str(root / "scripts" / "train.py"), *rest]
    targs = train.parse_args()
    views = args.cameras.split(",")
    joints = pq.ParquetFile(targs.parquets[0]).read_row_group(0, columns=["joints", "images", "task"]).slice(0, 1).to_pylist()[0]
    n_joints, n_images = len(joints["joints"]), len(joints["images"])
    if n_images != len(views):
        raise SystemExit(f"el Parquet tiene {n_images} imágenes por frame y --cameras {len(views)}")

    train.STATE_DIM = train.ACTION_DIM = n_joints
    train.IMAGE_KEYS = tuple(f"observation.images.cam{i}" for i in range(len(views)))
    train.compute_stats = partial(train.compute_stats, eps=args.std_floor)

    meta = {"robot": "elrobot", "views": views, "image_keys": list(train.IMAGE_KEYS),
            "state_dim": n_joints, "action_dim": n_joints, "std_floor": args.std_floor,
            "task": joints["task"],   # frase por defecto en inferencia (primer frame del dataset)
            "parquets": [str(p) for p in targs.parquets], "train_args": rest}
    save_pretrained = train.SmolVLAPolicy.save_pretrained

    def save_with_meta(self, save_directory):
        save_pretrained(self, save_directory)
        (Path(save_directory) / "elrobot.json").write_text(json.dumps(meta, indent=2))

    train.SmolVLAPolicy.save_pretrained = save_with_meta
    print(f"ElRobot: {n_joints} joints, cámaras {dict(zip(train.IMAGE_KEYS, views))}, std floor {args.std_floor}")
    train.main()


if __name__ == "__main__":
    main()
