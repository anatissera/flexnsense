#!/usr/bin/env python
r"""Conversor: Parquet de NormaCore Station (ElRobot) -> LeRobotDataset (LeRobot 0.7.x, formato v3).

    python convert_normacore.py \
        --src /ruta/al/dataset_normacore \          # contiene data/<task>/episode_XXXX.parquet
        --repo-id <hf_user>/<nombre>_lerobot

Qué hace (decisiones verificadas con el dataset flexnsense_skill1, 43 episodios):
  * fps = 10 (--fps): Station publica inference/normvla cada 100 ms. Si la grabación tiene
    huecos, se colapsan: los frames se guardan consecutivos (el script los cuenta e informa).
  * observation.state = position_norm (8), action = goal_norm (8), ordenados por motor ID 1..8
    (convención de NormaCore; el ID 8 es el gripper).
  * Cámaras: el orden de `images[]` NO es estable entre episodios (depende del orden en que
    Station arranca las cámaras USB) y las imágenes no llevan ID. Las vistas se definen con un
    episodio de referencia (--ref-episode, --ref-views) y cada slot de cada episodio se asigna
    a una vista por similitud de imagen. Si un episodio es ambiguo, el script aborta.

IMPORTANTE: mira a ojo el episodio de referencia y pasa en --ref-views el nombre de la vista
de images[0], images[1], images[2] de ESE episodio (por defecto: side,gripper,table).
"""

import argparse
import glob
import io
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

N_JOINTS = 8
DEFAULT_VIEWS = ("side", "gripper", "table")
MOTOR_NAMES = [f"motor_{i}.pos" for i in range(1, N_JOINTS + 1)]  # motor_8 = gripper


def _feat(jpeg: bytes) -> np.ndarray:
    a = np.asarray(Image.open(io.BytesIO(jpeg)).convert("L").resize((32, 32)), dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def _slot_means(images, step=5):
    n_slots = len(images[0])
    idxs = range(0, len(images), step)
    return [np.mean([_feat(images[i][s]["jpeg"]) for i in idxs], 0) for s in range(n_slots)]


def build_reference(images):
    return _slot_means(images, step=10)


def classify_slots(images, ref, min_margin=0.02):
    """Devuelve (perm, margen) con perm[slot] = índice de vista. Aborta si es ambiguo."""
    m = _slot_means(images)
    n = len(ref)
    sim = np.array([[float(np.dot(m[s], ref[v])) / len(ref[v]) for v in range(n)] for s in range(n)])
    perm = tuple(int(x) for x in sim.argmax(1))
    srt = np.sort(sim, 1)
    margin = float((srt[:, -1] - srt[:, -2]).min())
    if sorted(perm) != list(range(n)) or margin < min_margin:
        raise RuntimeError(f"Asignación de cámaras ambigua: perm={perm} margin={margin:.3f}\n{sim}")
    return perm, margin


def list_episodes(src: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(src, "data", "*", "*.parquet")))
    if not files:
        raise SystemExit(f"no hay parquet en {src}/data/*/")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="raíz del dataset NormaCore (contiene data/<task>/*.parquet)")
    ap.add_argument("--repo-id", required=True, help="nombre del LeRobotDataset, p.ej. <hf_user>/<nombre>_lerobot")
    ap.add_argument("--root", default=None, help="destino (por defecto $HF_LEROBOT_HOME/<repo-id>)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--ref-episode", type=int, default=0, help="índice (0-based) del episodio de referencia de cámaras")
    ap.add_argument("--ref-views", default=",".join(DEFAULT_VIEWS),
                    help="nombre de la vista de images[0],images[1],... en el episodio de referencia")
    ap.add_argument("--robot-type", default="elrobot")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # import perezoso (pesado)

    views = args.ref_views.split(",")
    files = list_episodes(args.src)
    print(f"{len(files)} episodios en {args.src}")

    ref_images = pq.read_table(files[args.ref_episode], columns=["images"]).to_pydict()["images"]
    if len(ref_images[0]) != len(views):
        raise SystemExit(f"el episodio de referencia tiene {len(ref_images[0])} imágenes y --ref-views {len(views)}")
    with Image.open(io.BytesIO(ref_images[0][0]["jpeg"])) as im:
        w, h = im.size

    features = {
        "observation.state": {"dtype": "float32", "shape": (N_JOINTS,), "names": MOTOR_NAMES},
        "action": {"dtype": "float32", "shape": (N_JOINTS,), "names": MOTOR_NAMES},
    }
    for v in views:
        features[f"observation.images.{v}"] = {
            "dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}

    if args.root:
        root = Path(args.root)
    else:
        from lerobot.utils.constants import HF_LEROBOT_HOME
        root = HF_LEROBOT_HOME / args.repo_id
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} ya existe (usa --overwrite)")
        shutil.rmtree(root)

    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, features=features, root=root,
                               robot_type=args.robot_type, use_videos=True, image_writer_threads=8)
    ref = build_reference(ref_images)

    total, total_gaps, tasks = 0, 0, set()
    for f in files:
        t = pq.read_table(f).to_pydict()
        n = len(t["timestamp_ns_since_episode_start"])
        perm, margin = classify_slots(t["images"], ref)
        view_to_slot = {views[v]: s for s, v in enumerate(perm)}
        ts = np.asarray(t["timestamp_ns_since_episode_start"], dtype=np.int64) / 1e9
        gaps = int((np.diff(ts) > 1.5 / args.fps).sum())

        for i in range(n):
            joints = t["joints"][i]
            if len(joints) != N_JOINTS or len(t["images"][i]) != len(views):
                raise RuntimeError(f"{f} frame {i}: {len(joints)} joints / {len(t['images'][i])} imágenes")
            frame = {
                "observation.state": np.array([j["position_norm"] for j in joints], dtype=np.float32),
                "action": np.array([j["goal_norm"] for j in joints], dtype=np.float32),
                "task": t["task"][i],
            }
            for v, s in view_to_slot.items():
                frame[f"observation.images.{v}"] = np.asarray(
                    Image.open(io.BytesIO(t["images"][i][s]["jpeg"])).convert("RGB"))
            ds.add_frame(frame)
        ds.save_episode()
        total += n
        total_gaps += gaps
        tasks |= set(t["task"])
        print(f"{Path(f).stem}: {n} frames, images[] = {[views[v] for v in perm]} (margen {margin:.2f}), huecos {gaps}")

    ds.finalize()
    print(f"\nOK -> {root}\n  {len(files)} episodios, {total} frames, {args.fps} fps, {total_gaps} huecos colapsados")
    print(f"  tasks: {sorted(tasks)}\n  cámaras: {[f'observation.images.{v}' for v in views]}")


if __name__ == "__main__":
    main()
