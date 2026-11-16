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

  * --hand: añade a observation.state las yemas del dedo origen y del dedo destino en la cámara
    `side` (5 valores, ver hand_features.py). Los dedos salen de la frase de cada episodio, así
    que se pueden mezclar tareas (índice -> corazón, corazón -> anular) en un mismo dataset. `dataset` usa las columnas hand_* que graba el script de
    data collection (WiLoR); `mediapipe` las recalcula aquí con MediaPipe. En inferencia hay que
    usar la misma fuente (--hand-source).

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

from hand_features import HAND_NAMES, HAND_VIEW, HandFill, draw_marks, pick_tips, task_fingers

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
    ap.add_argument("--hand", choices=["none", "dataset", "mediapipe"], default="none",
                    help="añadir las yemas de los dedos a observation.state: columnas hand_* del dataset "
                         "(WiLoR) o recalculadas con MediaPipe")
    ap.add_argument("--hand-marks", action="store_true",
                    help="con --hand: dibujar además las yemas origen y destino sobre la imagen side "
                         "(entrenar con HAND_MARKS=1 para que el checkpoint lo recuerde)")
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

    use_hand = args.hand != "none"
    if args.hand_marks and not use_hand:
        raise SystemExit("--hand-marks necesita --hand dataset o --hand mediapipe")
    if use_hand and HAND_VIEW not in views:
        raise SystemExit(f"--hand necesita la vista '{HAND_VIEW}' en --ref-views")
    state_names = MOTOR_NAMES + (HAND_NAMES if use_hand else [])
    hand_tips = None
    if args.hand == "mediapipe":
        from hand_features import HandTips
        hand_tips = HandTips()
    fill, hand_frames, hand_found = HandFill(), 0, 0
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
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
        if args.hand == "dataset":
            if "hand_fingertips_2d" not in t:
                raise SystemExit(f"{f} no tiene columnas hand_*: usa --hand mediapipe o vuelve a anotar el dataset")
            # hand_camera = índice de images[] sobre el que se calcularon: tiene que ser la vista side
            if set(t["hand_camera"]) != {view_to_slot[HAND_VIEW]}:
                raise RuntimeError(f"{f}: hand_camera={set(t['hand_camera'])} pero '{HAND_VIEW}' es "
                                   f"images[{view_to_slot[HAND_VIEW]}]: cámaras mal asignadas o mano de otra cámara")
        fill.reset()

        for i in range(n):
            joints = t["joints"][i]
            if len(joints) != N_JOINTS or len(t["images"][i]) != len(views):
                raise RuntimeError(f"{f} frame {i}: {len(joints)} joints / {len(t['images'][i])} imágenes")
            state = np.array([j["position_norm"] for j in joints], dtype=np.float32)
            frame = {
                "action": np.array([j["goal_norm"] for j in joints], dtype=np.float32),
                "task": t["task"][i],
            }
            for v, s in view_to_slot.items():
                frame[f"observation.images.{v}"] = np.asarray(
                    Image.open(io.BytesIO(t["images"][i][s]["jpeg"])).convert("RGB"))
            if use_hand:
                if args.hand == "dataset":       # hand_fingertips_2d = pulgar, índice, corazón, anular, meñique (x, y)
                    tips = t["hand_fingertips_2d"][i] if t["hand_detected"][i] else None
                else:
                    tips = hand_tips(frame[f"observation.images.{HAND_VIEW}"])
                tips = pick_tips(tips, t["task"][i])
                hand_frames += 1
                hand_found += tips is not None
                vec = fill.step(tips)
                state = np.concatenate([state, vec])
                if args.hand_marks and fill.seen:
                    key = f"observation.images.{HAND_VIEW}"
                    frame[key] = draw_marks(frame[key], vec)
            frame["observation.state"] = state
            ds.add_frame(frame)
        ds.save_episode()
        total += n
        total_gaps += gaps
        tasks |= set(t["task"])
        fingers = f", dedos {task_fingers(t['task'][0])}" if use_hand else ""
        print(f"{Path(f).parent.name[-28:]}/{Path(f).stem}: {n} frames, images[] = {[views[v] for v in perm]} "
              f"(margen {margin:.2f}), huecos {gaps}{fingers}")

    ds.finalize()
    print(f"\nOK -> {root}\n  {len(files)} episodios, {total} frames, {args.fps} fps, {total_gaps} huecos colapsados")
    print(f"  tasks: {sorted(tasks)}\n  cámaras: {[f'observation.images.{v}' for v in views]}")
    if use_hand:
        print(f"  mano ({args.hand}): detectada en {hand_found}/{hand_frames} frames ({100 * hand_found / hand_frames:.1f} %); "
              f"observation.state = {len(state_names)} valores"
              f"{', marcas en la imagen ' + HAND_VIEW if args.hand_marks else ''}. En inferencia: --hand-source "
              f"{'wilor' if args.hand == 'dataset' else 'mediapipe'}")


if __name__ == "__main__":
    main()
