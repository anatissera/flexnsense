#!/usr/bin/env python
r"""Error en episodios que el entrenamiento no vio, para cada checkpoint de un run.

    python eval_holdout.py --src /ruta/dataset_normacore --holdout holdout.json \
        --run /ruta/checkpoints/<RUN> [--out resultado.json]

holdout.json = {"episodes": [índices 0-based en el orden de convert_normacore.py], ...}; se
entrena con EPISODES=<el resto> (train_act.sh). Usa el mismo camino que la inferencia real
(inference/policy_runner.py, JPEG -> chunk, yemas y marcas incluidas) con las columnas hand_*
grabadas. Para cada frame mirado compara las primeras acciones del chunk con los goals que
grabó el operador y da el error medio en ticks. Sirve para ver si el modelo converge y para
elegir checkpoint; no sustituye a la prueba en el brazo.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "inference"))

from convert_normacore import DEFAULT_VIEWS, build_reference, classify_slots, list_episodes  # noqa: E402
from hand_features import HAND_VIEW, HandFill, mark_jpeg, pick_tips  # noqa: E402


def load_episode(f, ref, views):
    t = pq.read_table(f).to_pydict()
    perm, _ = classify_slots(t["images"], ref)
    slot = {views[v]: s for s, v in enumerate(perm)}
    return t, slot


def evaluate(runner, episodes, views, step, horizon):
    per_task = {}
    for t, slot in episodes:
        n = len(t["task"])
        task = t["task"][0]
        goal = np.array([[j["goal_norm"] for j in row] for row in t["joints"]], np.float32)
        span = np.array([j["range_max"] - j["range_min"] for j in t["joints"][0]], np.float32)
        fill, errs = HandFill(), []
        for i in range(n):
            vec = None
            if runner.state_dim > 8:        # el relleno causal hay que avanzarlo en todos los frames
                vec = fill.step(pick_tips(t["hand_fingertips_2d"][i] if t["hand_detected"][i] else None, task))
            if i % step:
                continue
            state = np.array([j["position_norm"] for j in t["joints"][i]], np.float32)
            jpegs = {v: t["images"][i][slot[v]]["jpeg"] for v in views}
            if vec is not None:
                state = np.concatenate([state, vec])
                if runner.hand_marks and fill.seen:
                    jpegs[HAND_VIEW] = mark_jpeg(jpegs[HAND_VIEW], vec)
            chunk, _ = runner.predict_from_jpegs(state, jpegs, task)
            k = min(horizon, n - i, len(chunk))
            errs.append(np.abs(chunk[:k] - goal[i:i + k]).mean(0) * span)
        per_task.setdefault(task, []).append(np.mean(errs, 0))
    return {task: np.mean(v, 0) for task, v in per_task.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--run", required=True, help="carpeta del run (contiene checkpoints/<paso>/pretrained_model)")
    ap.add_argument("--step", type=int, default=5, help="mirar 1 de cada N frames")
    ap.add_argument("--horizon", type=int, default=15, help="acciones del chunk que se comparan")
    ap.add_argument("--ref-views", default=",".join(DEFAULT_VIEWS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from policy_runner import load_runner

    views = a.ref_views.split(",")
    files = list_episodes(a.src)
    ref = build_reference(pq.read_table(files[0], columns=["images"]).to_pydict()["images"])
    held = json.loads(Path(a.holdout).read_text())["episodes"]
    episodes = [load_episode(files[i], ref, views) for i in held]
    print(f"{len(episodes)} episodios de validación, 1 de cada {a.step} frames, {a.horizon} acciones por chunk")

    result = {}
    for ck in sorted(p for p in (Path(a.run) / "checkpoints").iterdir() if p.is_dir() and not p.is_symlink()):
        runner = load_runner(ck / "pretrained_model")
        per_task = evaluate(runner, episodes, views, a.step, a.horizon)
        mean = float(np.mean([v.mean() for v in per_task.values()]))
        result[ck.name] = {"mean_ticks": mean, "per_task": {t: float(v.mean()) for t, v in per_task.items()},
                           "per_motor": np.mean(list(per_task.values()), 0).round(1).tolist()}
        short = "  ".join(f"{t.split('from ')[-1]}: {v.mean():.1f}" for t, v in per_task.items())
        print(f"{ck.name}: {mean:6.1f} ticks  | {short}")
        del runner
    best = min(result, key=lambda k: result[k]["mean_ticks"])
    print(f"\nmejor checkpoint: {best} ({result[best]['mean_ticks']:.1f} ticks de error medio)")
    if a.out:
        Path(a.out).write_text(json.dumps({"best": best, "checkpoints": result}, indent=2))


if __name__ == "__main__":
    main()
